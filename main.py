import os
import csv
import cv2
import copy
import tqdm
import yaml
import torch
import argparse
import warnings
import numpy as np
from PIL import Image, ImageDraw
from torch.utils import data
from torch import distributed as dist
from torch.nn.utils import clip_grad_norm_ as clip
from torch.nn.parallel import DistributedDataParallel

from nets import nn
from utils import util
from utils.dataset import Dataset


def train(args, params):
    util.init_seeds()
    device = params['device']
    model = nn.return_model_definition(args.model_size, args.num_cls, params['num_channels'], args)
    ckpt = torch.load(args.weights_path, map_location=device)
    state_dict = ckpt['model']
    
    # Keep only keys that match name AND shape
    filtered_state = {}
    for k, v in state_dict.items():
        if k in model.state_dict():
            if v.shape == model.state_dict()[k].shape:
                filtered_state[k] = v
            else:
                print(f"⚠️ Skipping {k}: shape mismatch {v.shape} vs {model.state_dict()[k].shape}")
        else:
            print(f"❌ Skipping {k}: not in current model")

    # Update only the compatible parameters
    model.state_dict().update(filtered_state)
    model.load_state_dict(model.state_dict())
    print(f"✅ Loaded {len(filtered_state)} / {len(model.state_dict())} compatible layers")

    with torch.no_grad():
        w = state_dict["backbone.p1.0.conv.weight"]  # old input conv weights
        new_w = model.state_dict()["backbone.p1.0.conv.weight"]   # new shape (out, 33, h, w)

        # e.g., repeat RGB weights across 33 channels
        repeat_factor = new_w.shape[1] // w.shape[1]
        new_w[:, :w.shape[1]*repeat_factor] = w.repeat(1, repeat_factor, 1, 1)
        model.state_dict()["backbone.p1.0.conv.weight"] = new_w



    # ckpt = torch.load('yolo11x_pretrained.pt', map_location='cpu')
    # keys_pretr = ckpt['model'].state_dict().keys()
    # keys_new = model.state_dict().keys()
    # mapping = dict(zip(keys_pretr, keys_new))
    # state_dict = ckpt["model"].state_dict()
    # new_state_dict = {mapping.get(k, k): v for k, v in state_dict.items()}
    # model.load_state_dict(new_state_dict, strict=False)
    # torch.save({"model": new_state_dict}, "yolo11_remapped.pt")

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    if args.distributed:
        util.setup_ddp(args)

    # Freeze DFL Layer
    util.freeze_layer(model)


    scaler = torch.amp.GradScaler(device=device, enabled=True)
    # DDP setup
    if args.distributed:
        model = DistributedDataParallel(module=model,
                                        device_ids=[args.rank],
                                        find_unused_parameters=True)

    ema = util.EMA(model) if args.rank == 0 else None

    sampler = None
    dataset = Dataset(args, params, True)
    if args.distributed:
        sampler = data.distributed.DistributedSampler(dataset)

    batch_size = args.batch_size // max(args.world_size, 1)
    loader = data.DataLoader(dataset, batch_size, sampler is None,
                             sampler, num_workers=8, pin_memory=True,
                             collate_fn=Dataset.collate_fn)

    accumulate = max(round(64 / args.batch_size * args.world_size), 1)
    decay = params['decay'] * args.batch_size * accumulate / 64
    optimizer = util.smart_optimizer(args, model, decay)
    linear = lambda x: (max(1 - x / args.epochs, 0) * (1.0 - 0.01) + 0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=linear)
    scheduler.last_epoch = - 1
    # criterion = util.DetectionLoss(model)
    criterion = util.DetectionLoss(model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model)



    opt_step = -1
    num_batch = len(loader)
    warm_up = max(round(3 * num_batch), 100)

    best_map = 0.0

    with open('weights/step.csv', 'w') as log:
        if args.rank == 0:
            if args.redundancy:
                logger = csv.DictWriter(log, fieldnames=['epoch',
                                                     'box', 'cls', 'dfl', 'redundancy_loss',
                                                'Recall', 'Precision', 'mAP@50', 'mAP'])
            else:
                logger = csv.DictWriter(log, fieldnames=['epoch',
                                                     'box', 'cls', 'dfl',
                                                     'Recall', 'Precision', 'mAP@50', 'mAP'])
            logger.writeheader()
        for epoch in range(args.epochs):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scheduler.step()

            model.train()
            if args.distributed:
                sampler.set_epoch(epoch)

            p_bar = enumerate(loader)
            if args.epochs - epoch == 10:
                loader.dataset.mosaic = False

            if args.rank == 0:
                if args.redundancy:
                    print("\n" + "%11s" * 6 % ("Epoch", "GPU", "box", "cls", "dfl", "rdt"))
                else:
                    print("\n" + "%11s" * 5 % ("Epoch", "GPU", "box", "cls", "dfl"))
                p_bar = tqdm.tqdm(enumerate(loader), total=num_batch)

            t_loss = None
            for i, batch in p_bar:
                glob_step = i + num_batch * epoch
                if glob_step <= warm_up:
                    xi = [0, warm_up]
                    accumulate = max(1, int(np.interp(glob_step, xi, [1, 64 / args.batch_size]).round()))
                    for j, x in enumerate(optimizer.param_groups):
                        x["lr"] = np.interp(glob_step, xi, [0.0 if j == 0 else 0.0,
                                                            x["initial_lr"] * linear(epoch)])

                        if "momentum" in x:
                            x["momentum"] = np.interp(glob_step, xi, [0.8, 0.937])

                
                images = batch["img"].to(device).float()


                # Use autocast only when CUDA is available; when disabled it's a no-op
                with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                    pred = model(images)
                    loss, loss_items = criterion(pred, batch)
                    if args.distributed:
                        loss *= args.world_size

                    t_loss = ((t_loss * i + loss_items) / (
                                i + 1) if t_loss is not None else loss_items)

                scaler.scale(loss).backward()
                if glob_step - opt_step >= accumulate:
                    scaler.unscale_(optimizer)
                    clip(model.parameters(), max_norm=10.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    if ema:
                        ema.update(model)
                    opt_step = glob_step

                if args.rank == 0:
                    fmt = "%11s" * 2 + "%11.4g" * 3
                    if torch.cuda.is_available():
                        memory = f'{torch.cuda.memory_reserved() / 1e9:.3g}G'
                    else:
                        memory = '0G'  # or 'CPU' if you prefer
                    if args.redundancy:
                        fmt = fmt = "%11s%11s" + "%11.4g" * 4
                    p_bar.set_description(fmt % (f"{epoch + 1}/{args.epochs}", memory, *t_loss))

            if args.rank == 0:
                
                if args.redundancy:
                    m_pre, m_rec, map50, mean_map, acc, prec, recall, f1 = validate(args, params, ema.ema)

                    box, cls, dfl, redundancy_loss = map(float, t_loss)
                    logger.writerow({'epoch': str(epoch + 1).zfill(3),
                                 'box': str(f'{box:.3f}'),
                                 'cls': str(f'{cls:.3f}'),
                                 'dfl': str(f'{dfl:.3f}'),
                                 'redundancy_loss': str(f'{redundancy_loss:.3f}'),
                                 'mAP': str(f'{mean_map:.3f}'),
                                 'mAP@50': str(f'{map50:.3f}'),
                                 'Recall': str(f'{m_rec:.3f}'),
                                 'Precision': str(f'{m_pre:.3f}')})
                else:
                    m_pre, m_rec, map50, mean_map, _, _, _, _ = validate(args, params, ema.ema)

                    box, cls, dfl = map(float, t_loss)

                    logger.writerow({'epoch': str(epoch + 1).zfill(3),
                                    'box': str(f'{box:.3f}'),
                                    'cls': str(f'{cls:.3f}'),
                                    'dfl': str(f'{dfl:.3f}'),
                                    'mAP': str(f'{mean_map:.3f}'),
                                    'mAP@50': str(f'{map50:.3f}'),
                                    'Recall': str(f'{m_rec:.3f}'),
                                    'Precision': str(f'{m_pre:.3f}')})
                log.flush()

                ckpt = {'epoch': epoch+1, 'model': copy.deepcopy(ema.ema)}
                torch.save(ckpt, f'weights/epoch_{epoch+1}.pt')

                if mean_map > best_map:
                    best_map = mean_map
                    torch.save(ckpt, 'weights/best.pt')

                del ckpt

            if args.distributed:
                dist.barrier()

        if args.distributed:
            dist.destroy_process_group()

        print("Training complete.")


def validate(args, params, model=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    iou_v = torch.linspace(0.5, 0.95, 10)
    n_iou = iou_v.numel()

    metric = {"tp": [], "conf": [], "pred_cls": [], "target_cls": [], "target_img": []}
    redund_metrics = {"pred": [], "target": []}  # <-- NEW

    if not model:
        args.plot = True
        model = torch.load(f='weights/best.pt', map_location=device)
        model = model['model'].float().fuse()

    # model.half()
    model.eval()
    dataset = Dataset(args, params, False)
    loader = data.DataLoader(dataset, batch_size=16,
                             shuffle=False, num_workers=4,
                             pin_memory=True, collate_fn=Dataset.collate_fn)

    for batch in tqdm.tqdm(loader, desc=('%10s' * 5) % (
    '', 'precision', 'recall', 'mAP50', 'mAP')):
        image = (batch["img"].to(device).float())

        if args.redundancy:
            for k in ["idx", "cls", "box", "redundant"]:
                batch[k] = batch[k].to(device)
        else:
            for k in ["idx", "cls", "box"]:
                batch[k] = batch[k].to(device)

        outputs = util.non_max_suppression(model(image), redundancy = args.redundancy)

        metric = util.update_metrics(outputs, batch, n_iou, iou_v, metric)

        if args.redundancy and "redundant" in batch:
            gt_redund = batch["redundant"].to(device).float().view(-1)
            preds = []
            for det in outputs:
                if det.shape[0] and det.shape[1] >= 7:  # last col = redundancy score
                    preds.append(det[:, -1])
            if len(preds):
                pred_redund = torch.cat(preds)
                gt_trimmed = gt_redund[:len(pred_redund)]  # align counts
                redund_metrics["pred"].append(pred_redund.cpu())
                redund_metrics["target"].append(gt_trimmed.cpu())

    stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in metric.items()}
    stats.pop("target_img", None)
    if len(stats) and stats["tp"].any():
        result = util.compute_ap(tp=stats['tp'],
                                 conf=stats['conf'],
                                 pred=stats['pred_cls'],
                                 target=stats['target_cls'],
                                 plot=args.plot,
                                 save_dir='weights/',
                                 names=params['names'])

        m_pre = result['precision']
        m_rec = result['recall']
        map50 = result['mAP50']
        mean_ap = result['mAP50-95']
    else:
        m_pre, m_rec, map50, mean_ap = 0.0, 0.0, 0.0, 0.0

    print(('%10s' + '%10.3g' * 4) % ('', m_pre, m_rec, map50, mean_ap))

    if "redund_pred" in metric and len(metric["redund_pred"]):
        y_pred = torch.cat(metric["redund_pred"])
        y_true = torch.cat(metric["redund_true"])

        print("Mean y_pred:", y_pred.mean().item())
        print("Fraction of positives in truth:", (y_true > 0.5).float().mean().item())
        print("Fraction of predicted positives:", (y_pred > 0.5).float().mean().item())

        y_bin = (y_pred > 0.5).float()

        acc = (y_bin == y_true).float().mean().item()
        tp = ((y_bin == 1) & (y_true == 1)).sum().item()
        fp = ((y_bin == 1) & (y_true == 0)).sum().item()
        fn = ((y_bin == 0) & (y_true == 1)).sum().item()

        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)

        print(f"Redundancy  Acc: {acc:.3f}  Prec: {prec:.3f}  Rec: {rec:.3f}  F1: {f1:.3f}")
    else:
        acc = prec = rec = f1 = 0.0

    model.float()

    return m_pre, m_rec, map50, mean_ap, acc, prec, rec, f1

@torch.no_grad()
def inference(args, params):
    #TODO: refactor to work with input with more channels and actual input
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model = nn.return_model_definition(args.model_size, args.num_cls, params['num_channels'], args)
    ckpt = torch.load('yolo11x_remapped.pt', map_location=device)

    state_dict = ckpt['model']

        # Keep only keys that match name AND shape
    filtered_state = {}
    for k, v in state_dict.items():
        if k in model.state_dict():
            if v.shape == model.state_dict()[k].shape:
                filtered_state[k] = v
            else:
                print(f"⚠️ Skipping {k}: shape mismatch {v.shape} vs {model.state_dict()[k].shape}")
        else:
            print(f"❌ Skipping {k}: not in current model")

    # Update only the compatible parameters
    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)
    print(f"✅ Loaded {len(filtered_state)} / {len(model.state_dict())} compatible layers")

    # with torch.no_grad():
    #     w = state_dict["backbone.p1.0.conv.weight"]  # old input conv weights
    #     new_w = model.state_dict()["backbone.p1.0.conv.weight"]   # new shape (out, 33, h, w)

    #     # e.g., repeat RGB weights across 33 channels
    #     repeat_factor = new_w.shape[1] // w.shape[1]
    #     new_w[:, :w.shape[1]*repeat_factor] = w.repeat(1, repeat_factor, 1, 1)
    #     model.state_dict()["backbone.p1.0.conv.weight"] = new_w

    model.to(device)
    model.eval()
    dataset = Dataset(args, params, False)
    loader = data.DataLoader(dataset, batch_size=16,
                             shuffle=False, num_workers=4,
                             pin_memory=True, collate_fn=Dataset.collate_fn)

    for batch in tqdm.tqdm(loader, desc=('%10s' * 5) % (
    '', 'precision', 'recall', 'mAP50', 'mAP')):
        image = (batch["img"].to(device).float())

        for k in ["idx", "cls", "box"]:
            batch[k] = batch[k].to(device)
        shape = image.shape[2:]
        height, width = image.shape[2:]

        outputs, _ = model(image)

        # NMS
        batch_outputs = util.non_max_suppression(outputs, 0.15, 0.2, redundancy = args.redundancy)
       
        for det, img, filename in zip(batch_outputs, image, batch['image']):
            img_to_draw = (img.cpu().numpy()*255).astype(np.uint8)
            pil_img = Image.fromarray(img_to_draw.transpose(1, 2, 0))
            draw = ImageDraw.Draw(pil_img)

            if det is not None:
                det[:, :4] /= min(height / shape[0], width / shape[1])

                det[:, 0].clamp_(0, shape[1])
                det[:, 1].clamp_(0, shape[0])
                det[:, 2].clamp_(0, shape[1])
                det[:, 3].clamp_(0, shape[0])

                for box in det:
                    box = box.cpu().numpy()
                    x1, y1, x2, y2, score, index = box
                    class_name = params['names'][int(index)]
                    label = f"{class_name} {score:.2f}"
                    
                    util.draw_box(img_to_draw, box, index, label)
                    draw.rectangle([int(x1), int(y1), int(x2), int(y2)], outline=(255, 0, 0), width=2)
                pil_img.save(f"./{filename.split('/')[-1].split('.')[0]}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', default=0, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--num-cls', type=int, default=4)
    parser.add_argument('--num-masks', type=int, default=2)
    parser.add_argument('--inp-size', type=int, default=640)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--data-dir', type=str, default='/Users/ninamasarykova/Documents/FIIT_STU/DizP/CooperativePerception/dataset_occupancy')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--redundancy', action=argparse.BooleanOptionalAction)
    parser.add_argument('--fusion', action=argparse.BooleanOptionalAction)
    parser.add_argument('--weights_path', default='yolo11x_remapped.pt', type=str)
    parser.add_argument('--model_size', default='x', type=str)
    parser.add_argument('--local-rank', type=int, default=0)

    args = parser.parse_args()

    args.local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    args.rank = int(os.environ.get("RANK", 0))
    args.world_size = int(os.getenv('WORLD_SIZE', 1))
    args.distributed = int(os.getenv('WORLD_SIZE', 1)) > 1

    with open('utils/args.yaml', errors='ignore') as f:
        params = yaml.safe_load(f)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.local_rank}")
    else:
        device = torch.device("cpu")
    params['device']=device

    if args.train:
        train(args, params)
    if args.validate:
        validate(args, params)
    if args.inference:
        inference(args, params)

if __name__ == "__main__":
    main()
