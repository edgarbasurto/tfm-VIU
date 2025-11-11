#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pycocotools.coco import COCO
from effdet import create_model
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
import csv
import random
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import json
from torchvision.transforms import functional as TF
import random

# -------------------- util: device --------------------
def resolve_device(arg):
    if arg == 'auto':
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    return torch.device(arg)

# -------------------- util: letterbox --------------------
def letterbox(im: np.ndarray, new_size=640, color=(114,114,114)):
    h, w = im.shape[:2]
    s = min(new_size / h, new_size / w)
    nh, nw = int(round(h * s)), int(round(w * s))
    im_rs = np.array(Image.fromarray(im).resize((nw, nh), Image.BILINEAR))
    canvas = np.full((new_size, new_size, 3), color, dtype=im_rs.dtype)
    top = (new_size - nh) // 2
    left = (new_size - nw) // 2
    canvas[top:top+nh, left:left+nw] = im_rs
    return canvas, s, left, top, (h, w)

# -------------------- Dataset --------------------
class CocoEffDetDataset(Dataset):
    def __init__(self, img_dir, ann_json, is_train=True, img_size=640):
        self.img_dir = Path(img_dir)
        self.coco = COCO(ann_json)
        self.img_size = img_size
        self.ids = self.coco.getImgIds()
        self.is_train = is_train
        self.p_hflip = 0.5      # prob. flip horizontal
        self.p_color = 0.6      # prob. color jitter
        # intensidades (ajústalas si quieres)
        self.jitter = dict(bright=0.2, contrast=0.2, saturation=0.15, hue=0.02)

        # filtra imágenes sin cajas (solo en train)
        if self.is_train:
            keep = []
            for i in self.ids:
                anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[i], iscrowd=None))
                if any(a["bbox"][2] > 1 and a["bbox"][3] > 1 for a in anns):
                    keep.append(i)
            dropped = len(self.ids) - len(keep)
            self.ids = keep
            if dropped:
                print(f"[INFO] Train: {dropped} imágenes sin cajas descartadas")

        # mapeo COCO id -> [0..79]
        cats = sorted(self.coco.loadCats(self.coco.getCatIds()), key=lambda c: c["id"])
        self.cat2idx = {c["id"]: i for i, c in enumerate(cats)}
        self.idx2cat = {i: cid for cid, i in self.cat2idx.items()}

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        info = self.coco.loadImgs([img_id])[0]
        path = self.img_dir / info["file_name"]
        img = Image.open(path).convert("RGB")
        img = np.array(img, copy=True)

        # letterbox a 640
        img_lb, scale, padw, padh, (h0,w0) = letterbox(img, new_size=self.img_size)

        # anotaciones
        ann_ids = self.coco.getAnnIds(imgIds=[img_id], iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)

        boxes, labels = [], []
        for a in anns:
            x, y, bw, bh = a["bbox"]
            if bw <= 1 or bh <= 1:
                continue
            x1 = x * scale + padw
            y1 = y * scale + padh
            x2 = (x + bw) * scale + padw
            y2 = (y + bh) * scale + padh
            boxes.append([x1, y1, x2, y2])
            labels.append(self.cat2idx[int(a["category_id"])])

        # Convierte a ndarray para poder modificar cajas
        boxes_np = np.array(boxes, dtype=np.float32) if len(boxes) else np.zeros((0,4), dtype=np.float32)

        # ----- AUGMENTACIONES SOLO EN TRAIN -----
        if self.is_train:
            H, W = img_lb.shape[0], img_lb.shape[1]

            # (A) Flip horizontal en el canvas letterboxed
            if random.random() < self.p_hflip:
                # invierte la imagen
                img_lb = img_lb[:, ::-1, :].copy()
                # ajusta cajas: x' = W - x
                if boxes_np.shape[0] > 0:
                    x1 = boxes_np[:, 0].copy()
                    x2 = boxes_np[:, 2].copy()
                    boxes_np[:, 0] = W - x2
                    boxes_np[:, 2] = W - x1
                    # clamp de seguridad
                    boxes_np[:, [0,2]] = np.clip(boxes_np[:, [0,2]], 0, W-1)

            # (B) Color jitter ligero (PIL + torchvision F)
            if random.random() < self.p_color:
                img_pil = Image.fromarray(img_lb)
                j = self.jitter
                img_pil = TF.adjust_brightness(img_pil,  1.0 + random.uniform(-j['bright'],    j['bright']))
                img_pil = TF.adjust_contrast(img_pil,    1.0 + random.uniform(-j['contrast'],  j['contrast']))
                img_pil = TF.adjust_saturation(img_pil,  1.0 + random.uniform(-j['saturation'],j['saturation']))
                img_pil = TF.adjust_hue(img_pil,               random.uniform(-j['hue'],       j['hue']))
                img_lb = np.array(img_pil)

        # Continúa igual: crea tensores desde img_lb y boxes_np
        img_t = torch.from_numpy(img_lb).permute(2,0,1).float() / 255.0
        # normalización ImageNet
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        img_t = (img_t - mean) / std

        if boxes_np.shape[0] > 0:
            bbox_t = torch.tensor(boxes_np, dtype=torch.float32)
            cls_t  = torch.tensor(labels,  dtype=torch.long)        # 1D long
        else:
            bbox_t = torch.zeros((0,4), dtype=torch.float32)
            cls_t  = torch.zeros((0,),   dtype=torch.long)

        H, W = img_lb.shape[0], img_lb.shape[1]  # 640, 640
        lnpos = max(bbox_t.shape[0], 1)

        target = {
            'bbox': bbox_t,
            'cls': cls_t,
            'img_size': torch.tensor([H, W], dtype=torch.float32),
            'img_scale': torch.tensor([1.0], dtype=torch.float32),
            'label_num_positives': torch.tensor([lnpos], dtype=torch.int32),

            # extras útiles (no usados por el bench)
            'orig_size': torch.tensor([h0, w0], dtype=torch.int32),
            'pad': torch.tensor([padw, padh], dtype=torch.float32),
            'img_id': torch.tensor([img_id], dtype=torch.int64),
        }
        return img_t, target

def collate_fn(batch):
    imgs, tgts = zip(*batch)
    return torch.stack(imgs, 0), list(tgts)

# -------------------- Targets -> list[dict] normalizado --------------------
def norm_targets_for_bench(targs, device, H=640, W=640):
    fixed = []
    for t in targs:
        # bbox
        b = t['bbox']
        if not torch.is_tensor(b): b = torch.as_tensor(b, dtype=torch.float32)
        b = b.to(device).float()

        # cls (1D, long)
        c = t['cls']
        if not torch.is_tensor(c): c = torch.as_tensor(c, dtype=torch.long)
        if c.ndim > 1:  # aplana si viene [N,1]
            c = c.view(-1)
        c = c.to(device, dtype=torch.long)

        # tamaños / escala
        img_size  = t.get('img_size',  torch.tensor([H, W], dtype=torch.float32))
        img_scale = t.get('img_scale', torch.tensor([1.0],   dtype=torch.float32))
        if not torch.is_tensor(img_size):  img_size  = torch.as_tensor(img_size,  dtype=torch.float32)
        if not torch.is_tensor(img_scale): img_scale = torch.as_tensor(img_scale, dtype=torch.float32)
        img_size  = img_size.to(device).float()
        img_scale = img_scale.to(device).float()

        lnpos = max(1, int(b.shape[0]))

        fixed.append({
            'bbox': b,
            'cls': c,  # 1D long
            'img_size': img_size,
            'img_scale': img_scale,
            'label_num_positives': lnpos,
        })
    return fixed

# -------------------- Empaquetar a dict-of-lists para DetBenchTrain --------------------
def pack_targets_dict(targs):
    out = {
        'bbox': [],
        'cls': [],
        'img_size': [],
        'img_scale': [],
        'label_num_positives': []
    }
    for t in targs:
        out['bbox'].append(t['bbox'])
        out['cls'].append(t['cls'])
        out['img_size'].append(t['img_size'])
        out['img_scale'].append(t['img_scale'])
        ln = t['label_num_positives']
        if isinstance(ln, torch.Tensor):
            ln = int(ln.item()) if ln.numel() == 1 else int(ln[0].item())
        out['label_num_positives'].append(int(ln))
    return out

# -------------------- Reducir salida de pérdida --------------------
def reduce_loss(loss_out):
    if torch.is_tensor(loss_out):
        return loss_out
    if isinstance(loss_out, dict):
        if 'loss' in loss_out and torch.is_tensor(loss_out['loss']):
            return loss_out['loss']
        scalars = [v for v in loss_out.values() if torch.is_tensor(v) and v.ndim == 0]
        if scalars:
            return torch.stack(scalars).sum()
        raise TypeError(f"loss_out dict sin tensores escalares: {list(loss_out.keys())}")
    if isinstance(loss_out, (list, tuple)):
        parts = [reduce_loss(v) for v in loss_out]
        return torch.stack(parts).sum()
    raise TypeError(f"Tipo de loss_out no esperado: {type(loss_out)}")



def _to_py(obj):
    import torch
    if torch.is_tensor(obj):
        if obj.ndim == 0:
            return float(obj.item())
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {k: _to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_py(x) for x in obj]
    return obj

def _round_scalars_inplace(d, ndigits=4):
    for k, v in list(d.items()):
        if isinstance(v, float):
            d[k] = round(v, ndigits)
        elif isinstance(v, list):
            d[k] = [round(x, ndigits) if isinstance(x, (int, float)) else x for x in v]
        elif isinstance(v, dict):
            _round_scalars_inplace(v, ndigits)
    return d

def _make_predictor_from_trainbench(train_bench, device):
    # Reconstruye un bench de predicción y le carga los pesos del bench de entrenamiento
    try:
        cfg = train_bench.model.config
        arch = getattr(cfg, 'name', 'tf_efficientdet_d0')
        img_size = getattr(cfg, 'image_size', (640, 640))
        num_classes = getattr(train_bench.model, 'num_classes', 80)
    except Exception:
        arch = 'tf_efficientdet_d0'
        img_size = (640, 640)
        num_classes = 80

    pred = create_model(
        arch,
        bench_task='predict',
        num_classes=num_classes,
        image_size=img_size,
        pretrained=False
    ).to(device)
    # Carga pesos del bench de train (claves compatibles)
    pred.load_state_dict(train_bench.state_dict(), strict=False)
    pred.eval()
    return pred

@torch.no_grad()
def evaluate_map(model, dataloader, device, num_classes=80, score_thr=0.05,
                 class_metrics=True, outfile=None):
    was_training = model.training
    model.eval()

    # 👉 crea el predictor a partir del bench de entrenamiento
    predictor = _make_predictor_from_trainbench(model, device)

    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        class_metrics=class_metrics,
        max_detection_thresholds=[1, 10, 100],
        iou_thresholds=None,
        backend="pycocotools"
    )

    for imgs, targs in dataloader:
        imgs = imgs.to(device)

        # 1) PREDICCIONES con DetBenchPredict (no requiere targets)
        out = predictor(imgs)

        # Normaliza salida a un Tensor [B, M, 6]
        if isinstance(out, dict):
            dets = out.get('detections', None)
            if dets is None:
                raise KeyError("El predictor devolvió un dict sin la clave 'detections'.")
        elif torch.is_tensor(out):
            dets = out
        elif isinstance(out, (list, tuple)):
            # Algunas variantes devuelven (detections,) o (detections, anchors)
            if len(out) == 0:
                raise ValueError("El predictor devolvió una lista/tupla vacía.")
            dets = out[0]
        else:
            raise TypeError(f"Salida de predictor no soportada: {type(out)}")

        # Asegura forma [B, M, 6]
        if dets.ndim == 2:
            # caso B=1 -> [M,6] -> [1,M,6]
            dets = dets.unsqueeze(0)
        elif dets.ndim != 3:
            raise ValueError(f"Se esperaba Tensor 3D [B,M,6], llegó {tuple(dets.shape)}")

        # justo después de construir 'dets'
        avg_dets = float((dets[..., 4] > 0.01).float().sum().item()) / dets.shape[0]
        print(f"[EVAL] avg dets >0.01 por imagen: {avg_dets:.1f}")

        B = dets.shape[0]
        preds = []
        for b in range(B):
            d = dets[b]
            keep = d[:, 4] > score_thr
            if keep.any():
                boxes = d[keep, :4]
                scores = d[keep, 4]
                labels = d[keep, 5].to(torch.long)
            else:
                boxes = torch.zeros((0, 4), device=device, dtype=torch.float32)
                scores = torch.zeros((0,), device=device, dtype=torch.float32)
                labels = torch.zeros((0,), device=device, dtype=torch.long)
            preds.append({'boxes': boxes, 'scores': scores, 'labels': labels})

        # 2) TARGETS (como ya lo tenías)
        gts = []
        for t in targs:
            b = t['bbox'].to(device).float()
            c = t['cls']
            if not torch.is_tensor(c): c = torch.as_tensor(c, dtype=torch.long, device=device)
            c = c.to(device, dtype=torch.long)
            if c.ndim > 1:
                c = c.view(-1)
            gts.append({'boxes': b, 'labels': c})

        metric.update(preds, gts)

    res = metric.compute()
    out = _to_py(res)
    _round_scalars_inplace(out, ndigits=4)

    if outfile is not None:
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        with open(outfile, "w") as f:
            json.dump(out, f, indent=2)

    model.train(was_training)
    return out

def count_val_classes(ds):
    from collections import Counter
    c = Counter()
    for i in range(len(ds)):
        _, t = ds[i]
        if t['cls'].numel():
            c.update(t['cls'].tolist())
    return dict(sorted(c.items(), key=lambda kv: kv[0]))

# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-images', default='datasets/coco_yolo_subset/train/images')
    ap.add_argument('--val-images',   default='datasets/coco_yolo_subset/val/images')
    ap.add_argument('--ann-train',    default='datasets/coco_yolo_subset/annotations/instances_train_subset.json')
    ap.add_argument('--ann-val',      default='datasets/coco_yolo_subset/annotations/instances_val_subset.json')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch',  type=int, default=8)
    ap.add_argument('--lr',     type=float, default=1e-3)
    ap.add_argument('--device', default='auto', help='auto|mps|cuda|cpu')
    ap.add_argument('--img-size', type=int, default=640)
    ap.add_argument('--out',    default='runs/train_effdet_d0_subset')
    args = ap.parse_args()

    device = resolve_device(args.device)
    print(f"[INFO] Device = {device.type}")
    os.makedirs(args.out, exist_ok=True)

    def set_seed(s=42):
        import random, numpy as np, torch
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
            import torch.backends.cudnn as cudnn
            cudnn.deterministic = False
            cudnn.benchmark = True

    set_seed(42)

    ds_tr = CocoEffDetDataset(args.train_images, args.ann_train, is_train=True, img_size=args.img_size)
    ds_va = CocoEffDetDataset(args.val_images,   args.ann_val,   is_train=False, img_size=args.img_size)

    print("[VAL] Frecuencia por clase (índice):", count_val_classes(ds_va))

    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True,  num_workers=0, collate_fn=collate_fn)
    dl_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False, num_workers=0, collate_fn=collate_fn)

    # EfficientDet-D0 (imagen 640x640 - 512x512, 80 clases)
    model = create_model(
        'tf_efficientdet_d0',
        bench_task='train',
        bench_labeler=True,  # <- que el bench construya los targets de ancla
        num_classes=80,
        image_size=(args.img_size, args.img_size),
        pretrained=True
    ).to(device)

    head_params = list(model.model.class_net.parameters()) + list(model.model.box_net.parameters())
    backbone_params = [p for n,p in model.named_parameters()
                    if 'class_net' not in n and 'box_net' not in n]

    optim = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr,       "weight_decay": 1e-4},
        {"params": head_params,     "lr": args.lr * 5.0, "weight_decay": 1e-4},
    ])

    # warmup 1 época + cosine el resto
    warmup_epochs = 1
    main_epochs   = max(1, args.epochs - warmup_epochs)
    sched_warm = LinearLR(optim, start_factor=0.1, end_factor=1.0, total_iters=max(1, len(dl_tr)*warmup_epochs))
    sched_main = CosineAnnealingLR(optim, T_max=max(1, len(dl_tr)*main_epochs))
    scheduler  = SequentialLR(optim, schedulers=[sched_warm, sched_main], milestones=[len(dl_tr)*warmup_epochs])

    metrics_csv = os.path.join(args.out, "metrics_effdet_d0.csv")
    os.makedirs(args.out, exist_ok=True)
    with open(metrics_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch","train_loss","val_loss","lr"])

    # ---- sanity check: forma de targets esperada ----
    with torch.no_grad():
        dummy = torch.zeros(2, 3, args.img_size, args.img_size, device=device)
        per_img = []
        for _ in range(dummy.shape[0]):
            per_img.append({
                'bbox': torch.tensor([[50., 50., 120., 120.]], device=device, dtype=torch.float32),
                'cls':  torch.tensor([0], device=device, dtype=torch.long),   # 1D long
                'img_size': torch.tensor([640., 640.], device=device, dtype=torch.float32),
                'img_scale': torch.tensor([1.], device=device, dtype=torch.float32),
                'label_num_positives': 1,
            })
        dummy_tgt = pack_targets_dict(per_img)
        _ = model(dummy, dummy_tgt)
    print("[SANITY] OK")

    best = 1e9
    patience, bad = 10, 0
    for epoch in range(1, args.epochs+1):
        model.train()
        t0, total = time.time(), 0.0

        for step, (imgs, targs) in enumerate(dl_tr):
            imgs = imgs.to(device)
            targs_list = norm_targets_for_bench(targs, device, H=imgs.shape[-2], W=imgs.shape[-1])
            targs_batched = pack_targets_dict(targs_list)

            optim.zero_grad(set_to_none=True)
            # dentro del bucle de entrenamiento, antes de backward:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                loss_out = model(imgs, targs_batched)
            loss = reduce_loss(loss_out)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optim.step()
            scheduler.step()
            total += float(loss.detach().cpu())

        # -------------------- VALIDACIÓN (loss) --------------------
        vtot = 0.0
        was_training = model.training
        model.train()  # DetBenchTrain retorna dict de pérdidas solo en modo train
        with torch.no_grad():
            for imgs, targs in dl_va:
                imgs = imgs.to(device)
                targs_list = norm_targets_for_bench(targs, device, H=imgs.shape[-2], W=imgs.shape[-1])
                targs_batched = pack_targets_dict(targs_list)
                vout = model(imgs, targs_batched)  # dict {'loss','class_loss','box_loss'}
                vtot += float(vout['loss'])
        model.train(was_training)

        val_loss = vtot / max(1, len(dl_va))
        ep_loss = total / max(1, len(dl_tr))  # <<< faltaba
        dt = time.time() - t0
        print(f"[Epoch {epoch}/{args.epochs}] train_loss={ep_loss:.4f} val_loss={val_loss:.4f} time={dt:.1f}s")

        lr_now = optim.param_groups[0]["lr"]
        with open(metrics_csv, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{ep_loss:.6f}", f"{val_loss:.6f}", f"{lr_now:.8f}"])

        if val_loss + 1e-6 < best:
            best, bad = val_loss, 0
            torch.save(model.state_dict(), os.path.join(args.out, "best.pth"))
        else:
            bad += 1
            if bad >= patience:
                print("EarlyStopping"); break

    # --- FIN DEL BUCLE DE ENTRENAMIENTO ---

    # 1) Guarda el último estado del entrenamiento tal cual terminó
    torch.save(model.state_dict(), os.path.join(args.out, "last.pth"))

    # 2) ---- Evaluación mAP (mejor checkpoint) ----
    best_pth = os.path.join(args.out, "best.pth")
    if os.path.exists(best_pth):
        model.load_state_dict(torch.load(best_pth, map_location=device))

    map_json = os.path.join(args.out, "map_results.json")
    map_res = evaluate_map(model, dl_va, device, num_classes=80,
                       score_thr=0.01, class_metrics=True, outfile=map_json)
    
    def _none_if_minus1(x):
        if isinstance(x, list):
            return [(_none_if_minus1(v)) for v in x]
        return None if isinstance(x, (int,float)) and x == -1.0 else x

    map_res_clean = {k: _none_if_minus1(v) for k,v in map_res.items()}
    with open(map_json, "w") as f:
        json.dump(map_res_clean, f, indent=2)

    print("[mAP] Resumen:", {k: v for k, v in map_res.items()
          if isinstance(v, float) or k.startswith(("map_", "mar_"))})
    print(f"[mAP] Guardado en: {map_json}")

if __name__ == '__main__':
    main()