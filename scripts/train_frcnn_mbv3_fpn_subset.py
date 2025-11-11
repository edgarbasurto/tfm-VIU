#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, time, json
from pathlib import Path
import torch, torchvision
from torchvision.ops import box_convert
from torchvision.transforms.functional import to_tensor
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def resolve_device():
    # torchvision detection en MPS suele fallar -> CPU
    use_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if use_mps:
        print("[WARN] FRCNN en MPS puede fallar; usando CPU por estabilidad.")
    return torch.device("cpu")

class CocoDetDataset(Dataset):
    def __init__(self, img_dir, ann_json, is_train=True):
        self.img_dir = Path(img_dir)
        self.coco = COCO(ann_json)
        self.ids = self.coco.getImgIds()
        self.is_train = is_train
        if is_train:
            keep = []
            for i in self.ids:
                anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[i], iscrowd=None))
                if any(a["bbox"][2] > 1 and a["bbox"][3] > 1 for a in anns):
                    keep.append(i)
            dropped = len(self.ids) - len(keep)
            self.ids = keep
            if dropped:
                print(f"[INFO] Train: {dropped} imágenes sin cajas descartadas")

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        info = self.coco.loadImgs([img_id])[0]
        path = self.img_dir / info["file_name"]
        img = Image.open(path).convert("RGB")
        w, h = img.size

        ann_ids = self.coco.getAnnIds(imgIds=[img_id], iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)

        boxes, labels, areas, iscrowd = [], [], [], []
        for a in anns:
            x, y, bw, bh = a["bbox"]
            if bw <= 1 or bh <= 1:  # filtra cajas diminutas
                continue
            boxes.append([x, y, x + bw, y + bh])  # xyxy
            labels.append(int(a["category_id"]))
            areas.append(float(a.get("area", bw * bh)))
            iscrowd.append(int(a.get("iscrowd", 0)))

        # --- CLAVE: tensores con forma correcta incluso si están vacíos ---
        boxes_t   = torch.as_tensor(boxes,  dtype=torch.float32)
        if boxes_t.numel() == 0:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
        labels_t  = torch.as_tensor(labels, dtype=torch.int64)
        if labels_t.numel() == 0:
            labels_t = torch.zeros((0,), dtype=torch.int64)
        area_t    = torch.as_tensor(areas,  dtype=torch.float32) if len(areas) else torch.zeros((0,), dtype=torch.float32)
        iscrowd_t = torch.as_tensor(iscrowd,dtype=torch.int64)   if len(iscrowd) else torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([img_id], dtype=torch.int64),
            "area": area_t,
            "iscrowd": iscrowd_t
        }

        # Augmentación simple
        if self.is_train and np.random.rand() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if boxes_t.numel() > 0:
                b = boxes_t.numpy()
                b[:, [0, 2]] = (w - b[:, [2, 0]])
                target["boxes"] = torch.from_numpy(b).to(torch.float32)

        return to_tensor(img), target

def collate_fn(batch): return tuple(zip(*batch))

@torch.no_grad()
def coco_eval(model, loader, coco_gt, device, out_json):
    model.eval()
    results = []
    for images, targets in loader:
        images = [i.to(device) for i in images]
        outputs = model(images)
        for tgt, out in zip(targets, outputs):
            img_id = int(tgt["image_id"])
            if "boxes" not in out or "scores" not in out or "labels" not in out:
                continue
            boxes = out["boxes"].cpu().numpy()
            scores = out["scores"].cpu().numpy()
            labels = out["labels"].cpu().numpy()
            # xyxy -> xywh
            xywh = box_convert(torch.from_numpy(boxes), in_fmt="xyxy", out_fmt="xywh").numpy()
            for b, s, l in zip(xywh, scores, labels):
                results.append({
                    "image_id": img_id,
                    "category_id": int(l),
                    "bbox": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                    "score": float(s)
                })
    with open(out_json, "w") as f:
        json.dump(results, f)

    coco_dt = coco_gt.loadRes(out_json)
    E = COCOeval(coco_gt, coco_dt, iouType="bbox")
    E.evaluate(); E.accumulate(); E.summarize()
    # devolver métricas clave
    return {
        "AP@[.5:.95]": float(E.stats[0]),
        "AP50": float(E.stats[1]),
        "AP75": float(E.stats[2]),
    }

def compute_val_loss_epoch(model, val_loader, device):
    # Para obtener 'loss' en val, el modelo debe estar en modo train y con targets.
    # Usamos no_grad para no guardar gráficas (más rápido).
    model.train()
    total = 0.0
    n = 0
    with torch.no_grad():
        for images, targets in val_loader:
            images = [i.to(device) for i in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            total += float(loss.detach().cpu())
            n += 1
    model.eval()
    return total / max(1, n)

def eval_coco_ap(model, val_loader, coco_gt, device, out_json):
    return coco_eval(model, val_loader, coco_gt, device, out_json)

def main():
    import csv


    ap = argparse.ArgumentParser()
    ap.add_argument("--train-images", default="datasets/coco_yolo_subset/train/images")
    ap.add_argument("--val-images",   default="datasets/coco_yolo_subset/val/images")
    ap.add_argument("--ann-train",    default="datasets/coco_yolo_subset/annotations/instances_train_subset.json")
    ap.add_argument("--ann-val",      default="datasets/coco_yolo_subset/annotations/instances_val_subset.json")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch",  type=int, default=2)
    ap.add_argument("--lr",     type=float, default=5e-3)
    ap.add_argument("--outdir", default="runs/frcnn_mbv3_subset")
    args = ap.parse_args()

    log_csv = os.path.join(args.outdir, "frcnn_history.csv")
    os.makedirs(args.outdir, exist_ok=True)
    with open(log_csv, "w", newline="") as f:
        csv.writer(f).writerow(["epoch","train_loss","val_loss","AP","AP50","AP75","lr","epoch_time_s"])

    device = resolve_device()
    os.makedirs(args.outdir, exist_ok=True)

    ds_train = CocoDetDataset(args.train_images, args.ann_train, is_train=True)
    ds_val   = CocoDetDataset(args.val_images,   args.ann_val,   is_train=False)

     # num_classes=91 (COCO: 1..90 + background)
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, 91)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    lr_sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(args.epochs//2,1), gamma=0.1)

    # === Forzar resolución equivalente a YOLO (640) ===
    # En torchvision reciente, min_size debe ser tupla
    if isinstance(model.transform.min_size, (list, tuple)):
        model.transform.min_size = (640,)   # ej. (800,) por defecto
    else:
        model.transform.min_size = 640
    model.transform.max_size = 640          # lado largo máximo

    # === Aceleraciones RPN/ROI sin cambiar imgsz ni dataset ===
    # API nueva (torchvision >= 0.13 aprox.): *_train / *_test
    if hasattr(model.rpn, "pre_nms_top_n_train"):
        model.rpn.pre_nms_top_n_train  = 1000  # default suele ser 2000
        model.rpn.pre_nms_top_n_test   = 1000
        model.rpn.post_nms_top_n_train = 200   # default suele ser 1000
        model.rpn.post_nms_top_n_test  = 200
    # API antigua: dict {'training':..., 'testing':...}
    elif hasattr(model.rpn, "pre_nms_top_n") and isinstance(model.rpn.pre_nms_top_n, dict):
        model.rpn.pre_nms_top_n["training"]  = 1000
        model.rpn.pre_nms_top_n["testing"]   = 1000
        model.rpn.post_nms_top_n["training"] = 200
        model.rpn.post_nms_top_n["testing"]  = 200

    # Menos ROIs por imagen en los heads (reduce cómputo)
    if hasattr(model.roi_heads, "batch_size_per_image"):
        model.roi_heads.batch_size_per_image = 128  # default 512

    # (Opcional) umbral de score para filtrar propuestas muy bajas
    if hasattr(model.rpn, "rpn_head") and hasattr(model.rpn, "box_coder"):
        # Nada que tocar aquí por defecto; solo ejemplo si quisieras ajustar nms_thresh:
        model.rpn.nms_thresh = 0.7  # mantener por defecto; bajar aceleraría algo pero cambia recall

    train_loader = DataLoader(ds_train, batch_size=args.batch, shuffle=True,  num_workers=4, collate_fn=collate_fn)
    val_loader   = DataLoader(ds_val,   batch_size=1,          shuffle=False, num_workers=4, collate_fn=collate_fn)

   

    # Entrenamiento corto
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        t0 = time.time()
        for images, targets in train_loader:
            images = [i.to(device) for i in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu())
        lr_sched.step()
        epoch_time = time.time() - t0
        train_loss = total / max(1, len(train_loader))
        print(f"[Epoch {epoch+1}/{args.epochs}] loss={total/len(train_loader):.4f} time={time.time()-t0:.1f}s")

        # --- val_loss (aprox) ---
        val_loss = compute_val_loss_epoch(model, val_loader, device)

        # --- APs COCO por época ---
        coco_gt = ds_val.coco
        out_json = str(Path(args.outdir) / f"detections_val_ep{epoch+1:03d}.json")
        metrics = eval_coco_ap(model, val_loader, coco_gt, device, out_json)

        avg = total / max(1, len(train_loader))
        lr_now = optimizer.param_groups[0]["lr"]
        with open(log_csv, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch+1, 
                f"{train_loss:.6f}", 
                f"{val_loss:.6f}", 
                f"{metrics['AP@[.5:.95]']:.6f}", 
                f"{metrics['AP50']:.6f}", 
                f"{metrics['AP75']:.6f}", 
                f"{lr_now:.8f}", 
                f"{epoch_time:.2f}"])

        print(f"[Epoch {epoch+1}/{args.epochs}] "
          f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
          f"AP={metrics['AP@[.5:.95]']:.3f} AP50={metrics['AP50']:.3f} "
          f"time={epoch_time:.1f}s")

    # Evaluación COCO
    coco_gt = ds_val.coco
    out_json = str(Path(args.outdir) / "detections_val.json")
    metrics = coco_eval(model, val_loader, coco_gt, device, out_json)
    with open(Path(args.outdir)/"metrics.json","w") as f:
        json.dump(metrics, f, indent=2)
    print("[METRICS]", metrics)

    # === Guardar checkpoints ===
    ckpt_last = Path(args.outdir) / "last.pth"
    ckpt_best = Path(args.outdir) / "best.pth"
    torch.save(model.state_dict(), ckpt_last)

    # criterio de "mejor": AP promedio COCO
    best_file = Path(args.outdir) / "best_ap.txt"
    best_ap = -1.0
    if best_file.exists():
        best_ap = float(best_file.read_text().strip())

    cur_ap = metrics["AP@[.5:.95]"]
    if cur_ap > best_ap:
        torch.save(model.state_dict(), ckpt_best)
        best_file.write_text(str(cur_ap))
        print(f"[CKPT] Nuevo BEST: AP={cur_ap:.4f} -> {ckpt_best}")
    else:
        print(f"[CKPT] BEST vigente: {best_ap:.4f}")

if __name__ == "__main__":
    main()