#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os
from pathlib import Path
import numpy as np
from PIL import Image

import torch
from torchvision.ops import batched_nms
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import pil_to_tensor

from pycocotools.coco import COCO
from effdet import create_model

# ---------- utils ----------
def letterbox(im: np.ndarray, new_size=512, color=(114,114,114)):
    h, w = im.shape[:2]
    s = min(new_size / h, new_size / w)
    nh, nw = int(round(h * s)), int(round(w * s))
    im_rs = np.array(Image.fromarray(im).resize((nw, nh), Image.BILINEAR))
    canvas = np.full((new_size, new_size, 3), color, dtype=im_rs.dtype)
    top = (new_size - nh) // 2
    left = (new_size - nw) // 2
    canvas[top:top+nh, left:left+nw] = im_rs
    return canvas  # (H,W,3) uint8

def load_names(coco: COCO):
    cats = sorted(coco.loadCats(coco.getCatIds()), key=lambda c: c["id"])
    return [c["name"] for c in cats]  # 0..79

def draw(img_pil, boxes, labels, scores, names, thr=0.25):
    keep = scores >= thr
    if keep.sum() == 0:
        return img_pil
    b = boxes[keep].cpu().float()
    s = scores[keep].cpu().float()
    l = labels[keep].cpu().long()
    txt = [f"{names[int(li)]} {float(si):.2f}" for li, si in zip(l, s)]
    t = pil_to_tensor(img_pil).to(torch.uint8)
    ann = draw_bounding_boxes(t, b, labels=txt, width=2)
    return Image.fromarray(ann.permute(1,2,0).cpu().numpy())

def make_predictor(arch, num_classes, img_size, ckpt, device):
    pred = create_model(
        arch,
        bench_task='predict',
        num_classes=num_classes,
        image_size=(img_size, img_size),
        pretrained=False
    ).to(device)
    ckpt_state = torch.load(ckpt, map_location=device)
    pred.load_state_dict(ckpt_state, strict=False)  # claves extras son normales
    pred.eval()
    return pred

def postproc_nms(boxes, scores, labels, iou_thr=0.5, max_det=50, per_class_topk=8):
    # filtra por clase top-k (opcional)
    if per_class_topk and per_class_topk > 0:
        keep_idx = []
        labs = labels.cpu().numpy()
        for c in np.unique(labs):
            idxc = torch.nonzero(labels == int(c), as_tuple=False).view(-1)
            if idxc.numel():
                sc = scores[idxc]
                k = min(per_class_topk, sc.numel())
                topk = torch.topk(sc, k=k).indices
                keep_idx.append(idxc[topk])
        if keep_idx:
            keep_idx = torch.cat(keep_idx, dim=0)
            boxes, scores, labels = boxes[keep_idx], scores[keep_idx], labels[keep_idx]

    # NMS en CPU para máxima compatibilidad
    keep = batched_nms(boxes.cpu(), scores.cpu(), labels.cpu(), iou_thr)
    keep = keep[:max_det]
    return boxes[keep], scores[keep], labels[keep]

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--val-images', required=True)
    ap.add_argument('--ann-val',   required=True)
    ap.add_argument('--ckpt',      required=True)
    ap.add_argument('--ids-file',  required=True)
    ap.add_argument('--out',       required=True)
    ap.add_argument('--img-size',  type=int, default=512)
    ap.add_argument('--score-thr', type=float, default=0.25)
    ap.add_argument('--nms-iou',   type=float, default=0.5)
    ap.add_argument('--max-det',   type=int,   default=50)
    ap.add_argument('--per-class-topk', type=int, default=8)
    ap.add_argument('--device',    default='mps')
    args = ap.parse_args()

    device = torch.device(args.device if (args.device!='mps' or (hasattr(torch.backends,'mps') and torch.backends.mps.is_available())) else 'cpu')

    coco = COCO(args.ann_val)
    names = load_names(coco)

    model = make_predictor('tf_efficientdet_d0', 80, args.img_size, args.ckpt, device)

    os.makedirs(args.out, exist_ok=True)
    ids = [l.strip() for l in open(args.ids_file) if l.strip()]

    for rel in ids:
        img_name = Path(rel).name  # por si vino con ruta completa
        p = Path(args.val_images) / img_name

        # 1) letterbox -> 512x512 (exacto)
        img = Image.open(p).convert('RGB')
        lb = letterbox(np.array(img), new_size=args.img_size)
        img_lb = Image.fromarray(lb)

        x = (pil_to_tensor(img_lb).float()/255.0).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(x)  # {'detections': Tensor[B,M,6]}
            dets = out['detections'][0] if isinstance(out, dict) else out[0]

        boxes  = dets[:, :4].float()
        scores = dets[:, 4].float()
        labels = dets[:, 5].to(torch.long)

        # 2) filtrar
        conf_keep = scores >= args.score_thr
        boxes, scores, labels = boxes[conf_keep], scores[conf_keep], labels[conf_keep]

        if boxes.numel():
            boxes, scores, labels = postproc_nms(
                boxes, scores, labels,
                iou_thr=args.nms_iou,
                max_det=args.max_det,
                per_class_topk=args.per_class_topk
            )

        vis = draw(img_lb, boxes, labels, scores, names, thr=-1)  # ya filtrado
        vis.save(os.path.join(args.out, f"{Path(img_name).stem}_effdet.jpg"))

if __name__ == '__main__':
    main()