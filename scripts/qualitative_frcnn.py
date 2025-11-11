#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, json
from pathlib import Path
import torch
from PIL import Image
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import pil_to_tensor
from pycocotools.coco import COCO
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_mobilenet_v3_large_320_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def resolve_device(arg: str) -> torch.device:
    if arg.lower() == "cpu":
        return torch.device("cpu")
    if arg.lower() == "mps":
        # En macOS, detection en MPS suele ser inestable → forzamos CPU si no está 100% OK
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(arg)


def coco_id2name(coco: COCO):
    cats = coco.loadCats(coco.getCatIds())
    return {c["id"]: c["name"] for c in cats}  # category_id → name


def load_checkpoint_state(ckpt_path: str, map_location: torch.device):
    state = torch.load(ckpt_path, map_location=map_location)
    # admite formatos: state_dict "plano" o {"model": state_dict, ...}
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    return state


def build_model_matching_ckpt(state_dict, try_320_first=False):
    """
    Construye FRCNN MobilenetV3 con el mismo nº de clases que el checkpoint
    y trata de empatar arquitectura: large_fpn vs large_320_fpn.
    """
    # Inferir num_classes desde el head del checkpoint
    head_w = None
    for k in [
        "roi_heads.box_predictor.cls_score.weight",
        "module.roi_heads.box_predictor.cls_score.weight",
    ]:
        if k in state_dict:
            head_w = state_dict[k]
            break
    if head_w is None:
        raise RuntimeError("No se encontró 'roi_heads.box_predictor.cls_score.weight' en el checkpoint.")
    num_classes = head_w.shape[0]

    def make_and_load(make_320: bool):
        if make_320:
            base = fasterrcnn_mobilenet_v3_large_320_fpn(weights=None, weights_backbone=None)
        else:
            base = fasterrcnn_mobilenet_v3_large_fpn(weights=None, weights_backbone=None)
        in_feats = base.roi_heads.box_predictor.cls_score.in_features
        base.roi_heads.box_predictor = FastRCNNPredictor(in_feats, num_classes)
        missing, unexpected = base.load_state_dict(state_dict, strict=False)
        return base, missing, unexpected

    # Probamos una, si hay muchas claves faltantes del backbone, probamos la otra
    order = [try_320_first, not try_320_first]
    for make_320 in order:
        model, missing, unexpected = make_and_load(make_320)
        # Heurística: si faltan muchas conv.* del backbone, prueba la otra
        too_many_backbone_miss = sum(1 for k in missing if k.startswith("backbone")) > 10
        if not too_many_backbone_miss:
            arch = "large_320_fpn" if make_320 else "large_fpn"
            print(f"[INFO] Arquitectura inferida: fasterrcnn_mobilenet_v3_{arch}")
            print("[load_state_dict] missing:", missing[:8], ("...(+%d)" % (len(missing)-8) if len(missing)>8 else ""))
            print("[load_state_dict] unexpected:", unexpected[:8], ("...(+%d)" % (len(unexpected)-8) if len(unexpected)>8 else ""))
            return model
    # Si llegamos aquí, nos quedamos con la segunda opción igualmente
    return model


def draw(img, boxes, labels, scores, id2name, thr=0.25):
    keep = scores >= thr
    if keep.sum() == 0:
        return img
    b = boxes[keep].cpu()
    s = scores[keep].cpu()
    l = labels[keep].cpu()
    texts = [f"{id2name.get(int(li), str(int(li)))} {float(si):.2f}" for li, si in zip(l, s)]
    t = pil_to_tensor(img).to(torch.uint8)  # CHW uint8
    ann = draw_bounding_boxes(t, b, labels=texts, width=2)
    return Image.fromarray(ann.permute(1, 2, 0).cpu().numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-images", required=True)
    ap.add_argument("--ann-val", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ids-file", required=True)  # rutas completas o nombres de archivo
    ap.add_argument("--out", required=True)
    ap.add_argument("--score-thr", type=float, default=0.25)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = resolve_device(args.device)
    coco = COCO(args.ann_val)
    id2name = coco_id2name(coco)

    # Cargar checkpoint (sin descargas)
    state = load_checkpoint_state(args.ckpt, map_location="cpu")

    # Construir modelo compatible y cargar pesos
    # Por defecto intentamos 'large_fpn' primero (más común); si falla, cae a 320.
    model = build_model_matching_ckpt(state, try_320_first=False)
    model.to(device).eval()

    os.makedirs(args.out, exist_ok=True)

    # Leer rutas/nombres del archivo
    lines = [l.strip() for l in open(args.ids_file, "r") if l.strip()]
    total, saved = len(lines), 0

    for line in lines:
        p = Path(line)
        if not p.is_file():
            p = Path(args.val_images) / line  # si viene solo el nombre
        if not p.is_file():
            print(f"[WARN] No existe: {line}")
            continue

        img = Image.open(p).convert("RGB")
        with torch.no_grad():
            inp = pil_to_tensor(img).float() / 255.0  # CHW float32 [0,1]
            pred = model([inp.to(device)])[0]

        boxes = pred.get("boxes", torch.zeros((0, 4))).to(device)
        labels = pred.get("labels", torch.zeros((0,), dtype=torch.long)).to(device)
        scores = pred.get("scores", torch.zeros((0,), dtype=torch.float32)).to(device)

        vis = draw(img, boxes, labels, scores, id2name, thr=args.score_thr)
        out_path = Path(args.out) / f"{p.stem}_frcnn.jpg"
        vis.save(out_path)
        saved += 1

    print(f"[DONE] Guardadas {saved}/{total} imágenes en: {args.out}")


if __name__ == "__main__":
    main()