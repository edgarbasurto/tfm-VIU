#!/usr/bin/env python3
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import argparse, os

def tile(imgs, labels, cols=2, pad=8, bg=(245,245,245)):
    # Normaliza a misma altura
    h = max(im.height for im in imgs)
    scale = lambda im: im.resize((int(im.width * h / im.height), h), Image.BILINEAR)
    imgs = [scale(im) for im in imgs]
    # calcula filas/cols
    rows = (len(imgs)+cols-1)//cols
    wcols = []
    for c in range(cols):
        wcols.append(max(imgs[i].width for i in range(c, len(imgs), cols)))
    W = sum(wcols) + pad*(cols+1)
    H = rows*(h) + pad*(rows+1) + 24*rows  # espacio para títulos
    canvas = Image.new("RGB", (W,H), bg)
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except:
        font = None
    x0 = pad; y = pad
    for r in range(rows):
        x = pad
        for c in range(cols):
            i = r*cols + c
            if i >= len(imgs): break
            im = imgs[i]
            canvas.paste(im, (x, y+20))
            # título
            draw.text((x, y), labels[i], fill=(20,20,20), font=font)
            x += wcols[c] + pad
        y += h + pad + 24
    return canvas

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-images", required=True)
    ap.add_argument("--ids-file", required=True)
    ap.add_argument("--yolo-dir", required=True)
    ap.add_argument("--frcnn-dir", required=True)
    ap.add_argument("--effdet-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ids = [l.strip() for l in open(args.ids_file) if l.strip()]
    for fn in ids:
        stem = Path(fn).stem
        raw    = Path(args.val_images) / fn
        yolo   = Path(args.yolo_dir)  / f"{stem}_yolo.jpg"
        frcnn  = Path(args.frcnn_dir) / f"{stem}_frcnn.jpg"
        effdet = Path(args.effdet_dir)/ f"{stem}_effdet.jpg"

        imgs, labels = [], []
        if raw.exists():    imgs.append(Image.open(raw).convert("RGB"));    labels.append("RAW")
        if yolo.exists():   imgs.append(Image.open(yolo).convert("RGB"));   labels.append("YOLOv8n")
        if frcnn.exists():  imgs.append(Image.open(frcnn).convert("RGB"));  labels.append("Faster R-CNN")
        if effdet.exists(): imgs.append(Image.open(effdet).convert("RGB")); labels.append("EfficientDet-D0")

        if len(imgs) >= 2:
            grid = tile(imgs, labels, cols=2, pad=8)
            grid.save(Path(args.out_dir)/f"{stem}_grid.jpg")
            print("[GRID]", stem)
        else:
            print("[SKIP] Faltan vistas para", stem)

if __name__ == "__main__":
    main()