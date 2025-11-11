#!/usr/bin/env python3
# scripts/make_panels.py
import argparse, os
from PIL import Image

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', required=True)  # assets/qualitative_ids.txt
    ap.add_argument('--yolo', required=True)
    ap.add_argument('--frcnn', required=True)
    ap.add_argument('--effdet', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for line in open(args.list):
        stem = os.path.splitext(os.path.basename(line.strip()))[0]
        paths = [
            os.path.join(args.yolo,   f"{stem}.jpg"),           # ajusta sufijos según tus nombres
            os.path.join(args.frcnn,  f"{stem}_frcnn.jpg"),
            os.path.join(args.effdet, f"{stem}_effdet.jpg"),
        ]
        imgs = [Image.open(p).convert('RGB') for p in paths]
        # redimensiona a altura común
        h = min(i.height for i in imgs)
        imgs = [i.resize((int(i.width*h/i.height), h)) for i in imgs]
        w = sum(i.width for i in imgs)
        panel = Image.new('RGB', (w, h))
        x = 0
        for im in imgs:
            panel.paste(im, (x,0))
            x += im.width
        panel.save(os.path.join(args.out, f"{stem}_panel.jpg"))
if __name__ == '__main__':
    main()