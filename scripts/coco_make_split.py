#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, os, random, argparse, shutil
from collections import defaultdict

def coco_to_yolo_bbox(x, y, w, h, iw, ih):
    # COCO xywh -> YOLO normalized x_center y_center w h
    xc = x + w/2.0
    yc = y + h/2.0
    return xc/iw, yc/ih, w/iw, h/ih

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-images', required=True, help='datasets/coco/train2017')
    ap.add_argument('--val-images',   required=True, help='datasets/coco/val2017')
    ap.add_argument('--ann-train',    required=True, help='datasets/coco/annotations/instances_train2017.json')
    ap.add_argument('--ann-val',      required=True, help='datasets/coco/annotations/instances_val2017.json')
    ap.add_argument('--out',          default='datasets/coco_yolo_subset', help='salida dataset YOLO')
    ap.add_argument('--max-train',    type=int, default=5000, help='n imágenes train')
    ap.add_argument('--max-val',      type=int, default=1000, help='n imágenes val (opcional)')
    ap.add_argument('--seed',         type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    os.makedirs(args.out, exist_ok=True)
    for split in ('train', 'val'):
        os.makedirs(os.path.join(args.out, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(args.out, split, 'labels'), exist_ok=True)

    def load_coco(ann_file):
        with open(ann_file, 'r') as f:
            coco = json.load(f)
        imgs = {im['id']: im for im in coco['images']}
        anns_by_img = defaultdict(list)
        for a in coco['annotations']:
            if a.get('iscrowd', 0) == 1:  # ignorar crowd
                continue
            anns_by_img[a['image_id']].append(a)
        # categoría: id -> idx [0..nc-1]
        cats = sorted(coco['categories'], key=lambda c: c['id'])
        catid2idx = {c['id']: i for i, c in enumerate(cats)}
        names = [c['name'] for c in cats]
        return imgs, anns_by_img, catid2idx, names

    imgs_tr, anns_tr, catid2idx, names = load_coco(args.ann_train)
    imgs_vl, anns_vl, _, _           = load_coco(args.ann_val)

    def sample_ids(imgs, k):
        ids = list(imgs.keys())
        random.shuffle(ids)
        return ids[:min(k, len(ids))]

    train_ids = sample_ids(imgs_tr, args.max_train)
    val_ids   = sample_ids(imgs_vl, args.max_val)

    def process_split(split, ids, imgs, anns, images_dir):
        out_img = os.path.join(args.out, split, 'images')
        out_lab = os.path.join(args.out, split, 'labels')
        kept = 0
        for img_id in ids:
            im = imgs[img_id]
            fn = im['file_name']
            src = os.path.join(images_dir, fn)
            if not os.path.exists(src):
                continue
            dst = os.path.join(out_img, fn)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)

            iw, ih = im['width'], im['height']
            lab_path = os.path.join(out_lab, fn.replace('.jpg', '.txt'))
            lines = []
            for a in anns.get(img_id, []):
                cat_id = a['category_id']
                if cat_id not in catid2idx:  # seguridad
                    continue
                cls = catid2idx[cat_id]
                x, y, w, h = a['bbox']
                xc, yc, ww, hh = coco_to_yolo_bbox(x, y, w, h, iw, ih)
                # descartar cajas degeneradas
                if ww <= 0 or hh <= 0:
                    continue
                lines.append(f"{cls} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}")
            with open(lab_path, 'w') as f:
                f.write("\n".join(lines))
            kept += 1
        return kept

    ntr = process_split('train', train_ids, imgs_tr, anns_tr, args.train_images)
    nvl = process_split('val',   val_ids,   imgs_vl, anns_vl,   args.val_images)
    print(f"[OK] Copiadas y etiquetadas: train={ntr} val={nvl}")

    # YAML del dataset
    yaml = f"""# Dataset YOLO (COCO subset)
path: {args.out}
train: train/images
val: val/images
nc: {len(names)}
names: {names}
"""
    with open(os.path.join(args.out, 'coco_subset.yaml'), 'w', encoding='utf-8') as f:
        f.write(yaml)
    print("[OK] Escrito", os.path.join(args.out, 'coco_subset.yaml'))

if __name__ == '__main__':
    main()