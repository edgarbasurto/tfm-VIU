#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, os
from pathlib import Path

def filter_coco(ann_path: str, images_dir: str, out_path: str):
    images_dir = Path(images_dir)
    fn_set = {p.name for p in images_dir.glob("*.jpg")} | {p.name for p in images_dir.glob("*.JPG")}
    with open(ann_path, "r") as f:
        coco = json.load(f)

    keep_images = [img for img in coco["images"] if img.get("file_name") in fn_set]
    keep_ids = {img["id"] for img in keep_images}
    keep_anns = [a for a in coco["annotations"] if a["image_id"] in keep_ids]
    out = {
        "images": keep_images,
        "annotations": keep_anns,
        "categories": coco["categories"],
        "licenses": coco.get("licenses", []),
        "info": coco.get("info", {})
    }
    os.makedirs(Path(out_path).parent, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"[OK] Guardado {out_path}: images={len(keep_images)} anns={len(keep_anns)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-images", required=True)
    ap.add_argument("--val-images",   required=True)
    ap.add_argument("--ann-train",    required=True)
    ap.add_argument("--ann-val",      required=True)
    ap.add_argument("--out-dir",      default="datasets/coco_yolo_subset/annotations")
    args = ap.parse_args()

    out_train = str(Path(args.out_dir) / "instances_train_subset.json")
    out_val   = str(Path(args.out_dir) / "instances_val_subset.json")
    os.makedirs(args.out_dir, exist_ok=True)

    filter_coco(args.ann_train, args.train_images, out_train)
    filter_coco(args.ann_val,   args.val_images,   out_val)

if __name__ == "__main__":
    main()