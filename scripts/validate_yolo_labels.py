#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, glob

ROOT = "datasets/coco_yolo_subset"
NC = 80  # COCO

def check_dir(split):
    bad = 0
    labs = glob.glob(os.path.join(ROOT, split, "labels", "*.txt"))
    for lp in labs:
        with open(lp, "r") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        for i, ln in enumerate(lines, 1):
            parts = ln.split()
            if len(parts) != 5:
                print(f"[{split}] {lp}:{i} -> columnas={len(parts)} (!=5)")
                bad += 1; continue
            try:
                c = int(float(parts[0]))
                x,y,w,h = map(float, parts[1:])
            except Exception:
                print(f"[{split}] {lp}:{i} -> no numérico")
                bad += 1; continue
            if not (0 <= c < NC):
                print(f"[{split}] {lp}:{i} -> clase fuera de rango: {c}")
                bad += 1
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                print(f"[{split}] {lp}:{i} -> coords fuera [0,1]: {x,y,w,h}")
                bad += 1
            if w <= 0 or h <= 0:
                print(f"[{split}] {lp}:{i} -> caja degenerada: {w,h}")
                bad += 1
    print(f"[{split}] Revisados {len(labs)} archivos; hallados {bad} problemas.")
    return bad

if __name__ == "__main__":
    tb = check_dir("train")
    vb = check_dir("val")
    if tb + vb == 0:
        print("[OK] Labels válidas.")