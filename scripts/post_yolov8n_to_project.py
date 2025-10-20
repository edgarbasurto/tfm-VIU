#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, shutil

RUN = 'runs/train_yolov8n_coco1285'  # o train_yolov8n_coco128
FIG_DST = 'figuras'
TAB_DST = 'outputs/tables'
os.makedirs(FIG_DST, exist_ok=True)
os.makedirs(TAB_DST, exist_ok=True)

to_copy = [
  'results.png',
  'PR_curve.png', 'F1_curve.png', 'P_curve.png', 'R_curve.png',
  'confusion_matrix.png'
]
for fn in to_copy:
    src = os.path.join(RUN, fn)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(FIG_DST, f'yolov8n_{fn}'))
        print('[OK] Copiado', fn)
    else:
        print('[WARN] Falta', src)

# copia también hipers y args para apéndices
for fn in ['hyp.yaml', 'args.yaml']:
    src = os.path.join(RUN, fn)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(TAB_DST, f'yolov8n_{fn}'))
        print('[OK] Copiado', fn)