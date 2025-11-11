#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from ultralytics import YOLO
import torch
def main():
    print('MPS available: ',torch.backends.mps.is_available())  # True esperado
    data_yaml = 'datasets/coco_yolo_subset/coco_subset.yaml'
    model = YOLO('yolov8n.pt')
    model.train(
        data=data_yaml,
        imgsz=640,
        epochs=50,              # ajusta a tu tiempo
        batch=8,                # más estable en MPS
        workers=2,              # 0 -> lento; pon 2–4
        device='mps',
        amp=False,              # MPS no soporta AMP
        deterministic=False,    # <- IMPORTANTE para MPS
        # Augmentación “segura” para evitar glitches en TAL/MPS
        mosaic=0.0, mixup=0.0, copy_paste=0.0,
        auto_augment='none', erasing=0.0,
        # Optimizador y LR
        optimizer='SGD', cos_lr=True,
        # Salidas
        project='runs', name='train_yolov8n_subset',
        val=True, plots=True, patience=15
    )
    metrics = model.val(project='runs', name='val_yolov8n_subset', imgsz=640, batch=16, device='mps')
    print(metrics.results_dict)

if __name__ == '__main__':
    main()