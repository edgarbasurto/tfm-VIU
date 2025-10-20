#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from ultralytics import YOLO

def main():
    # Modelo base preentrenado en COCO
    model = YOLO('yolov8n.pt')
    # Entrenamiento rápido en coco128 (mini)
    model.train(
        data='coco128.yaml',   # dataset mini integrado
        imgsz=640,
        epochs=80,             # 50–100 recomendado
        batch=16,              # ajusta si falta memoria
        device='mps',          # MPS en Apple Silicon
        optimizer='SGD',       # o 'AdamW'
        cos_lr=True,
        amp=False,             # AMP no va en MPS
        project='runs', name='train_yolov8n_coco128',
        val=True, plots=True,
        patience=20            # early stopping
    )
    # Evaluación al final (val set del yaml)
    model.val(project='runs', name='val_yolov8n_coco128', device='mps')

if __name__ == '__main__':
    main()