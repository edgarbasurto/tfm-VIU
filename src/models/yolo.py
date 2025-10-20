from __future__ import annotations
from typing import List, Dict, Any, Optional, Union

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

import numpy as np
from .base import Detector


class YOLOv8Detector(Detector):
    """Wrapper de YOLOv8 (ultralytics) con interfaz común."""

    def __init__(self, weights: str = "yolov8n.pt", device: Optional[str] = None, half: bool = True,
                 imgsz: int = 640, conf: float = 0.25, iou: float = 0.7):
        if YOLO is None:
            raise ImportError("ultralytics no está instalado. Ejecuta `pip install ultralytics`.")
        self.model = YOLO(weights)
        self.conf = conf
        self.iou  = iou
        self.imgsz = imgsz
        self.device = device
        self.max_det = 300
        self.half = half
        self.names = None
        try:
            # Ultralytics: YOLO -> .model (nn.Module) -> .names
            self.names = getattr(getattr(self.model, "model", None), "names", None)
        except Exception:
            pass

    def name(self) -> str:
        return f"YOLOv8({self.model.model.yaml.get('name', '?')})"

    def predict(self, images: List[Union[str, np.ndarray]]):
        # images: lista de paths o arrays (BGR/RGB). Ultralytics acepta ambos.
        results = self.model.predict(
            images,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
            half=self.half,
            max_det=self.max_det,
            device=self.device
        )
        out = []
        for r in results:
            dets = []
            if r.boxes is None:
                out.append(dets)
                continue
            for b in r.boxes:
                xyxy = b.xyxy[0].tolist()
                score = float(b.conf[0].item())
                cls_id = int(b.cls[0].item())
                cls_name = None
                if hasattr(r, 'names') and isinstance(r.names, dict):
                    cls_name = r.names.get(cls_id)
                dets.append({
                    "bbox": [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])],
                    "score": score,
                    "cls": cls_id,
                    "cls_name": cls_name
                })
            out.append(dets)
        return out
