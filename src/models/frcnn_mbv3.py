from __future__ import annotations
from typing import List, Dict, Any, Optional, Union
import torch, torchvision, numpy as np
from .base import Detector

class FasterRCNN_MBV3_FPN(Detector):
    """Faster R-CNN con MobileNetV3 Large FPN (rápido y ligero)."""
    def __init__(self, device: Optional[str] = None, conf: float = 0.25):
        self.conf = float(conf)  # <-- ¡IMPORTANTE! (arregla tu error)
        weights = torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT

        # Selección de dispositivo con fallback
        if device:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'

        # Modelo
        self.model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
        try:
            self.model.to(self.device)
        except Exception:
            print('[WARN] MBV3-FPN no soporta este backend; usando CPU.')
            self.device = 'cpu'
            self.model.to(self.device)

        # Ajustes de inferencia (menos FPs)
        self.model.eval()
        # Estos atributos existen en TorchVision 0.14+; si fallan, se ignoran
        try:
            self.model.roi_heads.score_thresh = 0.0
            self.model.roi_heads.nms_thresh   = 0.5
            self.model.roi_heads.detections_per_img = 300
        except Exception:
            pass

        self.tf = weights.transforms()  # espera PIL.Image

    def name(self) -> str:
        return "FRCNN(MBV3-FPN)"

    @torch.no_grad()
    def predict(self, images: List[Union[str, np.ndarray]]):
        import cv2
        from PIL import Image
        tensors = []
        for im in images:
            if isinstance(im, str):
                im = cv2.imread(im)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(im)
            t = self.tf(pil).to(self.device)
            tensors.append(t)

        outs = self.model(tensors)
        results = []
        for o in outs:
            boxes  = o['boxes'].detach().to('cpu').numpy()
            scores = o['scores'].detach().to('cpu').numpy()
            labels = o['labels'].detach().to('cpu').numpy().astype(int)  # category_id COCO
            keep   = scores >= self.conf
            dets = [{
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "score": float(s),
                "cls": int(c),
                "cls_name": None
            } for (x1, y1, x2, y2), s, c in zip(boxes, scores, labels)]
            results.append(dets)
        return results