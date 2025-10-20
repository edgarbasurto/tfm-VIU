from __future__ import annotations
from typing import List, Dict, Any, Optional, Union
import numpy as np
import torch

# Constantes de normalización (timm o fallback)
try:
    from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
except Exception:
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD  = (0.229, 0.224, 0.225)

from effdet import create_model
try:
    from effdet.bench import DetBenchPredict
except Exception:
    DetBenchPredict = None

from .base import Detector

COCO80_NAMES = [
 'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
 'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow',
 'elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis',
 'snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket',
 'bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli',
 'carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed','dining table','toilet',
 'tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator',
 'book','clock','vase','scissors','teddy bear','hair drier','toothbrush'
]

class EfficientDetDetector(Detector):
    """EfficientDet-D0 (effdet, COCO) tolerante a cambios de API."""
    def __init__(self, variant: str = 'tf_efficientdet_d0', device: Optional[str] = None, conf: float = 0.25):
        # Dispositivo
        if device:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'

        # Crear modelo (con/sin bench según versión)
        bench = None
        try:
            bench = create_model(variant, pretrained=True, bench_task='predict', num_classes=None)
        except TypeError:
            bench = create_model(variant, pretrained=True, num_classes=None)
        if DetBenchPredict is not None and bench.__class__.__name__ != 'DetBenchPredict':
            try:
                bench = DetBenchPredict(bench)
            except Exception:
                pass
        self.model = bench

        # Mover a device con fallback
        try:
            self.model.to(self.device)
        except Exception:
            print("[WARN] EfficientDet no soporta este backend; usando CPU.")
            self.device = 'cpu'; self.model.to(self.device)
        self.model.eval()
        self.conf = conf

        # Inferir image size
        img_size = 512
        try:
            cfg = getattr(self.model, 'config', None) or getattr(getattr(self.model, 'model', None), 'config', None)
            if cfg is not None:
                sz = getattr(cfg, 'image_size', None)
                if isinstance(sz, (list, tuple)): img_size = int(sz[0])
                elif isinstance(sz, int): img_size = int(sz)
        except Exception:
            pass
        self.img_size = img_size

        # Proveer nombres estilo YOLO para mapear por nombre -> category_id COCO
        try:
            self.model.names = {i: n for i, n in enumerate(COCO80_NAMES)}
        except Exception:
            pass

    def name(self) -> str:
        return "EfficientDet(D0)"

    @torch.no_grad()
    def predict(self, images: List[Union[str, np.ndarray]]):
        import cv2, torch
        outs_all: List[List[Dict[str, Any]]] = []

        mean = np.array(IMAGENET_DEFAULT_MEAN, dtype=np.float32)
        std  = np.array(IMAGENET_DEFAULT_STD,  dtype=np.float32)

        for im in images:
            if isinstance(im, str):
                im = cv2.imread(im)
            if im is None:
                outs_all.append([]); continue

            # BGR -> RGB
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            h, w = im.shape[:2]

            # Letterbox al tamaño del modelo (pegado arriba-izq)
            scale = float(self.img_size) / float(max(h, w))
            nh, nw = int(round(h * scale)), int(round(w * scale))
            imr = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
            canvas = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)
            canvas[:nh, :nw] = imr.astype(np.float32)

            # Normalización y CHW
            x = canvas / 255.0
            x = (x - mean) / std
            x = np.transpose(x, (2, 0, 1))
            t = torch.from_numpy(x).unsqueeze(0).to(self.device)

            # Predicción con tolerancia a distinta API
            out = self.model(t)
            d0 = out[0] if isinstance(out, (list, tuple)) else out

            boxes, scores, labels = None, None, None
            if isinstance(d0, dict):
                boxes  = d0.get('boxes')
                scores = d0.get('scores')
                labels = d0.get('labels')
            elif torch.is_tensor(d0):
                arr = d0.detach().to('cpu').numpy()
                if arr.ndim == 2 and arr.shape[1] >= 6:
                    boxes  = arr[:, 0:4]
                    scores = arr[:, 4]
                    labels = arr[:, 5].astype(np.int32)
            elif isinstance(d0, (list, tuple)) and len(d0) >= 3:
                bb, sc, lb = d0[:3]
                boxes  = bb.detach().to('cpu').numpy() if torch.is_tensor(bb) else np.asarray(bb)
                scores = sc.detach().to('cpu').numpy() if torch.is_tensor(sc) else np.asarray(sc)
                labels = lb.detach().to('cpu').numpy().astype(np.int32) if torch.is_tensor(lb) else np.asarray(lb).astype(np.int32)

            if boxes is None or scores is None or labels is None:
                outs_all.append([]); continue

            # Des-letterbox
            boxes = boxes / max(scale, 1e-6)

            keep = (scores >= self.conf)
            dets = []
            for (x1, y1, x2, y2), s, cls in zip(boxes[keep], scores[keep], labels[keep]):
                # Filtrar índices fuera de rango de nombres (si los hay)
                if hasattr(self.model, 'names'):
                    names = getattr(self.model, 'names')
                    if isinstance(names, dict) and (int(cls) not in names):
                        continue
                # xyxy -> guardar xyxy por ahora; convertiremos a xywh al exportar COCO
                w_box = float(abs(x2 - x1)); h_box = float(abs(y2 - y1))
                if w_box <= 1 or h_box <= 1:  # descarta cajas degeneradas
                    continue
                dets.append({
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "score": float(s),
                    "cls": int(cls),
                    "cls_name": None
                })
            outs_all.append(dets)
        return outs_all