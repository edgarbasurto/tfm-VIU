from __future__ import annotations
from typing import List, Dict, Any, Optional, Union
import numpy as np, torch

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

COCO80_NAMES = [ ... ]  # (mantén tu lista)

class EfficientDetDetector(Detector):
    """EfficientDet-D0 (effdet, COCO) tolerante a cambios de API."""
    def __init__(self,
                 variant: str = 'tf_efficientdet_d0',
                 device: Optional[str] = None,
                 conf: float = 0.25,
                 ckpt: Optional[str] = None,
                 weights: Optional[str] = None,
                 img_size: Optional[int] = None):
        # --- device ---
        if device:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'

        # --- crear modelo (bench predict si hay) ---
        try:
            bench = create_model(
                variant,
                pretrained=not bool(ckpt or weights),
                bench_task='predict',
                num_classes=None,
                image_size=img_size  # algunas versiones lo aceptan
            )
        except TypeError:
            bench = create_model(
                variant,
                pretrained=not bool(ckpt or weights),
                num_classes=None
            )
        if DetBenchPredict is not None and bench.__class__.__name__ != 'DetBenchPredict':
            try:
                bench = DetBenchPredict(bench)
            except Exception:
                pass
        self.model = bench

        # --- cargar checkpoint propio si llega ---
        sd_path = ckpt or weights
        if sd_path:
            try:
                state = torch.load(sd_path, map_location='cpu')
                self.model.load_state_dict(state, strict=False)
                print(f"[EFFDET] Pesos cargados desde {sd_path} (strict=False).")
            except Exception as e:
                print(f"[EFFDET] Aviso: no se pudo cargar {sd_path}: {e}")

        # --- mover a device ---
        try:
            self.model.to(self.device)
        except Exception:
            print("[WARN] EfficientDet no soporta este backend; usando CPU.")
            self.device = 'cpu'; self.model.to(self.device)
        self.model.eval()
        self.conf = conf

        # --- tamaño de entrada efectivo ---
        size_from_cfg = None
        try:
            cfg = getattr(self.model, 'config', None) or getattr(getattr(self.model, 'model', None), 'config', None)
            if cfg is not None:
                val = getattr(cfg, 'image_size', None)
                if isinstance(val, (list, tuple)): size_from_cfg = int(val[0])
                elif isinstance(val, int):         size_from_cfg = int(val)
        except Exception:
            pass
        self.img_size = int(img_size or size_from_cfg or 512)

        # nombres estilo YOLO para el mapeo por nombre
        try:
            self.model.names = {i: n for i, n in enumerate(COCO80_NAMES)}
        except Exception:
            pass

    def name(self) -> str:
        return "EfficientDet(D0)"

    @torch.no_grad()
    def predict(self, images: List[Union[str, np.ndarray]]):
        import cv2
        outs_all: List[List[Dict[str, Any]]] = []

        mean = np.array(IMAGENET_DEFAULT_MEAN, dtype=np.float32)
        std  = np.array(IMAGENET_DEFAULT_STD,  dtype=np.float32)

        for im in images:
            if isinstance(im, str):
                im = cv2.imread(im)
            if im is None:
                outs_all.append([]); continue

            # BGR->RGB y shape original
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            h, w = im.shape[:2]

            # letterbox top-left (sin offset) al tamaño del modelo
            scale = float(self.img_size) / float(max(h, w))
            nh, nw = int(round(h * scale)), int(round(w * scale))
            imr = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
            canvas = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)
            canvas[:nh, :nw] = imr.astype(np.float32)

            # normalización y CHW
            x = canvas / 255.0
            x = (x - mean) / std
            x = np.transpose(x, (2, 0, 1))
            t = torch.from_numpy(x).unsqueeze(0).to(self.device)

            # forward (distintas APIs)
            out = self.model(t)
            d0 = out[0] if isinstance(out, (list, tuple)) else out

            boxes, scores, labels = None, None, None
            if isinstance(d0, dict):
                boxes, scores, labels = d0.get('boxes'), d0.get('scores'), d0.get('labels')
                if torch.is_tensor(boxes):  boxes = boxes.detach().cpu().numpy()
                if torch.is_tensor(scores): scores = scores.detach().cpu().numpy()
                if torch.is_tensor(labels): labels = labels.detach().cpu().numpy().astype(np.int32)
            elif torch.is_tensor(d0):
                arr = d0.detach().cpu().numpy()
                if arr.ndim == 2 and arr.shape[1] >= 6:
                    boxes, scores, labels = arr[:, :4], arr[:, 4], arr[:, 5].astype(np.int32)
            elif isinstance(d0, (list, tuple)) and len(d0) >= 3:
                bb, sc, lb = d0[:3]
                boxes  = bb.detach().cpu().numpy() if torch.is_tensor(bb) else np.asarray(bb)
                scores = sc.detach().cpu().numpy() if torch.is_tensor(sc) else np.asarray(sc)
                labels = (lb.detach().cpu().numpy() if torch.is_tensor(lb) else np.asarray(lb)).astype(np.int32)

            if boxes is None or scores is None or labels is None:
                outs_all.append([]); continue

            # --- normalizado vs píxeles ---
            # Si parece normalizado (<=1.5), conviértelo a píxeles del canvas primero.
            if np.nanmax(boxes[:, :4]) <= 1.5:
                boxes = boxes * float(self.img_size)

            # des-letterbox al tamaño original
            boxes = boxes / max(scale, 1e-6)

            # clip a la imagen original
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w - 1)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h - 1)

            # filtrar por score y descartar cajas degeneradas
            keep = (scores >= self.conf)
            sel  = np.where(keep)[0]
            dets = []
            for i in sel:
                x1, y1, x2, y2 = boxes[i]
                if (x2 - x1) <= 1 or (y2 - y1) <= 1:
                    continue
                # valida idx de clase si se definieron names (para el mapeo por nombre)
                if hasattr(self.model, 'names'):
                    names = getattr(self.model, 'names')
                    if isinstance(names, dict) and int(labels[i]) not in names:
                        continue
                dets.append({'bbox': [float(x1), float(y1), float(x2), float(y2)],
                             'score': float(scores[i]),
                             'cls': int(labels[i]),
                             'cls_name': None})
            outs_all.append(dets)
        return outs_all