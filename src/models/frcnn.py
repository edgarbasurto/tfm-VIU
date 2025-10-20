from __future__ import annotations
from typing import List, Dict, Any, Optional, Union
import torch, torchvision, numpy as np
from .base import Detector


class FasterRCNNDetector(Detector):
    """Faster R-CNN ResNet50-FPN (TorchVision, COCO)."""

    def __init__(self, device: Optional[str] = None, conf: float = 0.25):
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)

        # Intentar usar MPS si está disponible; si no, caer a CPU
        if self.device == 'mps':
            try:
                self.model.to('mps')
            except Exception:
                print('[WARN] Faster R-CNN no soporta MPS en esta build; usando CPU.')
                self.device = 'cpu'

        self.model.to(self.device).eval()
        self.tf = weights.transforms()  # espera PIL.Image
        self.conf = conf

    def name(self) -> str:
        return "FasterRCNN(R50-FPN)"

    @torch.no_grad()
    def predict(self, images: List[Union[str, np.ndarray]]):
        import cv2
        from PIL import Image

        # 1) BGR->RGB  2) numpy -> PIL  3) preset transforms -> tensor
        tensors = []
        for im in images:
            if isinstance(im, str):
                im = cv2.imread(im)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(im)
            t = self.tf(pil).to(self.device)
            tensors.append(t)

        outputs = self.model(tensors)  # lista de dicts
        results: List[List[Dict[str, Any]]] = []

        for out in outputs:
            boxes = out['boxes'].detach().to('cpu').numpy()
            scores = out['scores'].detach().to('cpu').numpy()
            labels = out['labels'].detach().to('cpu').numpy().astype(int)  # category_id COCO
            keep = scores >= self.conf

            dets = [{
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "score": float(s),
                "cls": int(c),  # ya es category_id de COCO
                "cls_name": None
            } for (x1, y1, x2, y2), s, c in zip(boxes[keep], scores[keep], labels[keep])]
            results.append(dets)

        return results
