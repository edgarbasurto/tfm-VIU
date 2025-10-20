from __future__ import annotations
from typing import List, Dict, Any, Optional
import json
import os

def save_coco_detections_json(dets_per_image: Dict[str, List[Dict[str, Any]]], output_json: str) -> None:
    """Convierte detecciones en formato COCO y guarda a .json
    Formato por item: {image_id, category_id, bbox:[x,y,w,h], score}
    Nota: COCO usa [x,y,w,h] con x,y esquina sup-izq; convertimos desde [x1,y1,x2,y2].
    """
    coco_list = []
    for image_id, dets in dets_per_image.items():
        for d in dets:
            x1,y1,x2,y2 = d["bbox"]
            w = x2 - x1
            h = y2 - y1
            coco_list.append({
                "image_id": int(os.path.splitext(os.path.basename(image_id))[0]),
                "category_id": int(d.get("cls", 0)),
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "score": float(d.get("score", 0.0))
            })
    with open(output_json, "w") as f:
        json.dump(coco_list, f)

def evaluate_coco(ann_file: str, det_json: str) -> Optional[dict]:
    """Evalúa mAP con COCOeval si pycocotools está disponible. Devuelve dict resumido."""
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except Exception as e:
        print("[WARN] pycocotools no disponible:", e)
        return None

    coco_gt = COCO(ann_file)
    coco_dt = coco_gt.loadRes(det_json)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Resumen típico: AP@[.5:.95], AP50, AP75, APs, APm, APl
    stats = coco_eval.stats
    return {
        "AP@[.5:.95]": float(stats[0]),
        "AP50": float(stats[1]),
        "AP75": float(stats[2]),
        "APs": float(stats[3]),
        "APm": float(stats[4]),
        "APl": float(stats[5]),
    }
