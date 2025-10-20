from __future__ import annotations
from typing import Dict, Tuple
import json
import numpy as np

def load_coco_categories(ann_file: str) -> Tuple[Dict[int, str], Dict[str, int]]:
    data = json.load(open(ann_file, 'r'))
    id_to_name = {c['id']: c['name'] for c in data['categories']}
    name_to_id = {v: k for k, v in id_to_name.items()}
    return id_to_name, name_to_id

def build_index_to_catid_map(model_names: Dict[int, str], name_to_id: Dict[str, int]) -> Dict[int, int]:
    aliases = {
        'aeroplane': 'airplane', 'motorbike': 'motorcycle', 'sofa': 'couch',
        'tvmonitor': 'tv', 'cellphone': 'cell phone', 'cell-phone': 'cell phone',
        'diningtable': 'dining table', 'pottedplant': 'potted plant',
        'hairdryer': 'hair drier', 'hair-drier': 'hair drier',
        'traffic-light': 'traffic light', 'sports-ball': 'sports ball',
        'tennis-racket': 'tennis racket', 'wineglass': 'wine glass',
        'teddy-bear': 'teddy bear',
    }
    mapping = {}
    for idx, raw in model_names.items():
        key = str(raw).strip().lower().replace('-', ' ').replace('_', ' ')
        key = aliases.get(key, key)
        if key in name_to_id:
            mapping[idx] = name_to_id[key]
    return mapping

def per_class_ap_from_cocoeval(coco_eval) -> Dict[str, float]:
    precision = coco_eval.eval['precision']  # [T, R, K, A, M]
    if precision is None:
        return {}
    T, R, K, A, M = precision.shape
    aps = precision.mean(axis=(0, 1))[:, 0, -1]  # [K]
    id_to_name = {c['id']: c['name'] for c in coco_eval.cocoGt.dataset['categories']}
    ap_dict = {}
    for k, catId in enumerate(coco_eval.params.catIds):
        name = id_to_name.get(catId, str(catId))
        v = float(aps[k]) if not np.isnan(aps[k]) else 0.0
        ap_dict[name] = v
    return ap_dict

def pr_curve_from_cocoeval(coco_eval, iou_mode: str = 'avg'):
    precision = coco_eval.eval['precision']  # [T, R, K, A, M]
    if precision is None:
        return None, None
    T, R, K, A, M = precision.shape
    P = precision[:, :, :, 0, -1]  # área=all, maxDet último  -> [T, R, K]
    Pk = np.nanmean(P, axis=2)     # promedio sobre clases -> [T, R]
    ious = np.array(coco_eval.params.iouThrs)
    if iou_mode == '50':
        i = np.argmin(np.abs(ious - 0.50)); pr = Pk[i]
    elif iou_mode == '75':
        i = np.argmin(np.abs(ious - 0.75)); pr = Pk[i]
    else:
        pr = np.nanmean(Pk, axis=0)
    recall = np.array(coco_eval.params.recThrs)
    return recall, pr
