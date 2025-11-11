import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse, os, json, glob
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.coco import load_subset
from src.eval.metrics import evaluate_coco
from src.eval.coco_utils import (
    load_coco_categories, build_index_to_catid_map,
    per_class_ap_from_cocoeval, pr_curve_from_cocoeval
)
from src.viz.plots import pareto_map_fps, ap_bar_per_class, pr_curve
from src.bench.latency import measure_fps
import inspect


def resolve_device(requested: str | None) -> str:
    import torch
    if requested in (None, "", "auto"):
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA no disponible; usando CPU.")
        return "cpu"
    if requested == "mps":
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        print("[WARN] MPS no disponible; usando CPU.")
        return "cpu"
    return requested


def load_model(kind: str, imgsz: int, conf: float, iou: float, device: str, half: bool, weights: str | None = None):
    kind = kind.lower()

    if kind.startswith('yolo-v8'):
        from src.models.yolo import YOLOv8Detector
        w = weights or ('yolov8n.pt' if kind == 'yolo-v8n' else 'yolov8s.pt')
        return YOLOv8Detector(weights=w, device=device, half=half, imgsz=imgsz, conf=conf, iou=iou)

    if kind == 'frcnn-r50-fpn':
        from src.models.frcnn import FasterRCNNDetector
        try:
            return FasterRCNNDetector(device=device, conf=conf, weights=weights)
        except TypeError:
            m = FasterRCNNDetector(device=device, conf=conf)
            if weights:
                import torch
                m.model.load_state_dict(torch.load(weights, map_location=device), strict=False)
            return m

    if kind == 'frcnn-mbv3-fpn':
        from src.models.frcnn_mbv3 import FasterRCNN_MBV3_FPN
        try:
            return FasterRCNN_MBV3_FPN(device=device, conf=conf, weights=weights)
        except TypeError:
            m = FasterRCNN_MBV3_FPN(device=device, conf=conf)
            if weights:
                import torch
                m.model.load_state_dict(torch.load(weights, map_location=device), strict=False)
            return m

    if kind == 'effdet-d0':
        from src.models.effdet import EfficientDetDetector
        ctor = EfficientDetDetector
        sig = inspect.signature(ctor).parameters
        kw = dict(variant='tf_efficientdet_d0', device=device, conf=conf)
        if 'ckpt' in sig: kw['ckpt'] = weights
        elif 'weights' in sig: kw['weights'] = weights
        for name in ('img_size', 'image_size', 'train_img_size'):
            if name in sig:
                kw[name] = imgsz
                break
        return ctor(**kw)

    raise NotImplementedError(f"Modelo no soportado: {kind}")


def load_image_cv2(path: str):
    import cv2
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return img


def reconcile_subset(image_paths: List[str], images_dir: str) -> List[str]:
    if all(os.path.exists(p) for p in image_paths):
        return image_paths
    print("[WARN] Algunas rutas del subset no existen; reconciliando con --images-dir")
    pool = {os.path.basename(p): p for p in glob.glob(os.path.join(images_dir, "*.jpg"))}
    pool.update({os.path.basename(p): p for p in glob.glob(os.path.join(images_dir, "*.JPG"))})
    repaired = []
    for p in image_paths:
        repaired.append(p if os.path.exists(p) else pool.get(os.path.basename(p), p))
    image_paths = [p for p in repaired if os.path.exists(p)]
    if not image_paths:
        raise FileNotFoundError("No se pudieron reconciliar rutas del subset con --images-dir")
    return image_paths


def parse_device_map(s: str) -> Dict[str, str]:
    """Ej: 'frcnn-r50-fpn:cpu,effdet-d0:cpu' -> {'frcnn-r50-fpn':'cpu','effdet-d0':'cpu'}"""
    m = {}
    for kv in (s.split(',') if s else []):
        if ':' in kv:
            k, v = kv.split(':', 1)
            m[k.strip().lower()] = v.strip().lower()
    return m

def parse_map(s: str) -> Dict[str, str]:
    m={}
    for kv in (s.split(',') if s else []):
        if ':' in kv:
            k,v = kv.split(':',1)
            m[k.strip().lower()] = v.strip()
    return m

def save_coco_json_safe(dets_per_image: Dict[str, List[Dict[str, Any]]],
                        out_json: str,
                        valid_cat_ids: set[int]) -> int:
    """Convierte detecciones (xyxy, cls, score) a lista COCO (xywh), filtrando ids/cajas inválidas.
       Devuelve cuántas detecciones se guardaron."""
    import re, json
    results = []
    valid_cls, invalid_cls, tiny_boxes = 0, 0, 0
    pat = re.compile(r'(\d{6,12})')
    for path, dets in dets_per_image.items():
        m = pat.search(os.path.basename(path))
        if not m:
            continue
        image_id = int(m.group(1))
        for d in dets:
            if 'bbox' not in d or 'score' not in d or 'cls' not in d:
                continue
            x1, y1, x2, y2 = map(float, d['bbox'])
            x_min, y_min = min(x1, x2), min(y1, y2)
            w = abs(x2 - x1); h = abs(y2 - y1)
            if w <= 1 or h <= 1 or not np.isfinite(w + h):
                tiny_boxes += 1
                continue
            cat = int(d['cls'])
            if valid_cat_ids and (cat not in valid_cat_ids):
                invalid_cls += 1
                continue
            score = float(d['score'])
            if not np.isfinite(score):
                continue
            results.append({
                "image_id": image_id,
                "category_id": cat,
                "bbox": [x_min, y_min, w, h],
                "score": score
            })
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f)
    print(f"[SAVE] {len(results)} dets OK | {invalid_cls} cls inválidas | {tiny_boxes} cajas muy pequeñas")
    return len(results)

def get_names_dict_from_model(wrapper_obj):
    """
    Regresa un dict {idx: name} desde distintas estructuras:
    - wrapper.names
    - wrapper.model.names
    - wrapper.model.model.names (caso YOLOv8)
    Acepta list -> lo convierte a dict.
    """
    chains = [
        ("names",),
        ("model", "names"),
        ("model", "model", "names"),
    ]
    for chain in chains:
        obj = wrapper_obj
        ok = True
        for a in chain:
            obj = getattr(obj, a, None)
            if obj is None:
                ok = False
                break
        if ok:
            if isinstance(obj, list):
                return {i: n for i, n in enumerate(obj)}
            if isinstance(obj, dict):
                return obj
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', required=True, help='yolo-v8n yolo-v8s frcnn-r50-fpn frcnn-mbv3-fpn effdet-d0')
    ap.add_argument('--images-dir', required=True)
    ap.add_argument('--ann-file', required=True)
    ap.add_argument('--subset', default=None)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--conf', type=float, default=0.35)
    ap.add_argument('--iou', type=float, default=0.7)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--half', action='store_true')
    ap.add_argument('--max-images', type=int, default=300)
    ap.add_argument('--outputs', default='outputs')
    ap.add_argument('--stride', type=int, default=1, help='Usar 1 de cada N imágenes (acelera pruebas)')
    ap.add_argument('--batch', type=int, default=8, help='Tamaño de lote para inferencia (si el modelo lo soporta)')
    ap.add_argument('--device-map', default='', help='Overrides por modelo, ej: "frcnn-r50-fpn:cpu,effdet-d0:cpu"')
    ap.add_argument('--conf-map',   default='', help='Conf por modelo, ej: "yolo-v8n:0.001,frcnn-mbv3-fpn:0.001"')
    ap.add_argument('--imgsz-map',  default='', help='Img size por modelo, ej: "yolo-v8n:640,frcnn-mbv3-fpn:800"')
    ap.add_argument('--maxdet-map', default='', help='Max dets por modelo, ej: "yolo-v8n:300,frcnn-mbv3-fpn:300"')
    ap.add_argument('--weights-map', default='',
               help='Rutas por modelo: "yolo-v8n:/ruta/best.pt,frcnn-mbv3-fpn:/ruta/best.pth,effdet-d0:/ruta/best.pth"')
    args = ap.parse_args()

    weights_map = parse_map(args.weights_map)

    conf_map   = parse_map(args.conf_map)
    imgsz_map  = parse_map(args.imgsz_map)
    maxdet_map = parse_map(args.maxdet_map)

    dev_map = parse_device_map(args.device_map)
    args.device = resolve_device(args.device)
    print(f"[INFO] Dispositivo por defecto: {args.device} (usa --device-map para overrides por modelo)")

    os.makedirs(os.path.join(args.outputs, 'figures'), exist_ok=True)
    os.makedirs(os.path.join(args.outputs, 'tables'), exist_ok=True)

    image_paths = load_subset(args.subset, args.images_dir)
    image_paths = reconcile_subset(image_paths, args.images_dir)
    if args.max_images and len(image_paths) > args.max_images:
        image_paths = image_paths[:args.max_images]
    if args.stride > 1:
        image_paths = image_paths[::args.stride]
        print(f"[INFO] Usando 1 de cada {args.stride} imágenes del subset (n={len(image_paths)}).")

    id_to_name, name_to_id = load_coco_categories(args.ann_file)
    valid_cat_ids = set(name_to_id.values())

    rows, pareto_points = [], []
    best_model, best_map, best_eval = None, -1.0, None

    for kind in args.models:
        k_lower = kind.lower()
        requested = dev_map.get(k_lower, args.device)
        device_for_model = resolve_device(requested)
        print(f"\n=== Modelo: {kind} | device={device_for_model} ===")

        try:
            conf_for_model  = float(conf_map.get(k_lower, args.conf))
            imgsz_for_model = int(imgsz_map.get(k_lower, args.imgsz))
            maxdet_for_model = int(maxdet_map.get(k_lower, 300))
            print(f"[INFO] Hparams {kind}: conf={conf_for_model} imgsz={imgsz_for_model} max_det={maxdet_for_model}")
            
            weights_for_model = weights_map.get(k_lower)
            model = load_model(k_lower, imgsz_for_model, conf_for_model, args.iou, device_for_model, args.half, weights=weights_for_model)
            
            if hasattr(model, "max_det"):
                try: model.max_det = maxdet_for_model
                except Exception: pass
            # Si el modelo es FRCNN y expone detections_per_img:
            try:
                model.model.roi_heads.detections_per_img = maxdet_for_model
            except Exception:
                pass

            # nombres del modelo para mapear idx -> category_id por NOMBRE
            model_names = get_names_dict_from_model(model)
            idx_to_cat = {}
            if isinstance(model_names, dict) and model_names:
                idx_to_cat = build_index_to_catid_map(model_names, name_to_id)
            print(f"[INFO] Mapeo clases {kind}: {len(idx_to_cat)} categorías enlazadas.")

            # INFERENCIA EN LOTES
            dets_per_image: Dict[str, Any] = {}
            B = max(1, args.batch)
            for bi in tqdm(range(0, len(image_paths), B), desc=f'Inferencia {kind}'):
                paths_chunk = image_paths[bi:bi+B]
                imgs_chunk = [load_image_cv2(p) for p in paths_chunk]
                preds_list = model.predict(imgs_chunk)
                for p, preds in zip(paths_chunk, preds_list):
                    if idx_to_cat:
                        for d in preds:
                            d['cls'] = idx_to_cat.get(int(d.get('cls', 0)), int(d.get('cls', 0)))
                    dets_per_image[p] = preds

            # FPS
            sample_imgs = [load_image_cv2(p) for p in image_paths[:min(10, len(image_paths))]]
            fps = measure_fps(model, sample_imgs, warmup=10, iters=50, device=device_for_model)

            # Guardar detecciones (COCO seguro) y evaluar si hay contenido
            det_json = os.path.join(args.outputs, f'detections_{k_lower}.json')
            saved = save_coco_json_safe(dets_per_image, det_json, valid_cat_ids)
            if saved == 0:
                print("[WARN] No se guardaron detecciones válidas; omito evaluación COCO.")
                metrics = {}
                mAP = 0.0
            else:
                metrics = evaluate_coco(args.ann_file, det_json) or {}
                mAP = float(metrics.get('AP@[.5:.95]', 0.0))

            rows.append({'Modelo': kind, 'imgsz': args.imgsz, 'FPS': round(fps, 2), **metrics})
            pareto_points.append({'name': kind, 'FPS': float(fps), 'mAP': mAP})

            # PR/AP del mejor
            if mAP > best_map:
                best_map, best_model = mAP, kind
                from pycocotools.coco import COCO
                from pycocotools.cocoeval import COCOeval
                coco_gt = COCO(args.ann_file)
                coco_dt = coco_gt.loadRes(det_json)
                coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
                coco_eval.evaluate(); coco_eval.accumulate(); coco_eval.summarize()
                best_eval = coco_eval

        except Exception as e:
            print(f"[ERROR] Modelo {kind} falló: {e}")
            rows.append({'Modelo': kind, 'imgsz': args.imgsz, 'FPS': float('nan')})
            continue

    # Tabla comparativa
    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.outputs, 'tables', 'tabla_comparativa.csv')
    tex_path = os.path.join(args.outputs, 'tables', 'tabla_comparativa.tex')
    df.to_csv(csv_path, index=False)
    df.to_latex(tex_path, index=False, float_format='%.3f',
                caption='Comparativa de modelos en subconjunto COCO.',
                label='tab:comparativa')

    # Pareto
    if pareto_points:
        pareto_map_fps(pareto_points, os.path.join(args.outputs, 'figures', 'pareto_map_fps_all.pdf'))

    # PR global y AP por clase del mejor
    if best_eval is not None:
        r, p = pr_curve_from_cocoeval(best_eval, iou_mode='avg')
        if r is not None:
            mask = ~np.isnan(p)
            pr_curve(r[mask], p[mask], os.path.join(args.outputs, 'figures', 'pr_global_best.pdf'),
                     title=f'PR global (mejor: {best_model})')
        ap_dict = per_class_ap_from_cocoeval(best_eval)
        if ap_dict:
            top = dict(sorted(ap_dict.items(), key=lambda kv: kv[1], reverse=True)[:20])
            ap_bar_per_class(top, os.path.join(args.outputs, 'figures', 'ap_top20_best.pdf'))

    print("\nListo. Salidas:")
    print("- outputs/tables/tabla_comparativa.{tex,csv}")
    print("- outputs/figures/pareto_map_fps_all.pdf (si hubo puntos)")
    print("- outputs/figures/pr_global_best.pdf, ap_top20_best.pdf (si hubo best_eval)")


if __name__ == '__main__':
    main()