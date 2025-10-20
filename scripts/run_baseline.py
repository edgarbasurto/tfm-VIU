import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import argparse, os, json
from typing import Dict, Any
import numpy as np
from tqdm import tqdm
import pandas as pd

from src.data.coco import load_subset
from src.eval.metrics import save_coco_detections_json, evaluate_coco
from src.viz.plots import pareto_map_fps
from src.bench.latency import measure_fps

def load_model(kind: str, imgsz: int, conf: float, iou: float, device: str, half: bool):
    if kind.startswith('yolo-v8'):
        from src.models.yolo import YOLOv8Detector
        weights = 'yolov8n.pt' if kind == 'yolo-v8n' else 'yolov8s.pt'
        return YOLOv8Detector(weights=weights, device=device, half=half, imgsz=imgsz, conf=conf, iou=iou)
    raise NotImplementedError(f"Modelo no soportado aún: {kind}")

def load_image_cv2(path: str):
    import cv2
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    # ulralytics maneja BGR/RGB internamente, podemos pasar BGR
    return img

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='yolo-v8n | yolo-v8s (más adelante: yolo-v5s, frcnn-r50-fpn, effdet-d0)')
    ap.add_argument('--images-dir', required=True)
    ap.add_argument('--ann-file', default=None, help='annotations COCO; si se omite, no se calcula mAP')
    ap.add_argument('--subset', default=None, help='JSON con lista de imágenes')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--iou', type=float, default=0.7)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--half', action='store_true')
    ap.add_argument('--max-images', type=int, default=200, help='límite duro de imágenes para una corrida rápida')
    ap.add_argument('--outputs', default='outputs')
    args = ap.parse_args()

    os.makedirs(os.path.join(args.outputs, 'figures'), exist_ok=True)
    os.makedirs(os.path.join(args.outputs, 'tables'), exist_ok=True)

    model = load_model(args.model, args.imgsz, args.conf, args.iou, args.device, args.half)

    image_paths = load_subset(args.subset, args.images_dir)
    if args.max_images and len(image_paths) > args.max_images:
        image_paths = image_paths[:args.max_images]

    # Correr inferencia y recolectar detecciones
    dets_per_image: Dict[str, Any] = {}
    sample_imgs = []
    for p in tqdm(image_paths, desc='Inferencia'):
        img = load_image_cv2(p)
        sample_imgs.append(img)
        preds = model.predict([img])[0]
        dets_per_image[p] = preds

    # Medir FPS (aproximado) con una muestra de imágenes
    if len(sample_imgs) > 10:
        sample_imgs = sample_imgs[:10]
    fps = measure_fps(model, sample_imgs, warmup=10, iters=50, device=args.device)

    # Guardar detecciones COCO y evaluar (si hay anotaciones)
    det_json = os.path.join(args.outputs, 'detections.json')
    save_coco_detections_json(dets_per_image, det_json)

    metrics = {}
    if args.ann_file:
        m = evaluate_coco(args.ann_file, det_json)
        if m:
            metrics = m

    # Exportar tabla .tex sencilla
    df = pd.DataFrame([{
        'Modelo': args.model,
        'imgsz': args.imgsz,
        'FPS': round(fps, 2),
        **metrics
    }])
    csv_path = os.path.join(args.outputs, 'tables', 'tabla_baseline.csv')
    tex_path = os.path.join(args.outputs, 'tables', 'tabla_baseline.tex')
    df.to_csv(csv_path, index=False)
    df.to_latex(tex_path, index=False, float_format='%.3f', caption='Baseline de detección (subconjunto COCO).', label='tab:baseline')

    # Gráfico Pareto mAP vs FPS (si hay mAP)
    if 'AP@[.5:.95]' in metrics:
        pareto_map_fps([{'name': args.model, 'FPS': float(fps), 'mAP': float(metrics['AP@[.5:.95]'])}],
                       os.path.join(args.outputs, 'figures', 'pareto_map_fps.pdf'))

    # Guardar resumen JSON
    summary_json = os.path.join(args.outputs, 'baseline_summary.json')
    with open(summary_json, 'w') as f:
        json.dump({'model': args.model, 'fps': fps, 'metrics': metrics}, f, indent=2)

    print(f"Listo. Archivos en {args.outputs}/tables y {args.outputs}/figures")

if __name__ == '__main__':
    main()
