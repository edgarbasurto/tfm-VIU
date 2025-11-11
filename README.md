# TFM Benchmark — Detección de objetos (COCO)

Pipeline mínimo y **reproducible** para comparar detectores (YOLOv8, Faster R-CNN, EfficientDet) sobre **COCO val2017**, generando **tablas (.tex)** y **figuras (.pdf)** listas para Overleaf.

## Estructura

```
tfm-benchmark/
  configs/                 # JSON/auxiliares de subsets (opcional)
  scripts/                 # puntos de entrada (CLI)
  src/                     # código fuente modular
    data/                  # carga de datos / subsets
    models/                # wrappers con interfaz común (YOLOv8, FRCNN, EffDet)
    eval/                  # métricas (COCO mAP) y utilidades
    viz/                   # gráficos (PR, AP por clase, Pareto)
    bench/                 # medición de latencia/FPS
  outputs/
    figures/               # PDFs/PNGs exportados
    tables/                # .tex y .csv para LaTeX
  overleaf/                # (opcional) plantillas de inserción LaTeX
```

## Requisitos

- Python 3.11
- COCO **val2017** disponible localmente:
  ```
  /datasets/coco/
    val2017/                              # imágenes (.jpg)
    annotations/instances_val2017.json    # anotaciones
  ```

## Instalación rápida

```bash
python -m venv .venv && source .venv/bin/activate  # en macOS/Linux
# .\.venv\Scripts\activate                      # en Windows

pip install -r requirements.txt

```

## Datos: subset opcional

Puedes evaluar todo **val2017** o trabajar con un subset. Si no pasas `--subset`, el pipeline usa el directorio completo.

```bash
python scripts/make_subset.py   --images-dir /datasets/coco/val2017   --output configs/subset_val2017.json   --max-images 1000
```

> Si no tienes `make_subset.py`, puedes omitir `--subset` y usar `--max-images` y/o `--stride` en `run_compare.py`.

## Evaluación y comparación (mAP + FPS)

`run_compare.py` ejecuta inferencia multi-modelo, guarda detecciones en COCO JSON, evalúa con COCO API y produce tablas/figuras.

### Ejecución típica (3 modelos)

```bash
python scripts/run_compare.py   --models yolo-v8n frcnn-mbv3-fpn effdet-d0   --images-dir /ruta/a/coco/val2017   --ann-file   /ruta/a/coco/annotations/instances_val2017.json   --imgsz 640   --conf 0.001 --iou 0.50   --max-images 0 \                   # 0 = usar todas las imágenes
  --stride 1 \                       # usa 1 de cada N (para acelerar pruebas)
  --outputs outputs_final   --device-map "yolo-v8n:cpu,frcnn-mbv3-fpn:cpu,effdet-d0:cpu"   --imgsz-map  "yolo-v8n:640,frcnn-mbv3-fpn:640,effdet-d0:512"   --maxdet-map "yolo-v8n:300,frcnn-mbv3-fpn:300,effdet-d0:300"   --conf-map   "yolo-v8n:0.001,frcnn-mbv3-fpn:0.001,effdet-d0:0.001"   --weights-map "yolo-v8n:runs/train_yolov8n_subset/weights/best.pt,frcnn-mbv3-fpn:runs/frcnn_mbv3_subset/best.pth,effdet-d0:runs/train_effdet_d0_70e_img512/best.pth"
```

### Artefactos generados

- **Tablas**
  - `outputs/tables/tabla_comparativa.csv`
  - `outputs/tables/tabla_comparativa.tex`

- **Figuras**
  - `outputs/figures/pareto_map_fps_all.pdf`   (mAP vs FPS)
  - `outputs/figures/pr_global_best.pdf`       (PR del mejor mAP)
  - `outputs/figures/ap_top20_best.pdf`        (Top-20 AP por clase)

- **Detecciones COCO JSON**
  - `outputs_final/detections_<modelo>.json`

> El script también calcula FPS con un micro-benchmark interno y escoge automáticamente el “mejor” modelo (mAP) para dibujar PR/AP.

## Notas importantes por modelo

- **YOLOv8** (Ultralytics): usa el wrapper `src/models/yolo.py`.
  Pasa `--weights-map yolo-v8n:/runs/train_yolov8n_subset/best.pt` para usar tu checkpoint entrenado.

- **Faster R-CNN (MBV3-FPN)**: el wrapper ajusta `detections_per_img` si está disponible.

- **EfficientDet-D0**:
  - El wrapper `src/models/effdet.py` ya **corrige** el caso de cajas normalizadas `[0,1]` y hace **letterbox + des-letterbox** a `image_size` del modelo (D0 = 512).

## Solo medir FPS de EfficientDet (micro-benchmark)

```bash
python scripts/bench_effdet_fps.py   --images-dir /datasets/coco/val2017   --ids-file assets/qualitative_ids.txt   --ckpt runs/train_effdet_d0_70e_img512/best.pth   --train-img-size 512   --img-size 512   --device cpu   --threads 1
```

## Parámetros útiles (mapas por modelo)

- `--device-map`  (ej. `"yolo-v8n:cpu,effdet-d0:cpu"`)
- `--imgsz-map`   (ej. `"effdet-d0:512"`)
- `--conf-map`    (ej. `"yolo-v8n:0.001"`) ← para COCO eval es común usar `0.001`
- `--maxdet-map`  (ej. `"yolo-v8n:300"`)

Si no se especifica un mapa, se usa el valor global (`--imgsz`, `--conf`, etc.).

## Cita del repositorio

Si utilizas este código en tu trabajo académico, cita el repositorio (BibTeX de ejemplo):

```bibtex
@misc{basurto2025tfmbenchmark,
  author       = {Basurto Cruz, Edgar Daniel},
  title        = {TFM Benchmark — Detección de objetos (COCO)},
  year         = {2025},
  howpublished = {\url{https://github.com/edgarbasurto/tfm-VIU}},
  note         = {Accedido: 2025-11-10}
}
```
