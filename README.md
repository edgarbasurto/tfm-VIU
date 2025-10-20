# TFM Benchmark – Detección de objetos (COCO)

Pipeline mínimo y reproducible para comparar modelos de detección (YOLOv8/YOLOv5, Faster R-CNN, EfficientDet)
sobre un subconjunto de COCO, generando **tablas (.tex)** y **figuras (.pdf)** listas para Overleaf.

## Estructura

```
tfm-benchmark/
  configs/                # YAMLs de configuración por experimento
  scripts/                # puntos de entrada (CLI)
  src/                    # código fuente modular
    data/                 # carga de datos / subsets
    models/               # wrappers de modelos con interfaz común
    eval/                 # métricas (COCO mAP) y ayudantes
    viz/                  # gráficos (PR, AP por clase, Pareto)
    bench/                # mediciones de latencia/FPS y memoria
  outputs/
    figures/              # PDFs/PNGs para el TFM
    tables/               # .tex exportados para insertar en LaTeX
  overleaf/               # plantillas de inserción LaTeX
  notebooks/              # tu notebook original y migraciones
```

## Primeros pasos

1. Crea tu entorno y ejecuta:

```bash
pip install -r requirements.txt
```

2. Prepara un subconjunto de COCO (o usa `--max-images` para pruebas rápidas):

```bash
python scripts/make_subset.py   --images-dir /ruta/a/coco/val2017   --output configs/subset_val2017.json   --max-images 1000
```

3. Corre un **baseline** con YOLOv8n (pre-entrenado) y guarda tablas/figuras listas para Overleaf:

```bash
python scripts/run_baseline.py   --model yolo-v8n   --images-dir /ruta/a/coco/val2017   --ann-file /ruta/a/coco/annotations/instances_val2017.json   --subset configs/subset_val2017.json   --device cuda   --imgsz 640
```

4. Publica en Overleaf (copia figuras y tablas). Ajusta `--overleaf-dir` a tu carpeta sincronizada:

```bash
python scripts/publish.py --overleaf-dir /ruta/a/tu/proyecto/overleaf
```

## Notas

- Si no tienes `pycocotools`, instala compiladores o usa `pip install pycocotools` en Linux/Mac. En Windows, sugiere WSL.
- Los scripts detectan opcionalmente `ann-file`. Si no lo pasas, omiten mAP y solo reportan FPS/latencia.
- Para edge, ejecuta el mismo `run_baseline.py` en el dispositivo y compara CSV/figuras.
