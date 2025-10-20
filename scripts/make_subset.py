import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import argparse
from src.data.coco import make_subset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images-dir', required=True, help='Carpeta con imágenes .jpg (p.ej., COCO/val2017)')
    ap.add_argument('--output', required=True, help='Ruta de salida .json (lista de imágenes)')
    ap.add_argument('--max-images', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    make_subset(args.images_dir, args.output, args.max_images, args.seed)
    print(f"Subset guardado en {args.output}")

if __name__ == '__main__':
    main()
