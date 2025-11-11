#!/usr/bin/env python3
import argparse, time
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from effdet import create_model

def build_predictor(img_size, num_classes, device):
    m = create_model(
        'tf_efficientdet_d0',
        bench_task='predict',
        num_classes=num_classes,
        image_size=(img_size, img_size),
        pretrained=False
    ).to(device)
    m.eval()
    return m

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images-dir', required=True)
    ap.add_argument('--ids-file', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--train-img-size', type=int, default=512)  # tamaño usado al ENTRENAR el ckpt
    ap.add_argument('--img-size', type=int, default=640)        # tamaño objetivo para medir FPS
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--runs', type=int, default=1)
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.set_num_threads(max(1, args.threads))

    # 1) Carga ckpt en modelo al tamaño de entrenamiento (512)
    model_train = build_predictor(args.train_img_size, num_classes=80, device=device)
    sd = torch.load(args.ckpt, map_location=device)
    model_train.load_state_dict(sd, strict=False)  # los "Unexpected keys" son normales

    # 2) Reinstancia a 640 y copia pesos (sin anchors.*)
    if args.img_size != args.train_img_size:
        model = build_predictor(args.img_size, num_classes=80, device=device)
        sd2 = {k: v for k, v in model_train.state_dict().items() if not k.startswith('anchors.')}
        model.load_state_dict(sd2, strict=False)
    else:
        model = model_train

    # Lista de imágenes
    ids = [l.strip() for l in open(args.ids_file) if l.strip()]
    paths = [str(Path(args.images_dir) / p) for p in ids]

    # Warmup
    for _ in range(max(args.warmup, 0)):
        img = Image.open(paths[0]).convert('RGB')
        x = (pil_to_tensor(img).float() / 255.0).unsqueeze(0).to(device)
        x = F.interpolate(x, size=(args.img_size, args.img_size), mode='bilinear', align_corners=False)
        _ = model(x)

    # Cronometría
    n_images = len(paths) * max(1, args.runs)
    t0 = time.time()
    for _ in range(max(1, args.runs)):
        for p in paths:
            img = Image.open(p).convert('RGB')
            x = (pil_to_tensor(img).float() / 255.0).unsqueeze(0).to(device)
            x = F.interpolate(x, size=(args.img_size, args.img_size), mode='bilinear', align_corners=False)
            _ = model(x)
    dt = time.time() - t0
    fps = n_images / dt if dt > 0 else 0.0
    print(f"[EFFDET FPS] imgsz={args.img_size} images={n_images} time={dt:.3f}s -> FPS={fps:.2f}")

if __name__ == '__main__':
    main()