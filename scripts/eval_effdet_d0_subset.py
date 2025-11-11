#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from effdet import create_model
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def resolve_device(arg):
    if arg == 'auto':
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    return torch.device(arg)

def letterbox(im: np.ndarray, new_size=640, color=(114,114,114)):
    h, w = im.shape[:2]
    s = min(new_size / h, new_size / w)
    nh, nw = int(round(h * s)), int(round(w * s))
    im_rs = np.array(Image.fromarray(im).resize((nw, nh), Image.BILINEAR))
    canvas = np.full((new_size, new_size, 3), color, dtype=im_rs.dtype)
    top = (new_size - nh) // 2
    left = (new_size - nw) // 2
    canvas[top:top+nh, left:left+nw] = im_rs
    return canvas, s, left, top, (h, w)

class ValDataset(Dataset):
    def __init__(self, img_dir, ann_json):
        self.img_dir = Path(img_dir)
        self.coco = COCO(ann_json)
        self.ids = self.coco.getImgIds()
    def __len__(self): return len(self.ids)
    def __getitem__(self, idx):
        img_id = self.ids[idx]
        info = self.coco.loadImgs([img_id])[0]
        path = self.img_dir / info["file_name"]
        img = Image.open(path).convert("RGB")
        img_np = np.array(img, copy=True)
        lb, scale, padw, padh, (h0, w0) = letterbox(img_np, 640)
        img_t = torch.from_numpy(lb).permute(2,0,1).float()/255.0
        meta  = {'img_id': img_id, 'scale': scale, 'padw': padw, 'padh': padh, 'h0': h0, 'w0': w0}
        return img_t, meta

def collate_fn(batch):
    imgs, metas = zip(*batch)
    return torch.stack(imgs, 0), list(metas)

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--val-images', default='datasets/coco_yolo_subset/val/images')
    ap.add_argument('--ann-val',    default='datasets/coco_yolo_subset/annotations/instances_val_subset.json')
    ap.add_argument('--weights',    default='runs/train_effdet_d0_subset/best.pth')
    ap.add_argument('--device',     default='auto')
    ap.add_argument('--out-json',   default='runs/train_effdet_d0_subset/detections_val.json')
    args = ap.parse_args()

    device = resolve_device(args.device)
    ds = ValDataset(args.val_images, args.ann_val)
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_fn)

    model = create_model('tf_efficientdet_d0', bench_task='predict',
                         num_classes=80, image_size=(640,640), pretrained=False).to(device)
    state = torch.load(args.weights, map_location='cpu')
    model.load_state_dict(state, strict=True)
    model.eval()

    results = []
    for imgs, metas in dl:
        imgs = imgs.to(device)
        out = model(imgs)  # lista de dicts: boxes(xyxy), scores, labels
        for det, meta in zip(out, metas):
            boxes = det['boxes'].cpu().numpy()
            scores = det['scores'].cpu().numpy()
            labels = det['labels'].cpu().numpy()  # idx [0..79]
            if len(boxes) == 0: 
                continue
            # revertir letterbox -> coords del original
            scale, padw, padh, h0, w0 = meta['scale'], meta['padw'], meta['padh'], meta['h0'], meta['w0']
            b = boxes.copy()
            b[:, [0,2]] -= padw; b[:, [1,3]] -= padh
            b[:, [0,2]] /= scale; b[:, [1,3]] /= scale
            # clip a imagen original
            b[:, [0,2]] = b[:, [0,2]].clip(0, w0-1)
            b[:, [1,3]] = b[:, [1,3]].clip(0, h0-1)
            # xyxy -> xywh
            xywh = b.copy()
            xywh[:,2] = xywh[:,2] - xywh[:,0]
            xywh[:,3] = xywh[:,3] - xywh[:,1]
            for bb, sc, lb in zip(xywh, scores, labels):
                results.append({
                    "image_id": int(meta['img_id']),
                    "category_id": int(ds.coco.getCatIds()[lb]),  # idx->COCO id
                    "bbox": [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])],
                    "score": float(sc)
                })

    os.makedirs(Path(args.out_json).parent, exist_ok=True)
    with open(args.out_json, 'w') as f:
        json.dump(results, f)

    coco_gt = ds.coco
    coco_dt = coco_gt.loadRes(args.out_json)
    E = COCOeval(coco_gt, coco_dt, iouType='bbox')
    E.evaluate(); E.accumulate(); E.summarize()
    print(json.dumps({
        "AP@[.5:.95]": float(E.stats[0]),
        "AP50": float(E.stats[1]),
        "AP75": float(E.stats[2]),
    }, indent=2))

if __name__ == '__main__':
    main()