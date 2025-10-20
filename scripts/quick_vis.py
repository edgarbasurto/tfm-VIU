import os, random, argparse, json
import cv2
from pycocotools.coco import COCO

def draw_box(img, box, color, label=None):
    x,y,w,h = map(int, box)
    cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
    if label:
        cv2.putText(img, label, (x, max(0, y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images-dir', required=True)
    ap.add_argument('--ann-file', required=True)
    ap.add_argument('--det-json', required=True, help='detections_yolo-v8n.json, etc.')
    ap.add_argument('--outdir', default='outputs/qualitative')
    ap.add_argument('--num', type=int, default=12)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    coco = COCO(args.ann_file)
    cats = {c['id']: c['name'] for c in coco.loadCats(coco.getCatIds())}
    dets = json.load(open(args.det_json))
    by_img = {}
    for d in dets:
        by_img.setdefault(d['image_id'], []).append(d)

    imgs = list(by_img.keys())
    random.shuffle(imgs)
    imgs = imgs[:args.num]

    for image_id in imgs:
        iminfo = coco.loadImgs([image_id])[0]
        path = os.path.join(args.images_dir, iminfo['file_name'])
        img = cv2.imread(path)
        if img is None: 
            continue

        # GT en verde
        ann_ids = coco.getAnnIds(imgIds=[image_id], iscrowd=None)
        anns = coco.loadAnns(ann_ids)
        for a in anns:
            draw_box(img, a['bbox'], (0,255,0), cats.get(a['category_id'], str(a['category_id'])))

        # Pred en rojo
        for d in by_img.get(image_id, []):
            draw_box(img, d['bbox'], (0,0,255), cats.get(d['category_id'], str(d['category_id'])) + f" {d['score']:.2f}")

        out = os.path.join(args.outdir, f"{image_id}.jpg")
        cv2.imwrite(out, img)
        print("saved", out)

if __name__ == "__main__":
    main()