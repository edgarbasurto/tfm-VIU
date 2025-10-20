import argparse, json, os
import numpy as np
from src.eval.metrics import evaluate_coco

def refilter(in_json, out_json, thr):
    with open(in_json, 'r') as f:
        dets = json.load(f)
    dets2 = [d for d in dets if float(d.get('score', 0.0)) >= thr]
    with open(out_json, 'w') as f:
        json.dump(dets2, f)
    return len(dets), len(dets2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ann-file', required=True)
    ap.add_argument('--in-json', required=True)   # p.ej. outputs/detections_yolo-v8n.json
    ap.add_argument('--outdir', default='outputs/sweeps')
    ap.add_argument('--grid', default='0.001,0.005,0.01,0.02,0.03,0.05,0.1,0.2')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    thrs = [float(x) for x in args.grid.split(',')]

    rows = []
    for t in thrs:
        out_json = os.path.join(args.outdir, f"{os.path.splitext(os.path.basename(args.in_json))[0]}_thr{t:.3f}.json")
        n0, n1 = refilter(args.in_json, out_json, t)
        metrics = evaluate_coco(args.ann_file, out_json) or {}
        rows.append({'thr': t, 'n_dets': n1, **metrics})
        print(f"thr={t:.3f}  kept={n1}/{n0}  AP=.5:.95={metrics.get('AP@[.5:.95]', 0):.4f}  AP50={metrics.get('AP@.5', 0):.4f}")

    # guarda CSV rápido
    import pandas as pd
    df = pd.DataFrame(rows)
    csv = os.path.join(args.outdir, f"sweep_{os.path.splitext(os.path.basename(args.in_json))[0]}.csv")
    df.to_csv(csv, index=False)
    print("CSV:", csv)

if __name__ == "__main__":
    main()