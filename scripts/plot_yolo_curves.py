# scripts/plot_yolo_curves.py
import argparse, os, pandas as pd, matplotlib.pyplot as plt
ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True)
ap.add_argument("--out", default="outputs/figures/yolov8n_loss_map.png")
args = ap.parse_args()
os.makedirs(os.path.dirname(args.out), exist_ok=True)
df = pd.read_csv(args.csv)
e  = df["epoch"]
tl = df["train/box_loss"] + df["train/cls_loss"] + df.get("train/dfl_loss", 0)
vl = df["val/box_loss"]   + df["val/cls_loss"]   + df.get("val/dfl_loss", 0)
map5 = df.get("metrics/mAP50(B)", None)

fig, ax1 = plt.subplots(figsize=(8,4.6))
ax1.plot(e, tl, label="train_loss", linewidth=2)
ax1.plot(e, vl, label="val_loss", linewidth=2)
ax1.set_xlabel("Época"); ax1.set_ylabel("Pérdida"); ax1.grid(True, alpha=.25)
ax1.legend(loc="upper right")
if map5 is not None:
    ax2 = ax1.twinx(); ax2.plot(e, map5, linestyle="--", alpha=.6, label="mAP@0.5")
    ax2.set_ylabel("mAP@0.5")
fig.suptitle("YOLOv8n: Curvas de pérdida (y mAP)")
fig.tight_layout(); fig.savefig(args.out, dpi=200); print("[SAVE]", args.out)