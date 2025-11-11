#!/usr/bin/env python3
import os, pandas as pd
import matplotlib.pyplot as plt

RUN_DIR = "runs/frcnn_mbv3_subset"
CSV = os.path.join(RUN_DIR, "frcnn_history.csv")
OUT_DIR = "outputs/figures"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV)
# Asegurar tipos numéricos
for c in ["epoch","train_loss","val_loss","AP","AP50","AP75","lr","epoch_time_s"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# 1) Pérdidas + LR
fig, ax1 = plt.subplots(figsize=(8,4.5))
ax1.plot(df["epoch"], df["train_loss"], label="Train loss")
ax1.plot(df["epoch"], df["val_loss"],   label="Val loss")
ax1.set_xlabel("Época"); ax1.set_ylabel("Pérdida"); ax1.grid(True, alpha=.3)
ax2 = ax1.twinx()
ax2.plot(df["epoch"], df["lr"], linestyle="--", label="LR")
ax2.set_ylabel("Learning rate")
lns = ax1.get_lines() + ax2.get_lines()
labels = [l.get_label() for l in lns]
ax1.legend(lns, labels, loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "frcnn_loss_lr.png"), dpi=180)

# 2) AP por época
plt.figure(figsize=(8,4.5))
plt.plot(df["epoch"], df["AP"],  label="mAP@[.5:.95]")
plt.plot(df["epoch"], df["AP50"], label="mAP@0.5")
plt.plot(df["epoch"], df["AP75"], label="mAP@0.75")
plt.xlabel("Época"); plt.ylabel("mAP"); plt.grid(True, alpha=.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "frcnn_ap_curves.png"), dpi=180)

print("[OK] Figuras guardadas en outputs/figures/: frcnn_loss_lr.png, frcnn_ap_curves.png")