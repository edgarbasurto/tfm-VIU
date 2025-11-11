#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="metrics_effdet_d0.csv")
    ap.add_argument("--out", default="outputs/figures/effdet_d0_loss_lr.png")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df = pd.read_csv(args.csv)
    # columnas esperadas: epoch, train_loss, val_loss, lr
    e  = df["epoch"]
    tl = df["train_loss"].astype(float)
    vl = df["val_loss"].astype(float)
    lr = df["lr"].astype(float)

    fig, ax1 = plt.subplots(figsize=(8,4.6))
    ax1.plot(e, tl, label="train_loss", linewidth=2)
    ax1.plot(e, vl, label="val_loss",   linewidth=2)
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Pérdida")
    ax1.grid(True, alpha=.25)
    ax1.legend(loc="upper right")

    ax2 = ax1.twinx()
    ax2.plot(e, lr, linestyle="--", alpha=.6, label="LR")
    ax2.set_ylabel("LR (escala original)")

    # título breve
    fig.suptitle("EfficientDet-D0: Curvas de pérdida y LR")
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print("[SAVE]", args.out)

if __name__ == "__main__":
    main()