#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genera curvas sintéticas de entrenamiento para YOLOv8n y produce:
- figuras/curva_perdida_yolov8n.png
- figuras/curva_map_yolov8n.png
- figuras/curvas_entrenamiento_yolov8n.png  (combinada lado a lado)
- outputs/tables/curvas_entrenamiento_yolov8n.csv
- outputs/tables/curvas_yolov8n_resumen.csv
- outputs/tables/curvas_yolov8n_resumen.tex  (tabla LaTeX con 5 épocas)

Requisitos: numpy, pandas, matplotlib, pillow
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

def main():
    fig_dir = "figuras"
    tab_dir = os.path.join("outputs", "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)

    # Datos sintéticos reproducibles
    rng = np.random.default_rng(8)
    epochs = np.arange(1, 101)

    # Pérdida (train): decae con ruido leve
    base_loss = 2.8 * np.exp(-epochs / 42.0) + 0.06 * np.log1p(epochs)
    noise_loss = rng.normal(0, 0.03, size=epochs.size)
    train_loss = np.clip(base_loss + noise_loss, 0.04, None)

    # mAP@0.5 (validación): saturación suave con ruido leve
    base_map = 0.46 * (1 - np.exp(-epochs / 36.0)) + 0.015
    noise_map = rng.normal(0, 0.01, size=epochs.size)
    val_map50 = np.clip(base_map + noise_map, 0.0, 1.0)

    # Scheduler (cosine) típico
    lr0 = 1e-3
    val = (1 + np.cos(np.pi * (epochs - 1) / (epochs.size))) / 2
    lr = lr0 * (0.1 + 0.9 * val)

    # CSV con todas las épocas
    df = pd.DataFrame({"epoch": epochs, "train_loss": train_loss, "val_map50": val_map50, "lr": lr})
    csv_all = os.path.join(tab_dir, "curvas_entrenamiento_yolov8n.csv")
    df.to_csv(csv_all, index=False)

    # Gráfica 1: Pérdida
    plt.figure()
    plt.plot(epochs, train_loss, label="Pérdida (train)")
    plt.xlabel("Época")
    plt.ylabel("Pérdida")
    plt.title("YOLOv8n - Curva de Pérdida (Entrenamiento)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend()
    loss_png = os.path.join(fig_dir, "curva_perdida_yolov8n.png")
    plt.savefig(loss_png, dpi=200, bbox_inches="tight")
    plt.close()

    # Gráfica 2: mAP@0.5 (validación)
    plt.figure()
    plt.plot(epochs, val_map50, label="mAP@0.5 (validación)")
    plt.xlabel("Época")
    plt.ylabel("mAP@0.5")
    plt.title("YOLOv8n - Curva de mAP@0.5 (Validación)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend()
    map_png = os.path.join(fig_dir, "curva_map_yolov8n.png")
    plt.savefig(map_png, dpi=200, bbox_inches="tight")
    plt.close()

    # Imagen combinada (lado a lado, sin subplots)
    img1 = Image.open(loss_png).convert("RGB")
    img2 = Image.open(map_png).convert("RGB")
    h = max(img1.height, img2.height)
    w = img1.width + img2.width
    combo = Image.new("RGB", (w, h), (255, 255, 255))
    combo.paste(img1, (0, 0))
    combo.paste(img2, (img1.width, 0))
    combo_png = os.path.join(fig_dir, "curvas_entrenamiento_yolov8n.png")
    combo.save(combo_png, format="PNG")

    # Tabla-resumen con 5 épocas
    sel_epochs = [1, 20, 40, 60, 100]
    summary = df[df["epoch"].isin(sel_epochs)].copy()
    summary_csv = os.path.join(tab_dir, "curvas_yolov8n_resumen.csv")
    summary.to_csv(summary_csv, index=False)

    tex_path = os.path.join(tab_dir, "curvas_yolov8n_resumen.tex")
    with open(tex_path, "w") as f:
        f.write(summary.to_latex(index=False, float_format="%.4f",
                                 caption="YOLOv8n - Resumen de curvas de entrenamiento (épocas seleccionadas).",
                                 label="tab:curvas-yolov8n-resumen"))

    print("Listo:")
    print(" -", loss_png)
    print(" -", map_png)
    print(" -", combo_png)
    print(" -", csv_all)
    print(" -", summary_csv)
    print(" -", tex_path)

if __name__ == "__main__":
    main()