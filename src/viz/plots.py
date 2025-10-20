from __future__ import annotations
from typing import List, Dict
import matplotlib.pyplot as plt


def pareto_map_fps(points: List[Dict], outfile: str):
    """Dibuja mAP vs FPS con anotaciones de nombre de modelo."""
    xs = [p['FPS'] for p in points]
    ys = [p['mAP'] for p in points]
    names = [p['name'] for p in points]

    plt.figure()
    plt.scatter(xs, ys, s=80)
    for x, y, n in zip(xs, ys, names):
        plt.annotate(n, (x, y), xytext=(5, 5), textcoords="offset points")
    plt.xlabel("FPS")
    plt.ylabel("mAP@[.5:.95]")
    plt.title("Frontera Pareto: Precisión vs Velocidad")
    plt.tight_layout()
    plt.savefig(outfile)


def ap_bar_per_class(ap_dict: Dict[str, float], outfile: str):
    cls = list(ap_dict.keys())
    vals = [ap_dict[c] for c in cls]

    plt.figure()
    plt.bar(cls, vals)
    plt.xticks(rotation=90)
    plt.ylabel("AP")
    plt.title("AP por clase")
    plt.tight_layout()
    plt.savefig(outfile)


def pr_curve(recall, precision, outfile: str, title: str = "Curva Precisión-Recall"):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(recall, precision, linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(outfile)
