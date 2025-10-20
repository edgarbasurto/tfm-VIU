from __future__ import annotations
import time
from typing import List
import numpy as np


def _sync(device: str):
    try:
        import torch
        if device == 'cuda' and torch.cuda.is_available():
            torch.cuda.synchronize()
        elif device == 'mps' and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            torch.mps.synchronize()
    except Exception:
        pass


def measure_fps(model, images: List[np.ndarray], warmup: int = 10, iters: int = 100, device: str = 'cpu') -> float:
    for i in range(min(warmup, len(images))):
        _ = model.predict([images[i % len(images)]])
    _sync(device)
    t0 = time.perf_counter()
    for i in range(iters):
        _ = model.predict([images[i % len(images)]])
    _sync(device)
    dt = time.perf_counter() - t0
    return iters / dt if dt > 0 else 0.0
