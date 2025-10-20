from __future__ import annotations
import os, json, random
from glob import glob
from typing import List, Dict, Any, Optional

def make_subset(images_dir: str, output_json: str, max_images: int = 1000, seed: int = 42) -> None:
    """Genera una lista de paths de imágenes como subconjunto reproducible."""
    random.seed(seed)
    imgs = sorted(glob(os.path.join(images_dir, "*.jpg")))
    if max_images and len(imgs) > max_images:
        imgs = random.sample(imgs, max_images)
    with open(output_json, "w") as f:
        json.dump({"images": imgs}, f, indent=2)

def load_subset(subset_json: Optional[str], images_dir: str) -> List[str]:
    if subset_json and os.path.exists(subset_json):
        with open(subset_json) as f:
            return json.load(f).get("images", [])
    return sorted(glob(os.path.join(images_dir, "*.jpg")))
