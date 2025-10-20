from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class Detector(ABC):
    """Interfaz común para detectores de objetos.
    Cada `predict` devuelve, por imagen, una lista de dicts con:
      - bbox: [x1, y1, x2, y2]
      - score: float
      - cls: int
      - cls_name: str (opcional si se dispone)
    """

    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def predict(self, images: List[Any]) -> List[List[Dict[str, Any]]]: ...

    def to(self, device: str):
        """Mover a dispositivo (si aplica)."""
        return self
