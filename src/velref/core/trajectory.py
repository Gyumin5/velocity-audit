from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Pose2D:
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray

    def __post_init__(self):
        self.t = np.asarray(self.t, dtype=np.float64)
        self.x = np.asarray(self.x, dtype=np.float64)
        self.y = np.asarray(self.y, dtype=np.float64)
        if not (self.t.shape == self.x.shape == self.y.shape):
            raise ValueError("t, x, y must have the same shape")
        if self.t.ndim != 1:
            raise ValueError("expected 1-D arrays")
        if np.any(np.diff(self.t) <= 0):
            raise ValueError("t must be strictly increasing")

    def __len__(self) -> int:
        return int(self.t.size)
