"""Вариант 7 и контрольная задача для оценки пространственной точности."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Problem:
    x: np.ndarray
    y: np.ndarray
    manufactured: bool = False

    def __post_init__(self):
        self.xy = self.y[:, None] * self.x[None, :]
        self.mode = None
        if self.manufactured:
            self.mode = np.sin(np.pi*self.y[:, None])*np.sin(np.pi*self.x[None, :])
            self.mode[[0, -1], :] = 0
            self.mode[:, [0, -1]] = 0

    def initial(self):
        return self.xy.copy() if self.mode is None else self.xy + self.mode

    def exact(self, t, index=np.s_[:, :]):
        values = np.cos(t) * self.xy[index]
        if self.mode is not None:
            values += np.exp(-t) * self.mode[index]
        return values

    def source(self, t):
        interior = np.s_[1:-1, 1:-1]
        values = -np.sin(t) * self.xy[interior]
        if self.mode is not None:
            values += (2*np.pi**2 - 1)*np.exp(-t)*self.mode[interior]
        return values

    @staticmethod
    def boundary_scale(t):
        return np.cos(t)

    def set_boundary(self, field, scale):
        field[0, :] = 0
        field[:, 0] = 0
        field[-1, :] = self.x * scale
        field[:, -1] = self.y * scale
