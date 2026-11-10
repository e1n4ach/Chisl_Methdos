"""Постановка варианта 4. Точное решение вызывается только при проверке."""

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def initial_displacement(x: Array) -> Array:
    return np.exp(2.0 * x)


def initial_velocity(x: Array) -> Array:
    return np.zeros_like(x)


def initial_acceleration(x: Array) -> Array:
    # Из PDE: phi'' - 5*phi = (4 - 5)*exp(2x). Не из точного U(x,t).
    return -initial_displacement(x)


def exact_solution(x: Array, time: float | Array) -> Array:
    return np.exp(2.0 * x) * np.cos(time)
