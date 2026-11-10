"""Общий временной цикл для 12 комбинаций, подготовка матрицы и устойчивость."""

from dataclasses import dataclass
from functools import lru_cache
from math import ceil
from typing import Literal

import numpy as np
from scipy.linalg import eigh_tridiagonal, lapack

from boundaries import BOUNDARY_NAMES, SpatialOperator, restore_edges, spatial_operator
from problem import Array, initial_displacement, initial_velocity

Scheme = Literal["explicit", "implicit"]
Boundary = Literal["first", "three", "pde"]
SCHEME_NAMES = {"explicit": "Явная схема «крест»", "implicit": "Неявная схема"}
START_NAMES = {1: "1-й порядок", 2: "2-й порядок (оператор Lₕ)"}


@dataclass(frozen=True)
class Config:
    n: int = 40
    final_time: float = 1.0
    steps: int = 200
    scheme: Scheme = "explicit"
    initial_order: int = 2
    boundary: Boundary = "pde"

    @property
    def h(self) -> float:
        return 1.0 / self.n

    @property
    def tau(self) -> float:
        return self.final_time / self.steps

    def validate(self) -> None:
        if not isinstance(self.n, int) or self.n < 3:
            raise ValueError("Число интервалов N должно быть целым и не меньше 3.")
        if not isinstance(self.steps, int) or self.steps < 2:
            raise ValueError("Число шагов M должно быть целым и не меньше 2.")
        if not np.isfinite(self.final_time) or self.final_time <= 0:
            raise ValueError("Нужно конечное T > 0.")
        if self.scheme not in SCHEME_NAMES or self.boundary not in BOUNDARY_NAMES or self.initial_order not in START_NAMES:
            raise ValueError("Неизвестная комбинация методов.")
        if (self.steps + 1)*(self.n + 1) > 12_000_000:
            raise ValueError("Сетка превышает 12 млн значений. Уменьшите N или M.")


@dataclass(frozen=True)
class Stability:
    lambda_min: float
    lambda_max: float
    tau_limit: float
    parameter: float  # tau² * max(-lambda) / 4
    stable: bool


@lru_cache(maxsize=64)
def operator_spectrum(n: int, boundary: str) -> tuple[float, float]:
    operator = spatial_operator(n, boundary)
    # L подобна симметричной матрице: S_ii=L_ii, S_i,i+1=sqrt(L_i,i+1*L_i+1,i).
    products = operator.lower * operator.upper
    if np.any(products <= 0):
        raise ValueError("Не удалось симметризовать пространственный оператор.")
    eigenvalues = eigh_tridiagonal(operator.diagonal, np.sqrt(products), eigvals_only=True)
    return float(eigenvalues[0]), float(eigenvalues[-1])


def stability(config: Config) -> Stability:
    config.validate()
    smallest, largest = operator_spectrum(config.n, config.boundary)
    limit = 2 / np.sqrt(-smallest) if largest < 0 else 0.0
    parameter = config.tau**2 * (-smallest) / 4
    # Строгое неравенство: на границе 4 характеристический корень -1 кратный.
    return Stability(smallest, largest, float(limit), float(parameter),
                     bool(largest < 0 and parameter < 1 - 1e-12))


def steps_for_tau(final_time: float, requested_tau: float) -> int:
    if not np.isfinite(requested_tau) or requested_tau <= 0:
        raise ValueError("Запрошенный шаг времени должен быть конечным и положительным.")
    return max(2, ceil(final_time / requested_tau))


class TridiagonalSolver:
    """LU трёхдиагональной I-tau²*L с выбором главного элемента, один раз на расчёт."""

    def __init__(self, operator: SpatialOperator, tau: float):
        lower = -tau**2 * operator.lower
        diagonal = 1 - tau**2 * operator.diagonal
        upper = -tau**2 * operator.upper
        *self.factors, info = lapack.dgttrf(lower, diagonal, upper)
        if info != 0:
            raise ValueError(f"Не удалось разложить матрицу, LAPACK info={info}.")

    def solve(self, rhs: Array) -> Array:
        solution, info = lapack.dgttrs(*self.factors, rhs[:, None])
        if info != 0:
            raise ValueError(f"Ошибка решения системы, LAPACK info={info}.")
        return solution[:, 0]


@dataclass
class Solution:
    config: Config
    x: Array
    time: Array
    values: Array
    stability: Stability


def first_layer(initial: Array, velocity: Array, operator: SpatialOperator,
                config: Config) -> Array:
    """Taylor-старт для той же системы v_tt=L_h*v, что интегрируется далее."""
    active = slice(None) if config.boundary == "pde" else slice(1, -1)
    values = initial + config.tau * velocity
    if config.initial_order == 2:
        # Ускорение в t=0 вычисляется уже после пространственной дискретизации.
        # Для PDE-границы L_h включает обе крайние строки — отдельного замыкания нет.
        values[active] += config.tau**2 / 2 * operator.apply(initial[active])
    if config.boundary != "pde":
        restore_edges(values, config.h, config.boundary)
    return values


def solve(config: Config = Config(), *, terminal_only: bool = False) -> Solution:
    config.validate()
    diagnostic = stability(config)
    if config.scheme == "explicit" and not diagnostic.stable:
        raise ValueError(f"Неустойчивая явная сетка: S={diagnostic.parameter:.6g}, "
                         f"lambda_max={diagnostic.lambda_max:.6g}. "
                         f"Требуются lambda_max<0 и S<1; tau<{diagnostic.tau_limit:.6g}.")
    if config.scheme == "implicit" and diagnostic.lambda_max >= 0:
        raise ValueError("У пространственного оператора есть растущая мода. Увеличьте N.")
    x = np.linspace(0, 1, config.n + 1)
    time = np.array([0.0, config.final_time]) if terminal_only else np.linspace(0, config.final_time, config.steps + 1)
    values = np.empty((2 if terminal_only else config.steps + 1, config.n + 1))
    initial = initial_displacement(x)
    velocity = initial_velocity(x)
    values[0] = initial  # Начальное условие имеет приоритет при t=0.
    operator = spatial_operator(config.n, config.boundary)
    values[1] = first_layer(initial, velocity, operator, config)
    active = slice(None) if config.boundary == "pde" else slice(1, -1)
    linear_solver = TridiagonalSolver(operator, config.tau) if config.scheme == "implicit" else None
    previous, current_layer = values[0].copy(), values[1].copy()
    with np.errstate(over="raise", invalid="raise"):
        for layer in range(1, config.steps):
            current = current_layer[active]
            rhs = 2*current - previous[active]
            if linear_solver is None:
                next_values = rhs + config.tau**2 * operator.apply(current)
            else:
                next_values = linear_solver.solve(rhs)
            next_layer = np.empty_like(current_layer)
            next_layer[active] = next_values
            if config.boundary != "pde":
                restore_edges(next_layer, config.h, config.boundary)
            if not terminal_only:
                values[layer+1] = next_layer
            previous, current_layer = current_layer, next_layer
    if terminal_only:
        values[-1] = current_layer
    if not np.isfinite(values).all():
        raise FloatingPointError("Получены бесконечные значения или NaN.")
    return Solution(config, x, time, values, diagnostic)
