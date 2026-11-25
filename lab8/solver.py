"""Два расщепления и расчёт временных слоёв."""

from dataclasses import dataclass
from math import ceil, isfinite, nextafter

import numpy as np

from lab8.problem import Problem
from lab8.tridiagonal import factor_tridiagonal, solve_tridiagonal

METHOD_NAMES = {"adi": "Переменные направления (ADI)", "fractional": "Дробные шаги"}
MAX_VALUES = 10_000_000


@dataclass(frozen=True)
class Config:
    nx: int = 20
    ny: int = 16
    steps: int = 120
    final_time: float = 1.0
    method: str = "adi"

    def __post_init__(self):
        for value, minimum, label in ((self.nx, 2, "Nx"), (self.ny, 2, "Ny"), (self.steps, 1, "M")):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError(f"{label} должно быть целым числом ≥ {minimum}.")
        if not isfinite(self.final_time) or self.final_time <= 0:
            raise ValueError("Требуется конечное T > 0.")
        if self.method not in METHOD_NAMES:
            raise ValueError("Неизвестный метод.")
        if self.tau <= 0:
            raise ValueError("Шаг τ слишком мал для представления в float64.")

    @property
    def hx(self):
        return 1 / self.nx

    @property
    def hy(self):
        return 1 / self.ny

    @property
    def tau(self):
        return self.final_time / self.steps


def steps_for_tau(final_time, requested_tau):
    if not isfinite(final_time) or final_time <= 0 or not isfinite(requested_tau) or requested_tau <= 0:
        raise ValueError("Требуются конечные T > 0 и τ > 0.")
    ratio = final_time / requested_tau
    if not isfinite(ratio):
        raise ValueError("Слишком большое отношение T/τ.")
    # При кратных десятичных T и τ отношение может округлиться на 1 ULP вверх.
    return max(1, ceil(nextafter(ratio, 0.0)))


@dataclass
class Solution:
    config: Config
    problem: Problem
    time: np.ndarray
    values: np.ndarray


def diffusion_factors(intervals, ratio):
    side = np.full(intervals-2, -ratio)
    diag = np.full(intervals-1, 1 + 2*ratio)
    return factor_tridiagonal(side, diag, side)


class Stepper:
    def __init__(self, config, problem):
        self.config = config
        self.problem = problem
        implicit_time = config.tau/2 if config.method == "adi" else config.tau
        self.rx = implicit_time / config.hx**2
        self.ry = implicit_time / config.hy**2
        self.x_factors = diffusion_factors(config.nx, self.rx)
        self.y_factors = diffusion_factors(config.ny, self.ry)

    def advance(self, old, t0, t1):
        """Возвращает промежуточный проход по x и полный слой после прохода по y."""
        tau = self.config.tau
        g0, g1 = self.problem.boundary_scale(t0), self.problem.boundary_scale(t1)
        center = old[1:-1, 1:-1]
        stage = np.empty_like(old)
        new = np.empty_like(old)

        if self.config.method == "adi":
            forcing = (tau/2) * self.problem.source(t0 + (t1-t0)/2)
            # На x-краях Dyy g = 0. Складываем две стадии PR и получаем среднее g.
            stage_scale = (g0 + g1) / 2
            rhs_x = center + self.ry*(old[:-2, 1:-1] - 2*center + old[2:, 1:-1]) + forcing
        else:
            forcing = (tau/2) * self.problem.source(t1)
            # Из второго прохода на x-краях w = (I - tau Dyy)g(t1) - forcing.
            stage_scale = g1 + (tau/2)*np.sin(t1)
            rhs_x = center + forcing
        self.problem.set_boundary(stage, stage_scale)
        rhs_x[:, 0] += self.rx * stage[1:-1, 0]
        rhs_x[:, -1] += self.rx * stage[1:-1, -1]
        stage[1:-1, 1:-1] = solve_tridiagonal(self.x_factors, rhs_x.T).T

        self.problem.set_boundary(new, g1)
        center = stage[1:-1, 1:-1]
        rhs_y = center + forcing
        if self.config.method == "adi":
            rhs_y += self.rx*(stage[1:-1, :-2] - 2*center + stage[1:-1, 2:])
        rhs_y[0] += self.ry * new[0, 1:-1]
        rhs_y[-1] += self.ry * new[-1, 1:-1]
        new[1:-1, 1:-1] = solve_tridiagonal(self.y_factors, rhs_y)
        return stage, new


def solve(config: Config, *, terminal_only=False, manufactured=False) -> Solution:
    shape = (config.ny + 1, config.nx + 1)
    layers = 1 if terminal_only else config.steps + 1
    if layers * shape[0] * shape[1] > MAX_VALUES:
        raise ValueError(f"До {MAX_VALUES:,} значений на расчёт. Уменьшите Nx, Ny или M.")
    time = np.linspace(0, config.final_time, config.steps + 1) if not terminal_only else None
    problem = Problem(np.linspace(0, 1, shape[1]), np.linspace(0, 1, shape[0]), manufactured)
    field = problem.initial()
    values = np.empty((layers, *shape))
    values[0] = field
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        stepper = Stepper(config, problem)
        for k in range(config.steps):
            t0 = k*config.tau
            t1 = config.final_time if k+1 == config.steps else (k+1)*config.tau
            stage, field = stepper.advance(field, t0, t1)
            if not (np.isfinite(stage).all() and np.isfinite(field).all()):
                raise FloatingPointError(f"Неконечные значения на переходе k={k} → {k+1}.")
            values[0 if terminal_only else k+1] = field
    if terminal_only:
        time = np.array([config.final_time])
    return Solution(config, problem, time, values)
