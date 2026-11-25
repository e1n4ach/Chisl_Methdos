"""Якоби, лексикографический Зейдель и SOR для разностной системы."""

from dataclasses import dataclass
from math import isfinite

import numpy as np
from scipy.sparse import diags, tril
from scipy.sparse.linalg import splu

from lab7.boundaries import BOUNDARY_NAMES, assemble
from lab7.problem import LX, LY, exact_solution

METHOD_NAMES = {"jacobi": "Либман (Якоби)", "seidel": "Зейдель", "sor": "SOR"}
STATUS_NAMES = {
    "converged": "Сошёлся",
    "limit": "Достигнут max_iter",
    "nonfinite": "Переполнение или NaN",
}


@dataclass(frozen=True)
class Config:
    nx: int = 24
    ny: int = 32
    method: str = "sor"
    boundary: str = "three"
    epsilon: float = 1e-8
    omega: float = 1.8
    max_iter: int = 50_000

    def __post_init__(self):
        for value, minimum, name in ((self.nx, 2, "Nx"), (self.ny, 3, "Ny"),
                                     (self.max_iter, 1, "max_iter")):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError(f"{name} должно быть целым числом ≥ {minimum}.")
        if self.method not in METHOD_NAMES or self.boundary not in BOUNDARY_NAMES:
            raise ValueError("Неизвестный метод или аппроксимация границы.")
        if not isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("Требуется конечное ε > 0.")
        if not 1 <= self.omega < 2:
            raise ValueError("Для верхней релаксации требуется 1 ≤ ω < 2.")

    @property
    def hx(self):
        return LX / self.nx

    @property
    def hy(self):
        return LY / self.ny

    @property
    def sweep_weight(self):
        return self.omega if self.method == "sor" else 1.0


@dataclass
class Solution:
    config: Config
    x: np.ndarray
    y: np.ndarray
    values: np.ndarray
    exact: np.ndarray
    residual_history: np.ndarray
    error_history: np.ndarray
    status: str

    @property
    def iterations(self):
        return len(self.residual_history) - 1

    @property
    def absolute_error(self):
        return np.abs(self.values - self.exact)

    @property
    def iteration_error_bound(self):
        """Оценка отклонения от точного решения разностной системы."""
        factor = 5 / 3 if self.config.boundary == "three" else 1
        return factor * LX**2 / 8 * self.residual_history[-1]


def solve(config: Config) -> Solution:
    grid = assemble(config.nx, config.ny, config.boundary)
    diagonal = grid.matrix.diagonal()
    weight = config.sweep_weight
    if config.method != "jacobi":
        # Матрица треугольная. Повторно используем её факторизацию для прямого хода.
        lower = diags(diagonal, format="csc") + weight * tril(grid.matrix, k=-1, format="csc")
        forward = splu(lower, permc_spec="NATURAL", diag_pivot_thresh=0).solve

    field = grid.template.copy()
    exact = exact_solution(grid.x[None, :], grid.y[:, None])
    vector = np.zeros_like(grid.rhs)
    grid.expand(vector, field)
    residual = grid.rhs.copy()
    residual_history = [np.max(np.abs(residual))]
    error_history = [np.max(np.abs(field - exact))]
    status = "limit"

    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(config.max_iter):
            if residual_history[-1] <= config.epsilon:
                break
            correction = residual / diagonal if config.method == "jacobi" else forward(weight * residual)
            vector += correction
            grid.expand(vector, field)
            residual = grid.rhs - grid.matrix @ vector
            residual_history.append(np.max(np.abs(residual)))
            error_history.append(np.max(np.abs(field - exact)))
            if not (isfinite(residual_history[-1]) and isfinite(error_history[-1])):
                status = "nonfinite"
                break
    if status != "nonfinite" and residual_history[-1] <= config.epsilon:
        status = "converged"
    return Solution(config, grid.x, grid.y, field, exact, np.array(residual_history),
                    np.array(error_history), status)
