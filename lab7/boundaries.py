"""Две аппроксимации производной по y и сборка разностной системы."""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, diags, eye, kron

from lab7.problem import (
    LX, LY, bottom_derivative, left_value, right_value, top_derivative,
)

BOUNDARY_NAMES = {
    "three": "Трёхточечная односторонняя",
    "ghost": "Центральная с фиктивными узлами",
}


@dataclass
class Discretization:
    x: np.ndarray
    y: np.ndarray
    active_rows: slice
    matrix: csr_matrix
    rhs: np.ndarray
    template: np.ndarray
    bottom_shift: np.ndarray
    top_shift: np.ndarray

    def expand(self, interior: np.ndarray, field: np.ndarray) -> None:
        """Записывает неизвестные и восстанавливает исключённые края на месте."""
        field[self.active_rows, 1:-1] = interior.reshape(-1, len(self.x) - 2)
        if self.active_rows.start == 1:
            column = slice(1, -1)
            field[0, column] = (4 * field[1, column] - field[2, column]) / 3 + self.bottom_shift
            field[-1, column] = (4 * field[-2, column] - field[-3, column]) / 3 + self.top_shift

def assemble(nx: int, ny: int, boundary: str) -> Discretization:
    x, y = np.linspace(0, LX, nx + 1), np.linspace(0, LY, ny + 1)
    hx, hy = LX / nx, LY / ny
    a, b = hx**-2, hy**-2
    nx_inner = nx - 1
    three = boundary == "three"
    rows = slice(1, -1) if three else slice(None)
    nrows = len(y[rows])

    x_matrix = diags([-a, 2*a, -a], [-1, 0, 1], shape=(nx_inner, nx_inner))
    y_diag = np.full(nrows, 2*b)
    y_lower = np.full(nrows - 1, -b)
    y_upper = y_lower.copy()
    if three:
        y_diag[[0, -1]] -= 4*b/3
        y_upper[0] += b/3
        y_lower[-1] += b/3
    else:
        y_upper[0] = y_lower[-1] = -2*b
    y_matrix = diags([y_lower, y_diag, y_upper], [-1, 0, 1])
    matrix = kron(eye(nrows), x_matrix, format="csr") + kron(
        y_matrix, eye(nx_inner), format="csr"
    )

    field = np.zeros((ny + 1, nx + 1))
    field[:, 0], field[:, -1] = left_value(y), right_value(y)
    rhs = np.zeros((nrows, nx_inner))
    rhs[:, 0] += a * field[rows, 0]
    rhs[:, -1] += a * field[rows, -1]
    divisor = 3 if three else 1
    bottom_shift = -2 * hy / divisor * bottom_derivative(x[1:-1])
    top_shift = 2 * hy / divisor * top_derivative(x[1:-1])
    rhs[0] += b * bottom_shift
    rhs[-1] += b * top_shift
    return Discretization(x, y, rows, matrix, rhs.ravel(), field, bottom_shift, top_shift)


def boundary_residuals(values: np.ndarray, x: np.ndarray, y: np.ndarray,
                       boundary: str) -> dict[str, float]:
    """Невязка каждого ГУ. В углах задано значение Дирихле."""
    hx, hy = x[1] - x[0], y[1] - y[0]
    inner = values[:, 1:-1]
    if boundary == "three":
        bottom = (-3*inner[0] + 4*inner[1] - inner[2]) / (2*hy)
        top = (3*inner[-1] - 4*inner[-2] + inner[-3]) / (2*hy)
    else:
        # Восстанавливаем фиктивные узлы из уравнения, затем проверяем производную.
        dxx = (values[[0, -1], :-2] - 2*inner[[0, -1]] + values[[0, -1], 2:]) / hx**2
        bottom = (inner[1] - inner[0]) / hy + hy/2 * dxx[0]
        top = (inner[-1] - inner[-2]) / hy - hy/2 * dxx[-1]
    return {
        "x=0": float(np.max(np.abs(values[:, 0] - left_value(y)))),
        "x=1": float(np.max(np.abs(values[:, -1] - right_value(y)))),
        "y=0": float(np.max(np.abs(bottom - bottom_derivative(x[1:-1])))),
        "y=π/2": float(np.max(np.abs(top - top_derivative(x[1:-1])))),
    }
