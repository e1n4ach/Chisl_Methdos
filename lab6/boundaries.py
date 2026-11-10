"""Три варианта Robin-границ; производная везде по положительному x."""

from dataclasses import dataclass

import numpy as np

from problem import Array

BOUNDARY_NAMES = {
    "first": "2 точки, 1-й порядок",
    "three": "3 точки, 2-й порядок",
    "pde": "2 точки, 2-й порядок с уравнением",
}


@dataclass(frozen=True)
class SpatialOperator:
    lower: Array
    diagonal: Array
    upper: Array

    def apply(self, values: Array) -> Array:
        result = self.diagonal * values
        result[1:] += self.lower * values[:-1]
        result[:-1] += self.upper * values[1:]
        return result


def edge_coefficients(h: float, boundary: str) -> tuple[float, float, float, float]:
    """u0=a*u1+b*u2; uN=c*u(N-1)+d*u(N-2)."""
    if boundary == "first":
        return 1 / (1 + 2*h), 0.0, 1 / (1 - 2*h), 0.0
    if boundary == "three":
        return 4 / (3 + 4*h), -1 / (3 + 4*h), 4 / (3 - 4*h), -1 / (3 - 4*h)
    raise ValueError("Эти коэффициенты определены для first и three.")


def restore_edges(values: Array, h: float, boundary: str) -> None:
    left_near, left_far, right_near, right_far = edge_coefficients(h, boundary)
    values[0] = left_near*values[1] + left_far*values[2]
    values[-1] = right_near*values[-2] + right_far*values[-3]


def spatial_operator(n: int, boundary: str) -> SpatialOperator:
    """L для v_tt=L*v. В PDE-варианте v включает границы, иначе только внутренние узлы."""
    h = 1.0 / n
    inverse_h2 = 1.0 / h**2
    size = n + 1 if boundary == "pde" else n - 1
    lower = np.full(size - 1, inverse_h2)
    diagonal = np.full(size, -2*inverse_h2 - 5)
    upper = np.full(size - 1, inverse_h2)
    if boundary == "pde":
        # Из двухточечного Taylor-ГУ: A0=2(u1-u0)/h²-4u0/h-5u0;
        # AN=2(u(N-1)-uN)/h²+4uN/h-5uN. Знаки 4/h различаются.
        diagonal[0] -= 4/h
        diagonal[-1] += 4/h
        upper[0] = lower[-1] = 2*inverse_h2
    else:
        a, b, c, d = edge_coefficients(h, boundary)
        # Подстановка u0=a*u1+b*u2 в строку i=1 сохраняет три диагонали.
        diagonal[0] += inverse_h2*a
        upper[0] += inverse_h2*b
        diagonal[-1] += inverse_h2*c
        lower[-1] += inverse_h2*d
    return SpatialOperator(lower, diagonal, upper)


def boundary_residuals(values: Array, h: float, boundary: str,
                       acceleration: Array | None = None) -> Array:
    """Невязки исходных разностных ГУ (единицы производной), не строк L."""
    if boundary == "three":
        return np.array([(-3*values[0]+4*values[1]-values[2])/(2*h)-2*values[0],
                         (3*values[-1]-4*values[-2]+values[-3])/(2*h)-2*values[-1]])
    residual = np.array([(values[1]-values[0])/h-2*values[0],
                         (values[-1]-values[-2])/h-2*values[-1]])
    if boundary == "pde":
        if acceleration is None:
            raise ValueError("Для PDE-границы нужно ускорение на соответствующем слое.")
        residual += np.array([-h/2*(acceleration[0]+5*values[0]),
                              h/2*(acceleration[-1]+5*values[-1])])
    return residual
