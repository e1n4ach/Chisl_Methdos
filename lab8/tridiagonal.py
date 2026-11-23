"""Прогонка с повторным использованием множителей для пакета строк."""

import numpy as np


def factor_tridiagonal(lower, diagonal, upper):
    pivots = np.asarray(diagonal, dtype=float).copy()
    multipliers = np.asarray(lower, dtype=float).copy()
    upper = np.asarray(upper, dtype=float)
    if pivots.ndim != 1 or pivots.size == 0 or multipliers.shape != upper.shape or upper.shape != (pivots.size-1,):
        raise ValueError("Неверные размеры диагоналей.")
    for i in range(1, len(pivots)):
        if pivots[i-1] == 0:
            raise ValueError("Нулевой ведущий элемент прогонки.")
        multipliers[i-1] /= pivots[i-1]
        pivots[i] -= multipliers[i-1] * upper[i-1]
    if not np.isfinite(pivots).all() or pivots[-1] == 0:
        raise ValueError("Вырожденная матрица прогонки.")
    return multipliers, pivots, upper


def solve_tridiagonal(factors, rhs):
    """Решает A x = rhs вдоль первой оси, по столбцу на правую часть."""
    multipliers, pivots, upper = factors
    result = np.array(rhs, dtype=float, copy=True, order="C")
    if result.shape[0] != len(pivots):
        raise ValueError("Размер правой части не согласован с матрицей.")
    for i in range(1, len(pivots)):
        result[i] -= multipliers[i-1] * result[i-1]
    result[-1] /= pivots[-1]
    for i in range(len(pivots)-2, -1, -1):
        result[i] = (result[i] - upper[i]*result[i+1]) / pivots[i]
    return result
