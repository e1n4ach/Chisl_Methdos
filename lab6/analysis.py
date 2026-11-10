"""Ошибки, сравнение методов и независимое сгущение h и tau."""

from dataclasses import replace
from itertools import product
from typing import Callable

import numpy as np
import pandas as pd

from boundaries import BOUNDARY_NAMES, boundary_residuals, spatial_operator
from problem import Array, exact_solution
from solver import Config, SCHEME_NAMES, START_NAMES, Solution, solve

COMBINATIONS = tuple(product(SCHEME_NAMES, START_NAMES, BOUNDARY_NAMES))
Progress = Callable[[float, str], None]


def method_name(scheme: str, initial_order: int, boundary: str) -> str:
    return f"{SCHEME_NAMES[scheme]} · старт {initial_order} · {BOUNDARY_NAMES[boundary]}"


def max_errors(solution: Solution) -> Array:
    exact = exact_solution(solution.x[None, :], solution.time[:, None])
    return np.max(np.abs(solution.values - exact), axis=1)


def boundary_error(solution: Solution) -> float:
    """ГУ на k>=1: трёхслойное ускорение там, где есть временной шаблон.

    У неявного метода на k=1 и у явного на последнем слое ускорение L_h*u
    проверяет пространственную Robin-формулу. Сам Taylor-старт проверяется отдельно.
    """
    config, values = solution.config, solution.values
    residuals = []
    operator = spatial_operator(config.n, "pde") if config.boundary == "pde" else None
    for layer in range(1, config.steps + 1):
        acceleration = None
        if config.boundary == "pde":
            if (config.scheme == "implicit" and layer == 1
                    or config.scheme == "explicit" and layer == config.steps):
                acceleration = operator.apply(values[layer])
            else:
                center = layer - 1 if config.scheme == "implicit" else layer
                acceleration = (values[center+1]-2*values[center]+values[center-1])/config.tau**2
        residuals.append(boundary_residuals(values[layer], config.h, config.boundary, acceleration))
    return float(np.max(np.abs(residuals)))


def compare(config: Config, combinations: tuple = COMBINATIONS,
            progress: Progress | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, histories = [], {}
    for index, (scheme, order, boundary) in enumerate(combinations):
        name = method_name(scheme, order, boundary)
        try:
            result = solve(replace(config, scheme=scheme, initial_order=order, boundary=boundary))
            errors = max_errors(result)
            histories[name] = errors
            rows.append({"Метод": name, "ε(T)": errors[-1], "max ε(t)": errors.max(),
                         "Невязка ГУ": boundary_error(result), "Статус": "OK"})
        except (ValueError, FloatingPointError) as exc:
            rows.append({"Метод": name, "Статус": str(exc)})
        if progress:
            progress((index+1)/len(combinations), name)
    history = pd.DataFrame(histories, index=np.linspace(0, config.final_time, config.steps+1))
    history.index.name = "t"
    return pd.DataFrame(rows), history


def convergence(config: Config, axis: str, grids: tuple[int, ...], fixed: int,
                progress: Progress | None = None) -> pd.DataFrame:
    """h: grids=N, fixed=M; tau: grids=M, fixed=N. Ничего не меняется совместно."""
    if axis not in ("h", "tau") or len(grids) < 2:
        raise ValueError("Нужны axis=h/tau и хотя бы две сетки.")
    rows, terminal_values = [], []
    for index, count in enumerate(grids):
        current = replace(config, n=count, steps=fixed) if axis == "h" else replace(config, n=fixed, steps=count)
        result = solve(current)
        errors = max_errors(result)
        terminal_values.append(result.values[-1].copy())
        rows.append({"N": current.n, "M": current.steps, "h": current.h, "τ": current.tau,
                     "ε(T)": float(errors[-1]), "max ε(t)": float(errors.max()),
                     "S": result.stability.parameter})
        if progress:
            progress((index+1)/(len(grids)+1), f"N={current.n}, M={current.steps}")
    table = pd.DataFrame(rows)
    step_column = "h" if axis == "h" else "τ"
    table["p(T)"] = np.log(table["ε(T)"].shift()/table["ε(T)"]) / np.log(table[step_column].shift()/table[step_column])
    table["p(max)"] = np.log(table["max ε(t)"].shift()/table["max ε(t)"]) / np.log(table[step_column].shift()/table[step_column])
    # Контроль: ещё вдвое уменьшаем фиксированный шаг на самой мелкой сетке.
    last = replace(config, n=grids[-1], steps=2*fixed) if axis == "h" else replace(config, n=2*fixed, steps=grids[-1])
    control_error = float(max_errors(solve(last))[-1])
    table.attrs["control_error"] = control_error
    table.attrs["fixed_step_effect"] = abs(control_error-table.iloc[-1]["ε(T)"])/max(table.iloc[-1]["ε(T)"], 1e-300)
    if axis == "tau":
        # Один и тот же N: пространственная ошибка вычитается при сравнении решений.
        # Два эталона проверяют, что временной эталон достаточно точен.
        reference_config = replace(config, n=fixed, steps=16*grids[-1])
        reference = solve(reference_config, terminal_only=True).values[-1]
        finer = solve(replace(reference_config, steps=32*grids[-1]), terminal_only=True).values[-1]
        distances = []
        for values in terminal_values:
            distances.append(float(np.max(np.abs(values-finer))))
        table["Разность с временным эталоном"] = distances
        table["p(время)"] = np.log(table["Разность с временным эталоном"].shift()/table["Разность с временным эталоном"]) / np.log(table["τ"].shift()/table["τ"])
        table.attrs["reference_difference"] = float(np.max(np.abs(reference-finer)))
        table.attrs["reference_steps"] = 32*grids[-1]
    if progress:
        progress(1.0, "Контроль фиксированного шага выполнен")
    return table
