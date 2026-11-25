"""Таблицы, сравнение методов и измерение порядка при сгущении сетки."""

from dataclasses import replace

import numpy as np
import pandas as pd
from scipy.sparse.linalg import spsolve

from lab7.boundaries import BOUNDARY_NAMES, assemble, boundary_residuals
from lab7.solver import METHOD_NAMES, STATUS_NAMES, Config, Solution, solve

AXIS_NAMES = {"hx": "По x", "hy": "По y", "both": "По обеим координатам"}
REFERENCE_FACTORS = (8, 16)


def summary(result: Solution) -> dict:
    config = result.config
    return {
        "Метод": METHOD_NAMES[config.method],
        "Аппроксимация ГУ": BOUNDARY_NAMES[config.boundary],
        "Nx": config.nx, "Ny": config.ny, "hx": config.hx, "hy": config.hy,
        "ε": config.epsilon, "ω": config.sweep_weight, "max_iter": config.max_iter,
        "Итерации": result.iterations, "Статус": STATUS_NAMES[result.status],
        "E": result.error_history[-1], "||r||∞": result.residual_history[-1],
        "Оценка итерационной ошибки": result.iteration_error_bound,
    }


def history_table(result: Solution) -> pd.DataFrame:
    return pd.DataFrame({"k": np.arange(result.iterations + 1), "E": result.error_history,
                         "||r||∞": result.residual_history})


def result_tables(result: Solution, i: int, j: int) -> dict[str, pd.DataFrame]:
    x, y = np.meshgrid(result.x, result.y)
    values = pd.DataFrame({"x": x.ravel(), "y": y.ravel(), "u": result.values.ravel(),
                           "U": result.exact.ravel(), "|u−U|": result.absolute_error.ravel()})
    boundaries = boundary_residuals(result.values, result.x, result.y, result.config.boundary)
    return {
        "summary": pd.DataFrame([summary(result)]),
        "values": values,
        "section_y": values.iloc[j*len(result.x):(j+1)*len(result.x)],
        "section_x": values.iloc[i::len(result.x)],
        "history": history_table(result),
        "boundaries": pd.DataFrame(boundaries.items(), columns=["Грань", "Норма невязки"]),
    }


def compare(config: Config) -> tuple[Solution, ...]:
    return tuple(solve(replace(config, method=method)) for method in METHOD_NAMES)


def comparison_tables(results: tuple[Solution, ...]) -> dict[str, pd.DataFrame]:
    return {
        "comparison": pd.DataFrame(summary(result) for result in results),
        "comparison_history": pd.concat(
            [history_table(result).assign(Метод=METHOD_NAMES[result.config.method]) for result in results],
            ignore_index=True,
        ),
    }


def study_grid(config: Config, axis: str, count: int, fixed: int) -> Config:
    if axis == "hx":
        return replace(config, nx=count, ny=fixed)
    if axis == "hy":
        return replace(config, nx=fixed, ny=count)
    return replace(config, nx=count, ny=fixed)


def reference_field(config: Config) -> np.ndarray:
    grid = assemble(config.nx, config.ny, config.boundary)
    field = grid.template.copy()
    grid.expand(spsolve(grid.matrix, grid.rhs), field)
    return field


def on_coarse_grid(field: np.ndarray, fine: Config, coarse: Config) -> np.ndarray:
    return field[::fine.ny // coarse.ny, ::fine.nx // coarse.nx]


def add_orders(table: pd.DataFrame, error: str, order: str) -> None:
    values = table[error].where((table["Статус"] == STATUS_NAMES["converged"]) & (table[error] > 0))
    table[order] = np.log(values.shift() / values) / np.log(table["h"].shift() / table["h"])


def convergence(config: Config, axis: str, base: int, fixed: int, levels: int) -> pd.DataFrame:
    if axis not in AXIS_NAMES or levels < 2:
        raise ValueError("Выберите направление и хотя бы две сетки.")
    grids = [
        study_grid(config, axis, base*2**power, fixed*2**power if axis == "both" else fixed)
        for power in range(levels)
    ]
    reference = control = None
    if axis != "both":
        last = grids[-1]
        count = last.nx if axis == "hx" else last.ny
        control_config, reference_config = (
            study_grid(config, axis, factor*count, fixed) for factor in REFERENCE_FACTORS
        )
        control = reference_field(control_config)
        reference = reference_field(reference_config)

    rows = []
    for current in grids:
        result = solve(current)
        row = summary(result)
        row["h"] = current.hy if axis == "hy" else current.hx
        if reference is not None:
            fine = on_coarse_grid(reference, reference_config, current)
            control_sample = on_coarse_grid(control, control_config, current)
            row["E_ref"] = float(np.max(np.abs(result.values - fine)))
            row["Разность эталонов"] = float(np.max(np.abs(fine - control_sample)))
            row["Nx_ref"], row["Ny_ref"] = reference_config.nx, reference_config.ny
        rows.append(row)

    table = pd.DataFrame(rows)
    add_orders(table, "E", "p")
    if reference is not None:
        add_orders(table, "E_ref", "p_ref")
    return table
