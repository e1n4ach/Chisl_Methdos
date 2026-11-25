"""Ошибки, таблицы, измерение времени и независимое сгущение сетки."""

from dataclasses import dataclass, replace
from time import perf_counter

import numpy as np
import pandas as pd

from lab8.solver import METHOD_NAMES, Config, Solution, solve

AXIS_NAMES = {"hx": "По x", "hy": "По y", "tau": "По времени"}
REFERENCE_FACTORS = (4, 8)


def maximum_errors(result: Solution) -> np.ndarray:
    return np.array([np.max(np.abs(values-result.problem.exact(t)))
                     for t, values in zip(result.time, result.values)])


def trace_points(config: Config, i: int, j: int):
    candidates = [(i, j), (max(1, config.nx//4), max(1, config.ny//4)),
                  (3*config.nx//4, 3*config.ny//4)]
    return list(dict.fromkeys(candidates))


def point_history(result: Solution, i: int, j: int) -> pd.DataFrame:
    tables = []
    for xi, yj in trace_points(result.config, i, j):
        x, y = result.problem.x[xi], result.problem.y[yj]
        values = result.values[:, yj, xi]
        exact = result.problem.exact(result.time, (yj, xi))
        tables.append(pd.DataFrame({"i": xi, "j": yj, "x": x, "y": y, "t": result.time,
                                    "u": values, "U": exact, "|u−U|": np.abs(values-exact)}))
    return pd.concat(tables, ignore_index=True)


def parameters(config):
    return {"Метод": METHOD_NAMES[config.method], "Nx": config.nx, "Ny": config.ny,
            "M": config.steps, "T": config.final_time,
            "hx": config.hx, "hy": config.hy, "τ": config.tau}


def solution_tables(result, k, i, j):
    t = result.time[k]
    x, y = np.meshgrid(result.problem.x, result.problem.y)
    values, exact = result.values[k], result.problem.exact(t)
    errors = maximum_errors(result)
    table = pd.DataFrame({"i": np.tile(np.arange(x.shape[1]), x.shape[0]),
                          "j": np.repeat(np.arange(x.shape[0]), x.shape[1]),
                          "x": x.ravel(), "y": y.ravel(), "t": t, "u": values.ravel(),
                          "U": exact.ravel(), "|u−U|": np.abs(values-exact).ravel()})
    return {
        "summary": pd.DataFrame([{**parameters(result.config), "k": k, "t": t,
                                  "E_max(t)": errors[k], "E(T)": errors[-1],
                                  "max_t E_max": errors.max()}]),
        "values": table,
        "section_y": table.iloc[j*x.shape[1]:(j+1)*x.shape[1]],
        "section_x": table.iloc[i::x.shape[1]],
        "error_time": pd.DataFrame({"t": result.time, "E_max(t)": errors}),
        "point_history": point_history(result, i, j),
    }


@dataclass
class Comparison:
    summary: pd.DataFrame
    histories: pd.DataFrame
    timings: pd.DataFrame

    def tables(self):
        return {"comparison": self.summary, "comparison_history": self.histories,
                "comparison_timings": self.timings}


def compare(config: Config, repeats: int = 3) -> Comparison:
    if repeats < 1:
        raise ValueError("Требуется хотя бы один замер.")
    rows, histories, timings = [], [], []
    for method, name in METHOD_NAMES.items():
        current = replace(config, method=method)
        elapsed = []
        for run in range(repeats):
            start = perf_counter()
            result = solve(current)
            elapsed.append(perf_counter()-start)
            timings.append({"Метод": name, "Запуск": run + 1, "Время, с": elapsed[-1]})
        errors = maximum_errors(result)
        rows.append({**parameters(current), "Повторы": repeats, "Медиана, с": np.median(elapsed),
                     "Минимум, с": min(elapsed), "Максимум, с": max(elapsed),
                     "E(T)": errors[-1], "max_t E_max": errors.max()})
        histories.append(pd.DataFrame({"Метод": name, "t": result.time, "E_max(t)": errors}))
        del result
    return Comparison(pd.DataFrame(rows), pd.concat(histories, ignore_index=True), pd.DataFrame(timings))


def terminal(config, manufactured=False):
    result = solve(config, terminal_only=True, manufactured=manufactured)
    return result.values[-1], result.problem.exact(config.final_time)


def orders(errors, steps):
    # Значения порядка ниже разрешения округления численно неинформативны.
    values = errors.where(errors > 64*np.finfo(float).eps)
    return np.log(values.shift()/values) / np.log(steps.shift()/steps)


def convergence(config: Config, direction: str, base: int, levels: int) -> pd.DataFrame:
    if direction not in AXIS_NAMES or levels < 2 or base < (1 if direction == "tau" else 2):
        raise ValueError("Неверные параметры серии сгущения.")
    parameter = {"hx": "nx", "hy": "ny", "tau": "steps"}[direction]
    tables = []
    for method in METHOD_NAMES:
        current = replace(config, method=method)
        rows = []
        if direction != "tau":
            finest = base*2**(levels-1)
            coarse_factor, fine_factor = REFERENCE_FACTORS
            control, reference = (terminal(replace(current, **{parameter: factor*finest}), True)[0]
                                  for factor in REFERENCE_FACTORS)
            axis = 1 if direction == "hx" else 0
        for level in range(levels):
            count = base*2**level
            grid = replace(current, **{parameter: count})
            values, exact = terminal(grid)
            row = {**parameters(grid), "s": getattr(grid, direction),
                   "E(T)": float(np.max(np.abs(values-exact)))}
            if direction != "tau":
                diagnostic, _ = terminal(grid, True)
                factor = finest//count
                sample = [slice(None), slice(None)]
                sample[axis] = slice(None, None, fine_factor*factor)
                fine_values = reference[tuple(sample)]
                sample[axis] = slice(None, None, coarse_factor*factor)
                control_values = control[tuple(sample)]
                row["E_ref(T)"] = float(np.max(np.abs(diagnostic-fine_values)))
                row["Разность эталонов"] = float(np.max(np.abs(fine_values-control_values)))
                row["Интервалы эталона"] = fine_factor*finest
            rows.append(row)
        table = pd.DataFrame(rows)
        table["p"] = orders(table["E(T)"], table["s"])
        if direction != "tau":
            table["p_ref"] = orders(table["E_ref(T)"], table["s"])
        tables.append(table)
    return pd.concat(tables, ignore_index=True)
