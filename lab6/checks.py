"""Запуск: python lab6/checks.py. Проверки независимыми формулами и плотной системой."""

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from analysis import COMBINATIONS, boundary_error, compare, convergence, max_errors
from boundaries import boundary_residuals, restore_edges, spatial_operator
from problem import exact_solution, initial_acceleration, initial_displacement, initial_velocity
from plots import save_report_plots
from solver import Config, TridiagonalSolver, solve, stability


def check_exact_solution() -> None:
    x = np.linspace(0, 1, 17)[:, None]
    time = np.linspace(0, 2, 13)[None, :]
    exponential = np.exp(2*x)
    u = exponential*np.cos(time)
    u_tt = -exponential*np.cos(time)
    u_xx = 4*exponential*np.cos(time)
    assert_allclose(u_tt-u_xx+5*u, 0, atol=1e-14)
    for edge in (0.0, 1.0):
        u_edge = np.exp(2*edge)*np.cos(time)
        derivative = 2*np.exp(2*edge)*np.cos(time)
        assert_allclose(derivative-2*u_edge, 0, atol=1e-14)
    assert_allclose(exact_solution(x, 0), initial_displacement(x))
    assert_allclose(-exponential*np.sin(0), 0)
    assert_allclose(initial_acceleration(x), 4*exponential-5*exponential)


def check_dense_system() -> None:
    """Исходная полная матрица с тремя точками на краях против исключения границ."""
    for boundary in ("first", "three", "pde"):
        config = Config(n=20, steps=20, scheme="implicit", boundary=boundary)
        result = solve(config)
        n, h, tau = config.n, config.h, config.tau
        matrix = np.zeros((n+1, n+1))
        for i in range(1, n):
            matrix[i, i-1:i+2] = [-tau**2/h**2, 1+2*tau**2/h**2+5*tau**2, -tau**2/h**2]
        if boundary == "first":
            matrix[0, :2] = [-1/h-2, 1/h]
            matrix[-1, -2:] = [-1/h, 1/h-2]
        elif boundary == "three":
            matrix[0, :3] = [-3/(2*h)-2, 2/h, -1/(2*h)]
            matrix[-1, -3:] = [1/(2*h), -2/h, 3/(2*h)-2]
        else:
            matrix[0, :2] = [1+tau**2*(2/h**2+4/h+5), -2*tau**2/h**2]
            matrix[-1, -2:] = [-2*tau**2/h**2, 1+tau**2*(2/h**2-4/h+5)]
        for layer in (2, 5, 20):
            rhs = 2*result.values[layer-1]-result.values[layer-2]
            if boundary != "pde":
                rhs[[0, -1]] = 0
            assert_allclose(result.values[layer], np.linalg.solve(matrix, rhs), atol=2e-12)
        operator = spatial_operator(config.n, boundary)
        reduced_matrix = (np.diag(1-tau**2*operator.diagonal)
                          +np.diag(-tau**2*operator.lower, -1)+np.diag(-tau**2*operator.upper, 1))
        rhs = np.arange(len(operator.diagonal), dtype=float)
        assert_allclose(TridiagonalSolver(operator, tau).solve(rhs), np.linalg.solve(reduced_matrix, rhs), atol=1e-12)


def check_combinations() -> None:
    for scheme, order, boundary in COMBINATIONS:
        config = Config(scheme=scheme, initial_order=order, boundary=boundary)
        result = solve(config)
        assert np.isfinite(result.values).all()
        assert_allclose(result.values[0], initial_displacement(result.x), atol=0)
        expected = initial_displacement(result.x)
        # Независимое вычисление L_h*phi через исходные разностные формулы.
        # Алгебраические края восстанавливаются ДО вычисления ускорения.
        discrete_initial = expected.copy()
        if boundary != "pde":
            restore_edges(discrete_initial, config.h, boundary)
        acceleration = np.empty_like(expected)
        acceleration[1:-1] = (np.diff(discrete_initial, n=2)/config.h**2
                              -5*discrete_initial[1:-1])
        if boundary == "pde":
            acceleration[0] = 2*(expected[1]-expected[0])/config.h**2 - (4/config.h+5)*expected[0]
            acceleration[-1] = 2*(expected[-2]-expected[-1])/config.h**2 + (4/config.h-5)*expected[-1]
        active = slice(None) if boundary == "pde" else slice(1, -1)
        if order == 2:
            expected[active] += config.tau**2/2*acceleration[active]
        assert_allclose(result.values[1, active], expected[active], rtol=0, atol=2e-14)
        if boundary == "pde" and order == 2:
            # A*=L_h*phi относится к t=0, а не к u^1.
            startup_acceleration = 2*(result.values[1]-result.values[0]
                                     -config.tau*initial_velocity(result.x))/config.tau**2
            assert_allclose(boundary_residuals(result.values[0], config.h, boundary,
                                               startup_acceleration), 0, atol=2e-11)
        assert boundary_error(result) < 5e-9
        # Независимо проверяем, на каком слое взяты Dxx и -5u.
        for center in (1, 17, config.steps-1):
            acceleration = (result.values[center+1]-2*result.values[center]+result.values[center-1])/config.tau**2
            level = center if scheme == "explicit" else center+1
            values = result.values[level]
            rhs = (values[:-2]-2*values[1:-1]+values[2:])/config.h**2-5*values[1:-1]
            assert_allclose(acceleration[1:-1], rhs, atol=3e-9, rtol=1e-9)
        coarse = solve(replace(config, n=20, steps=2000))
        fine = solve(replace(config, n=40, steps=4000))
        assert max_errors(fine)[-1] < max_errors(coarse)[-1], (scheme, order, boundary)
        terminal = solve(config, terminal_only=True)
        assert_allclose(terminal.values[-1], result.values[-1], atol=0)


def check_stability() -> None:
    for boundary in ("first", "three", "pde"):
        base = Config(boundary=boundary)
        diagnostic = stability(base)
        assert diagnostic.stable
        operator = spatial_operator(base.n, boundary)
        dense = np.diag(operator.diagonal)+np.diag(operator.lower, -1)+np.diag(operator.upper, 1)
        eigenvalues = np.linalg.eigvals(dense)
        assert_allclose([diagnostic.lambda_min, diagnostic.lambda_max],
                        [eigenvalues.min(), eigenvalues.max()], atol=2e-11)
        for eigenvalue in (diagnostic.lambda_min, diagnostic.lambda_max):
            roots = np.roots([1.0, -(2+base.tau**2*eigenvalue), 1.0])
            assert np.max(np.abs(roots)) < 1+1e-10
        too_large = replace(base, steps=10)
        try:
            solve(too_large)
            raise AssertionError("Неустойчивая сетка не отклонена")
        except ValueError as exc:
            assert "Неустойчивая" in str(exc)
    # Реакция уже меняет условие для внутренних узлов при tau/h=1.
    assert not stability(Config(n=40, steps=40, boundary="pde")).stable
    # Крупная Robin-сетка может иметь положительную моду при любом tau.
    assert stability(Config(n=3, boundary="first")).lambda_max > 0
    assert not stability(Config(n=3, boundary="first")).stable
    # Неявный метод реально работает за пределом явного ограничения.
    result = solve(Config(steps=10, scheme="implicit"))
    assert np.isfinite(result.values).all()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    checks = [check_exact_solution, check_dense_system, check_combinations, check_stability]
    for check in checks:
        check()
        print(f"PASS {check.__name__}", flush=True)
    output = Path(__file__).parent / "results"
    output.mkdir(exist_ok=True)
    comparison, histories = compare(Config())
    comparison.to_csv(output / "comparison.csv", index=False, encoding="utf-8-sig")
    print("\nN=40, M=200, T=1\n"+comparison.to_string(index=False), flush=True)
    studies = []
    for scheme, order, boundary in COMBINATIONS:
        config = Config(scheme=scheme, initial_order=order, boundary=boundary)
        table = convergence(config, "h", (12, 24, 48, 96), 20000)
        table.insert(0, "boundary", boundary)
        table.insert(0, "initial_order", order)
        table.insert(0, "scheme", scheme)
        table["fixed_step_effect"] = table.attrs["fixed_step_effect"]
        assert np.all(np.diff(table["ε(T)"]) < 0), (scheme, order, boundary)
        studies.append(table)
        print(f"h: {scheme:8} start={order} {boundary:5} "
              f"error={table['ε(T)'].iloc[-1]:.6e} p={table['p(T)'].iloc[-1]:.4f} "
              f"fixed_effect={table.attrs['fixed_step_effect']:.2%}", flush=True)
    space_table = pd.concat(studies, ignore_index=True)
    space_table.to_csv(output / "convergence_h.csv", index=False, encoding="utf-8-sig")
    time_studies = []
    for scheme, order, boundary in COMBINATIONS:
        config = Config(scheme=scheme, initial_order=order, boundary=boundary)
        # У first/implicit высокочастотные моды несовместимого начального профиля
        # требуют меньшего tau для выхода на асимптотический режим.
        grids = (640, 1280, 2560, 5120) if (scheme, order, boundary) == ("implicit", 2, "first") else (160, 320, 640, 1280)
        table = convergence(config, "tau", grids, 60)
        table.insert(0, "boundary", boundary)
        table.insert(0, "initial_order", order)
        table.insert(0, "scheme", scheme)
        table["reference_difference"] = table.attrs["reference_difference"]
        table["reference_steps"] = table.attrs["reference_steps"]
        differences = table["Разность с временным эталоном"]
        assert np.all(np.diff(differences) < 0), (scheme, order, boundary)
        assert table.attrs["reference_difference"] < 0.1*differences.iloc[-1]
        # Это регрессионные допуски к ИЗМЕРЕННЫМ порядкам, не значения для таблицы.
        measured_orders = table["p(время)"].iloc[-2:].to_numpy()
        if scheme == "explicit" and order == 2:
            assert np.all((1.85 < measured_orders) & (measured_orders < 2.15)), (boundary, measured_orders)
        else:
            assert np.all((0.85 < measured_orders) & (measured_orders < 1.15)), (scheme, order, boundary, measured_orders)
        time_studies.append(table)
        print(f"\ntau: {scheme}, start={order}, {boundary}\n"+table.to_string(index=False), flush=True)
        print(f"reference difference={table.attrs['reference_difference']:.3e}; "
              f"fixed h effect={table.attrs['fixed_step_effect']:.2%}", flush=True)
    time_table = pd.concat(time_studies, ignore_index=True)
    time_table.to_csv(output / "convergence_tau.csv", index=False, encoding="utf-8-sig")
    save_report_plots(histories, space_table, time_table, output)
    print("\nPASS: все 12 комбинаций, согласованный старт, независимые системы, ГУ, устойчивость, сгущение и порядки по времени.")


if __name__ == "__main__":
    main()
