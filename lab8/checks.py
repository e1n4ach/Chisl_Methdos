"""Численные проверки схем, прогонки, граничных условий и сходимости."""

from dataclasses import replace
import unittest

import numpy as np

from lab8.analysis import compare, convergence, maximum_errors
from lab8.problem import Problem
from lab8.solver import METHOD_NAMES, Config, Stepper, solve, steps_for_tau
from lab8.tridiagonal import factor_tridiagonal, solve_tridiagonal


def dxx(field, h):
    return (field[1:-1, :-2] - 2*field[1:-1, 1:-1] + field[1:-1, 2:]) / h**2


def dyy(field, h):
    return (field[:-2, 1:-1] - 2*field[1:-1, 1:-1] + field[2:, 1:-1]) / h**2


class NumericalChecks(unittest.TestCase):
    def assert_dirichlet(self, field, x, y, scale):
        np.testing.assert_allclose(field[:, 0], 0, rtol=0, atol=0)
        np.testing.assert_allclose(field[0, :], 0, rtol=0, atol=0)
        np.testing.assert_allclose(field[:, -1], y*scale, rtol=0, atol=0)
        np.testing.assert_allclose(field[-1, :], x*scale, rtol=0, atol=0)

    def test_analytic_solution(self):
        x, y = np.linspace(0, 1, 9), np.linspace(0, 1, 12)
        problem = Problem(x, y)
        xy = x[None, :]*y[:, None]
        np.testing.assert_array_equal(problem.exact(0), xy)
        np.testing.assert_array_equal(problem.initial(), xy)
        for t in (0, 0.37, 1.1, 2.3):
            with self.subTest(t=t):
                exact = problem.exact(t)
                self.assert_dirichlet(exact, x, y, np.cos(t))
                delta = 1e-5
                dt = (problem.exact(t+delta) - problem.exact(t-delta))/(2*delta)
                np.testing.assert_allclose(dt[1:-1, 1:-1],
                                           dxx(exact, x[1]) + dyy(exact, y[1]) + problem.source(t),
                                           rtol=0, atol=4e-11)

    def test_tridiagonal_batch(self):
        rng = np.random.default_rng(7)
        for size in (1, 2, 13):
            lower, upper = rng.uniform(-1, 1, (2, size-1))
            diagonal = rng.uniform(3, 4, size)
            rhs = rng.normal(size=(size, 6))
            matrix = np.diag(diagonal) + np.diag(lower, -1) + np.diag(upper, 1)
            factors = factor_tridiagonal(lower, diagonal, upper)
            np.testing.assert_allclose(solve_tridiagonal(factors, rhs), np.linalg.solve(matrix, rhs),
                                       rtol=1e-13, atol=1e-14)
            np.testing.assert_allclose(solve_tridiagonal(factors, rhs[:, 0]),
                                       np.linalg.solve(matrix, rhs[:, 0]), rtol=1e-13, atol=1e-14)
        with self.assertRaises(ValueError):
            factor_tridiagonal([1], [0, 1], [1])

    def test_intermediate_and_full_step_equations(self):
        for nx, ny in ((2, 2), (3, 8), (11, 4)):
            x, y = np.linspace(0, 1, nx+1), np.linspace(0, 1, ny+1)
            problem = Problem(x, y)
            interior = np.s_[1:-1, 1:-1]
            for method in METHOD_NAMES:
                with self.subTest(nx=nx, ny=ny, method=method):
                    config = Config(nx, ny, 8, 1.6, method)
                    t0, t1 = 0.8, 1.0
                    old = problem.exact(t0)
                    # Нелинейные по координатам значения различают явные и неявные проходы.
                    old[interior] += 0.08*np.sin(7*x[None, 1:-1])*np.cos(5*y[1:-1, None])
                    stage, new = Stepper(config, problem).advance(old, t0, t1)
                    self.assert_dirichlet(new, x, y, np.cos(t1))
                    tau = config.tau
                    if method == "adi":
                        force = tau/2 * problem.source((t0+t1)/2)
                        expected_boundary = (np.cos(t0)+np.cos(t1))/2
                        residual_x = (stage[interior] - old[interior] - force
                                      - tau/2*(dxx(stage, config.hx)+dyy(old, config.hy)))
                        residual_y = (new[interior] - stage[interior] - force
                                      - tau/2*(dxx(stage, config.hx)+dyy(new, config.hy)))
                    else:
                        force = tau/2 * problem.source(t1)
                        expected_boundary = np.cos(t1) + tau/2*np.sin(t1)
                        residual_x = stage[interior] - old[interior] - force - tau*dxx(stage, config.hx)
                        residual_y = new[interior] - stage[interior] - force - tau*dyy(new, config.hy)
                    self.assert_dirichlet(stage, x, y, expected_boundary)
                    np.testing.assert_allclose(residual_x, 0, rtol=0, atol=2e-14)
                    np.testing.assert_allclose(residual_y, 0, rtol=0, atol=2e-14)

    def test_history_boundaries_and_finite_values(self):
        for nx, ny, count, time in ((2, 9, 1, 1.7), (7, 2, 13, 2.4), (18, 12, 50, 0.35)):
            for method in METHOD_NAMES:
                with self.subTest(nx=nx, ny=ny, M=count, method=method):
                    config = Config(nx, ny, count, time, method)
                    result = solve(config)
                    self.assertEqual(result.values.shape, (count+1, ny+1, nx+1))
                    self.assertEqual(result.time[-1], time)
                    self.assertTrue(np.isfinite(result.values).all())
                    np.testing.assert_array_equal(result.values[0], result.problem.initial())
                    for t, field in zip(result.time, result.values):
                        self.assert_dirichlet(field, result.problem.x, result.problem.y, np.cos(t))
                    np.testing.assert_array_equal(solve(config, terminal_only=True).values[0],
                                                  result.values[-1])
                    self.assertEqual(maximum_errors(result)[0], 0)
        self.assertEqual(steps_for_tau(1.0, 0.3), 4)
        self.assertEqual(steps_for_tau(0.28, 0.01), 28)
        with self.assertRaises(ValueError):
            steps_for_tau(1.0, float("nan"))
        with self.assertRaises((ValueError, FloatingPointError)):
            solve(Config(nx=6, ny=7, steps=1, final_time=1e308))
        with self.assertRaises(ValueError):
            solve(Config(nx=150, ny=150, steps=1000))

    def test_convergence_and_comparison(self):
        config = Config()
        for axis, fixed in (("hx", {"Ny": config.ny, "M": config.steps}),
                            ("hy", {"Nx": config.nx, "M": config.steps}),
                            ("tau", {"Nx": config.nx, "Ny": config.ny})):
            space = axis != "tau"
            table = convergence(config, axis, 6 if space else 60, 3)
            for column, value in {"T": 1, **fixed}.items():
                self.assertEqual(set(table[column]), {value})
            for method, name in METHOD_NAMES.items():
                with self.subTest(axis=axis, method=method):
                    rows = table[table["Метод"] == name]
                    err, order = ("E_ref(T)", "p_ref") if space else ("E(T)", "p")
                    self.assertTrue((np.diff(rows[err]) < 0).all())
                    measured = np.log(rows[err].iloc[-2]/rows[err].iloc[-1])/np.log(2)
                    self.assertAlmostEqual(rows[order].iloc[-1], measured)
                    self.assertAlmostEqual(measured, 2 if space or method == "adi" else 1, delta=0.08)
                    if space:
                        self.assertTrue((rows["Разность эталонов"] < 0.07*rows[err]).all())
        comparison = compare(replace(config, nx=8, ny=12), 2)
        self.assertEqual(len(comparison.timings), 4)
        self.assertTrue((comparison.timings["Время, с"] > 0).all())
        self.assertLess(comparison.summary["E(T)"].iloc[0], comparison.summary["E(T)"].iloc[1])
