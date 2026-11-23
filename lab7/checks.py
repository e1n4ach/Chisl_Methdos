"""Численные проверки варианта 3. Запуск python -m unittest lab7.checks."""

from dataclasses import replace
from math import pi
import unittest

import numpy as np

from lab7.analysis import convergence
from lab7.boundaries import BOUNDARY_NAMES, assemble, boundary_residuals
from lab7.problem import bottom_derivative, exact_solution, left_value, right_value, top_derivative
from lab7.solver import METHOD_NAMES, Config, solve


class NumericalChecks(unittest.TestCase):
    def test_exact_solution(self):
        x = np.linspace(0, 1, 11)[None, :]
        y = np.linspace(0, pi/2, 13)[:, None]
        delta = 1e-4
        exact = exact_solution(x, y)
        dxx = (exact_solution(x+delta, y) - 2*exact + exact_solution(x-delta, y)) / delta**2
        dyy = (exact_solution(x, y+delta) - 2*exact + exact_solution(x, y-delta)) / delta**2
        np.testing.assert_allclose(dxx, exact, atol=1e-7)
        np.testing.assert_allclose(dyy, -exact, atol=1e-7)
        np.testing.assert_array_equal(exact_solution(0, y), left_value(y))
        np.testing.assert_array_equal(exact_solution(1, y), right_value(y))
        for edge, target in ((0, bottom_derivative(x)), (pi/2, top_derivative(x))):
            derivative = (exact_solution(x, edge+delta) - exact_solution(x, edge-delta)) / (2*delta)
            np.testing.assert_allclose(derivative, target, atol=1e-8)

    def test_stencils_and_discrete_solution(self):
        for boundary in BOUNDARY_NAMES:
            for nx, ny in ((2, 3), (8, 11), (12, 7)):
                grid = assemble(nx, ny, boundary)
                # Независимый эталон для итераций на небольшой сетке.
                reference = grid.template.copy()
                grid.expand(np.linalg.solve(grid.matrix.toarray(), grid.rhs), reference)
                for method in METHOD_NAMES:
                    with self.subTest(boundary=boundary, method=method, nx=nx, ny=ny):
                        config = Config(nx, ny, method, boundary)
                        result = solve(config)
                        self.assertEqual(result.status, "converged")
                        self.assertLessEqual(result.residual_history[-1], config.epsilon)
                        self.assertGreater(result.residual_history[-2], config.epsilon)
                        error_to_reference = np.max(np.abs(result.values - reference))
                        self.assertLessEqual(error_to_reference, result.iteration_error_bound + 2e-13)
                        u = result.values
                        center = u[1:-1, 1:-1]
                        laplacian = ((u[1:-1, :-2] - 2*center + u[1:-1, 2:])/config.hx**2
                                     + (u[:-2, 1:-1] - 2*center + u[2:, 1:-1])/config.hy**2)
                        self.assertLessEqual(np.max(np.abs(laplacian)), config.epsilon + 1e-12)
                        checks = boundary_residuals(u, result.x, result.y, boundary)
                        self.assertEqual(checks["x=0"], 0)
                        self.assertEqual(checks["x=1"], 0)
                        tolerance = config.hy/2 * config.epsilon if boundary == "ghost" else 0
                        self.assertLessEqual(max(checks.values()), tolerance + 1e-13)
                        self.assertEqual(result.error_history[-1], result.absolute_error.max())

    def test_first_sweep_and_iteration_limit(self):
        for boundary in BOUNDARY_NAMES:
            grid = assemble(5, 7, boundary)
            matrix = grid.matrix.toarray()
            config = Config(5, 7, boundary=boundary, max_iter=1)
            for method in METHOD_NAMES:
                with self.subTest(boundary=boundary, method=method):
                    current = replace(config, method=method)
                    result = solve(current)
                    self.assertEqual(result.status, "limit")
                    self.assertEqual(result.iterations, 1)
                    vector = np.zeros(matrix.shape[0])
                    for i in range(len(vector)):
                        used = matrix[i, :i] @ vector[:i] if method != "jacobi" else 0
                        vector[i] = current.sweep_weight * (grid.rhs[i] - used) / matrix[i, i]
                    np.testing.assert_allclose(result.values[grid.active_rows, 1:-1].ravel(), vector)
            seidel = solve(replace(config, method="seidel"))
            sor = solve(replace(config, omega=1))
            np.testing.assert_array_equal(sor.values, seidel.values)
        for invalid in ({"ny": 2}, {"epsilon": np.nan}, {"omega": 2}, {"max_iter": 0}):
            with self.assertRaises(ValueError):
                Config(**invalid)

    def test_second_order_and_independent_steps(self):
        for boundary in BOUNDARY_NAMES:
            config = Config(boundary=boundary)
            for direction, fixed in (("hx", 24), ("hy", 24), ("both", 8)):
                with self.subTest(boundary=boundary, direction=direction):
                    table = convergence(config, direction, 6, fixed, 3)
                    error, order = ("E", "p") if direction == "both" else ("E_ref", "p_ref")
                    self.assertTrue(np.all(np.diff(table[error]) < 0))
                    self.assertTrue(np.all(np.abs(table[order].iloc[1:] - 2) < 0.15))
                    self.assertTrue(np.all(table["Оценка итерационной ошибки"] < 0.01*table[error]))
                    if direction != "both":
                        fixed_column = "Ny" if direction == "hx" else "Nx"
                        self.assertEqual(set(table[fixed_column]), {fixed})
                        self.assertTrue(np.all(table["Разность эталонов"] < 0.05*table[error]))
        stopped = convergence(Config(max_iter=1), "both", 6, 8, 2)
        self.assertTrue(stopped["p"].isna().all())
