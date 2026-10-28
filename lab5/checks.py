"""Проверки формул, аналитического решения и сходимости: python lab5/checks.py."""

import numpy as np

from lab5 import (Problem, SCHEMES, BOUNDARIES, boundary_relation,
                  explicit_stability, factor_tridiagonal, max_error,
                  solve, solve_tridiagonal)


class ManufacturedProblem(Problem):
    """U=exp(-t)*(sin x+cos x): Uxx(0,t) != 0, нет сверхсходимости ГУ."""

    def initial(self, x):
        return np.sin(x)+np.cos(x)

    def exact(self, x, t):
        return np.exp(-t)*(np.sin(x)+np.cos(x))

    def source(self, x, t):
        return (-1+self.a-self.c)*self.exact(x, t)

    def left(self, t):
        return (self.alpha+self.beta)*np.exp(-t)

    def right(self, t):
        return self.exact(self.length, t)


def check_tridiagonal():
    lower = np.array([-1., -2., -1.])
    diagonal = np.array([4., 5., 6., 4.])
    upper = np.array([-2., -1., -1.])
    rhs = np.arange(1., 5.)
    matrix = np.diag(diagonal)+np.diag(lower, -1)+np.diag(upper, 1)
    actual = solve_tridiagonal(factor_tridiagonal(lower, diagonal, upper), rhs)
    np.testing.assert_allclose(actual, np.linalg.solve(matrix, rhs), atol=1e-14)


def check_convergence(problem, generic=False):
    sizes = [10, 20, 40]
    for boundary in BOUNDARIES:
        for scheme in SCHEMES:
            errors = []
            for n in sizes:
                x, t, u = solve(problem, n, 0.4, sigma=0.35, scheme=scheme, boundary=boundary)
                np.testing.assert_array_equal(u[0], problem.initial(x))
                np.testing.assert_allclose(u[:, -1], problem.right(t), atol=1e-14)
                if boundary != "pde":
                    k1, k2, g = boundary_relation(problem, x[1], boundary)
                    np.testing.assert_allclose(u[1:, 0], k1*u[1:, 1]+k2*u[1:, 2]
                                               +g*problem.left(t[1:]), atol=1e-13)
                errors.append(max_error(problem, x, t, u)[-1])
            order = np.log2(errors[-2]/errors[-1])
            target = 1 if generic and boundary == "first" else 2
            assert all(a > b for a, b in zip(errors, errors[1:])), (scheme, boundary, errors)
            assert abs(order-target) < 0.35, (scheme, boundary, order)
            print(f"{'Общая задача' if generic else 'Вариант 8':13s} "
                  f"{scheme:8s}/{boundary:5s}: p={order:.3f}, error={errors[-1]:.3e}")


def check_stability():
    problem = Problem()
    h = problem.length/40
    for boundary in BOUNDARIES:
        assert explicit_stability(problem, 40, 0.4*h*h, boundary)[1] < 1
        assert explicit_stability(problem, 40, 0.6*h*h, boundary)[1] > 1
        try:
            solve(problem, 40, 0.3, sigma=0.6, scheme="explicit", boundary=boundary)
        except ValueError as error:
            assert "неустойчива" in str(error)
        else:
            raise AssertionError("Неустойчивая сетка не отклонена")


def main():
    check_tridiagonal()
    check_convergence(Problem())
    # Проверяет также ненулевые beta и f, другие a,c и длину области.
    check_convergence(ManufacturedProblem(a=0.7, c=-0.4, beta=-0.3, length=1.2), True)
    check_stability()
    print("Все проверки пройдены.")


if __name__ == "__main__":
    main()
