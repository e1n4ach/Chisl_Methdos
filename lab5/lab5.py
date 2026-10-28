"""Лабораторная 5, вариант 8. Разностные схемы для u_t = a*u_xx + c*u."""

import argparse
from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import matplotlib.pyplot as plt


SCHEMES = {"explicit": 0.0, "implicit": 1.0, "cn": 0.5}
BOUNDARIES = ("first", "three", "pde")
SCHEME_NAMES = {"explicit": "Явная", "implicit": "Неявная", "cn": "Кранк — Николсон"}
BC_NAMES = {"first": "2 точки, 1-й порядок", "three": "3 точки, 2-й порядок",
            "pde": "2 точки, 2-й порядок (уравнение)"}


@dataclass
class Problem:
    a: float = 1.0
    c: float = -1.0
    length: float = np.pi / 2
    alpha: float = 1.0  # alpha*u_x(0,t) + beta*u(0,t) = left(t)
    beta: float = 0.0
    left_amplitude: float = 1.0
    right_amplitude: float = 1.0
    left_rate: float | None = None
    right_rate: float | None = None

    def initial(self, x):
        return np.sin(x)  # начальное условие задаётся точно, включая границы

    def source(self, x, t):
        return np.zeros_like(x, dtype=float)  # при необходимости заменить f(x,t)

    def left(self, t):
        rate = self.c - self.a if self.left_rate is None else self.left_rate
        return self.left_amplitude * np.exp(rate * t)

    def right(self, t):
        rate = self.c - self.a if self.right_rate is None else self.right_rate
        return self.right_amplitude * np.exp(rate * t)

    def exact(self, x, t):
        return np.exp((self.c - self.a) * t) * np.sin(x)

    def has_exact(self):
        """При независимом изменении ГУ прежнее U может перестать быть решением."""
        return (np.isclose(self.left_amplitude, self.alpha)
                and np.isclose(self.right_amplitude, np.sin(self.length))
                and (self.left_rate is None or np.isclose(self.left_rate, self.c-self.a))
                and (self.right_rate is None or np.isclose(self.right_rate, self.c-self.a)))


def boundary_relation(problem, h, boundary):
    """Алгебраическое ГУ: u0 = k1*u1 + k2*u2 + g*left(t)."""
    alpha, beta = problem.alpha, problem.beta
    if boundary == "first":
        denominator = beta - alpha / h
        coefficients = (-alpha / h, 0.0, 1.0)
    else:
        denominator = beta - 3 * alpha / (2*h)
        coefficients = (-2*alpha/h, alpha/(2*h), 1.0)
    if abs(denominator) < 1e-14:
        raise ValueError("Вырожденная аппроксимация левого ГУ; измените h или alpha, beta.")
    return np.array(coefficients) / denominator


def spatial_operator(problem, n, boundary):
    """Три диагонали L и добавка g(t) в системе v_t = L*v + g(t)."""
    h = problem.length / n
    diffusion = problem.a / h**2
    start = 0 if boundary == "pde" else 1
    x = np.linspace(0, problem.length, n+1)
    size = n - start
    lower = np.full(size-1, diffusion)
    diagonal = np.full(size, -2*diffusion + problem.c)
    upper = np.full(size-1, diffusion)
    if boundary == "pde":
        # u1 = u0 + h*ux + h²/(2a)*(ut-c*u0-f) + O(h³).
        # Исключаем ux=(left-beta*u0)/alpha и выражаем ut в узле 0.
        diagonal[0] += 2*problem.a*problem.beta/(problem.alpha*h)
        upper[0] = 2*diffusion
    else:
        k1, k2, _ = boundary_relation(problem, h, boundary)
        diagonal[0] += diffusion*k1
        upper[0] += diffusion*k2

    def forcing(t):
        values = np.broadcast_to(problem.source(x[start:n], t), (size,)).copy()
        values[-1] += diffusion * problem.right(t)
        if boundary == "pde":
            values[0] -= 2*problem.a/(problem.alpha*h) * problem.left(t)
        else:
            values[0] += diffusion*boundary_relation(problem, h, boundary)[2]*problem.left(t)
        return values

    return lower, diagonal, upper, forcing


def apply_operator(lower, diagonal, upper, values):
    result = diagonal * values
    result[1:] += lower * values[:-1]
    result[:-1] += upper * values[1:]
    return result


def factor_tridiagonal(lower, diagonal, upper):
    """Прямая часть прогонки; матрица постоянна на всей временной сетке."""
    pivots = diagonal.copy()
    multipliers = lower.copy()
    for i in range(1, len(pivots)):
        if abs(pivots[i-1]) < 1e-14:
            raise ValueError("Нулевой ведущий элемент прогонки.")
        multipliers[i-1] /= pivots[i-1]
        pivots[i] -= multipliers[i-1]*upper[i-1]
    if abs(pivots[-1]) < 1e-14:
        raise ValueError("Вырожденная матрица схемы.")
    return multipliers, pivots, upper


def solve_tridiagonal(factors, rhs):
    multipliers, pivots, upper = factors
    result = rhs.copy()
    for i in range(1, len(result)):
        result[i] -= multipliers[i-1]*result[i-1]
    result[-1] /= pivots[-1]
    for i in range(len(result)-2, -1, -1):
        result[i] = (result[i]-upper[i]*result[i+1])/pivots[i]
    return result


def explicit_stability(problem, n, tau, boundary):
    """Спектр полной схемы с выбранным ГУ, без неоднородных добавок."""
    lower, diagonal, upper, _ = spatial_operator(problem, n, boundary)
    matrix = np.diag(diagonal) + np.diag(lower, -1) + np.diag(upper, 1)
    eigenvalues = np.linalg.eigvals(matrix)
    radius = float(np.max(np.abs(1 + tau*eigenvalues)))
    sigma = problem.a*tau/(problem.length/n)**2
    # Для c<=0 достаточное условие монотонности внутренних строк:
    # 2*sigma - c*tau <= 1. Спектр дополнительно учитывает левое ГУ.
    interior_ok = 2*sigma - problem.c*tau <= 1 + 1e-12 if problem.c <= 0 else None
    return sigma, radius, interior_ok


def solve(problem, n=40, final_time=1.0, steps=None, sigma=0.4,
          scheme="cn", boundary="pde", allow_unstable=False, initial_values=None):
    """Возвращает x, t, u[k,i]. steps задаёт tau=T/steps, иначе используется sigma."""
    if n < 3 or final_time <= 0 or problem.a <= 0 or problem.length <= 0:
        raise ValueError("Нужны n>=3, T>0, a>0, length>0.")
    if problem.alpha == 0:
        raise ValueError("Для левого ГУ с производной требуется alpha != 0.")
    if scheme not in SCHEMES or boundary not in BOUNDARIES:
        raise ValueError("Неизвестная схема или аппроксимация ГУ.")
    
    h = problem.length/n
    if steps is None:
        if sigma <= 0:
            raise ValueError("sigma должна быть положительной.")
        steps = int(np.ceil(final_time*problem.a/(sigma*h*h)))
    if steps < 1 or int(steps) != steps:
        raise ValueError("steps должно быть положительным целым.")
    steps = int(steps)
    tau = final_time/steps
    if scheme == "explicit":
        _, radius, _ = explicit_stability(problem, n, tau, boundary)
        if radius > 1+1e-10:
            message = f"Явная схема: rho(I+tau*L)={radius:.6f}>1; сетка неустойчива."
            if not allow_unstable:
                raise ValueError(message + " Для эксперимента: --allow-unstable.")
            warnings.warn(message, stacklevel=2)
    theta = SCHEMES[scheme]
    lower, diagonal, upper, forcing = spatial_operator(problem, n, boundary)
    factors = None
    if theta:
        factors = factor_tridiagonal(-theta*tau*lower, 1-theta*tau*diagonal,
                                     -theta*tau*upper)
    x = np.linspace(0, problem.length, n+1)
    t = np.linspace(0, final_time, steps+1)
    u = np.zeros((steps+1, n+1))
    u[0] = problem.initial(x) if initial_values is None else initial_values
    if not np.isclose(u[0, -1], problem.right(0), atol=1e-12, rtol=1e-12):
        raise ValueError("Начальное условие несовместимо с правым ГУ при t=0.")
    start = 0 if boundary == "pde" else 1
    old_forcing = forcing(0)
    for k in range(steps):
        old = u[k, start:n]
        new_forcing = forcing(t[k+1])
        # (I-theta*tau*L) v^{k+1} = (I+(1-theta)*tau*L)v^k
        #                                + tau*((1-theta)g^k+theta*g^{k+1}).
        rhs = old + tau*((1-theta)*apply_operator(lower, diagonal, upper, old)
                         + (1-theta)*old_forcing + theta*new_forcing)
        u[k+1, start:n] = rhs if factors is None else solve_tridiagonal(factors, rhs)
        u[k+1, n] = problem.right(t[k+1])  # точное ГУ первого рода
        if boundary != "pde":
            k1, k2, g = boundary_relation(problem, h, boundary)
            u[k+1, 0] = k1*u[k+1, 1] + k2*u[k+1, 2] + g*problem.left(t[k+1])
        old_forcing = new_forcing
        if not np.all(np.isfinite(u[k+1])):
            raise FloatingPointError("Переполнение решения: уменьшите шаг по времени.")
    return x, t, u


def max_error(problem, x, t, u):
    return np.max(np.abs(u - problem.exact(x[None, :], t[:, None])), axis=1)


def save_plot(figure, output, name):
    figure.tight_layout()
    figure.savefig(output/name, dpi=160)


def comparison_plots(problem, n, final_time, steps, sigma, output):
    figure, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for axis, boundary in zip(axes, BOUNDARIES):
        for scheme in SCHEMES:
            x, t, u = solve(problem, n, final_time, steps, sigma, scheme, boundary)
            error = max_error(problem, x, t, u)
            axis.plot(t, error, label=SCHEME_NAMES[scheme])
            print(f"{scheme:8s} {boundary:5s}: delta(T)={error[-1]:.6e}, max delta={error.max():.6e}")
        axis.set(title=BC_NAMES[boundary], xlabel="t")
        axis.grid(True)
        axis.legend()
    axes[0].set_ylabel("Максимальная ошибка δ(t)")
    save_plot(figure, output, "comparison.png")


def convergence_study(problem, final_time, sigma, sizes, output):
    """tau пропорционально h²; при удвоении N число шагов увеличивается в 4 раза."""
    base = sizes[0]
    if len(sizes) < 3 or base < 3 or sigma <= 0 or any(n % base for n in sizes):
        raise ValueError("Сетки сгущения должны быть целыми кратными первой.")
    base_steps = int(np.ceil(final_time*problem.a/(sigma*(problem.length/base)**2)))
    figure, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    error_figure, error_axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    rows = []
    for axis, error_axis, boundary in zip(axes, error_axes, BOUNDARIES):
        for scheme in SCHEMES:
            errors = []
            hs = problem.length/np.asarray(sizes, dtype=float)
            for n in sizes:
                steps = base_steps*(n//base)**2
                x, t, u = solve(problem, n, final_time, steps, sigma, scheme, boundary)
                error = max_error(problem, x, t, u)[-1]
                errors.append(error)
                rows.append((scheme, boundary, n, problem.length/n, final_time/steps,
                             problem.a*(final_time/steps)/(problem.length/n)**2, error))
            slope = np.polyfit(np.log(hs[-3:]), np.log(errors[-3:]), 1)[0]
            print(f"Порядок {scheme:8s}/{boundary:5s}: p={slope:.3f}; epsilon={errors[-1]:.6e}")
            axis.plot(np.log(hs), np.log(errors), "o-",
                      label=f"{SCHEME_NAMES[scheme]}, p={slope:.2f}")
            error_axis.plot(hs, errors, "o-", label=SCHEME_NAMES[scheme])
        axis.set(title=BC_NAMES[boundary], xlabel="ln h")
        axis.grid(True)
        axis.legend(fontsize=8)
        error_axis.set(title=BC_NAMES[boundary], xlabel="h")
        error_axis.grid(True)
        error_axis.legend(fontsize=8)
    axes[0].set_ylabel("ln ε,  ε = δ(T)")
    error_axes[0].set_ylabel("ε = δ(T)")
    save_plot(figure, output, "convergence.png")
    save_plot(error_figure, output, "error_vs_h.png")
    with (output/"convergence.csv").open("w", encoding="utf-8") as stream:
        stream.write("scheme,boundary,N,h,tau,sigma,error_T\n")
        for row in rows:
            stream.write(",".join(map(str, row))+"\n")
    return rows


def stability_experiment(problem, n, final_time, output):
    """Рост малого высокочастотного возмущения при sigma<1/2 и sigma>1/2."""
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    x = np.linspace(0, problem.length, n+1)
    perturbation = 1e-8*np.cos((n-0.5)*np.pi*x/problem.length)
    perturbation[-1] = 0
    for axis, boundary in zip(axes, BOUNDARIES):
        for sigma in (0.4, 0.6):
            steps = max(1, int(np.ceil(final_time*problem.a/(sigma*(problem.length/n)**2))))
            actual_sigma, radius, _ = explicit_stability(problem, n, final_time/steps, boundary)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, t, clean = solve(problem, n, final_time, steps, scheme="explicit",
                                    boundary=boundary, allow_unstable=True)
                _, _, perturbed = solve(problem, n, final_time, steps, scheme="explicit",
                                        boundary=boundary, allow_unstable=True,
                                        initial_values=problem.initial(x)+perturbation)
            difference = np.max(np.abs(perturbed-clean), axis=1)
            axis.semilogy(t, np.maximum(difference, 1e-16),
                          label=f"σ={actual_sigma:.3f}, ρ={radius:.3f}")
            print(f"Устойчивость {boundary:5s}: sigma={actual_sigma:.4f}, rho={radius:.6f}, "
                  f"возмущение {difference[0]:.2e} -> {difference[-1]:.2e}")
        axis.set(title=BC_NAMES[boundary], xlabel="t")
        axis.grid(True)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("Норма возмущения")
    save_plot(figure, output, "stability.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheme", choices=SCHEMES, default="cn")
    parser.add_argument("--bc", choices=BOUNDARIES, default="pde")
    parser.add_argument("--n", type=int, default=40, help="число пространственных интервалов")
    parser.add_argument("--time", type=float, default=1.0)
    parser.add_argument("--steps", type=int, help="число шагов времени, имеет приоритет над sigma")
    parser.add_argument("--sigma", type=float, default=0.4)
    for name, default in (("a", 1.0), ("c", -1.0), ("length", np.pi/2), ("alpha", 1.0),
                          ("beta", 0.0), ("left-amplitude", 1.0), ("right-amplitude", 1.0),
                          ("left-rate", None), ("right-rate", None)):
        parser.add_argument("--"+name, type=float, default=default)
    parser.add_argument("--study", action="store_true", help="все сравнения, сгущение и устойчивость")
    parser.add_argument("--allow-unstable", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent/"plots")
    args = parser.parse_args()
    problem = Problem(**{name: getattr(args, name) for name in Problem.__dataclass_fields__})
    x, t, u = solve(problem, args.n, args.time, args.steps, args.sigma,
                    args.scheme, args.bc, args.allow_unstable)
    args.output.mkdir(parents=True, exist_ok=True)
    sigma, radius, interior_ok = explicit_stability(problem, args.n, t[1], args.bc)
    print(f"h={x[1]:.8f}, tau={t[1]:.8f}, sigma={sigma:.6f}")
    print(f"Для явной схемы: rho={radius:.6f}, условие 2*sigma-c*tau<=1: {interior_ok}")
    exact_available = problem.has_exact()
    if not exact_available:
        print("ГУ изменены: аналитическое решение варианта 8 неприменимо; графики ошибок отключены.")
    figure, axis = plt.subplots(figsize=(8, 5))
    for index in np.unique(np.round(np.linspace(0, len(t)-1, 5)).astype(int)):
        line, = axis.plot(x, u[index], "o", markersize=3, label=f"Численное, t={t[index]:.3f}")
        if exact_available:
            fine_x = np.linspace(0, problem.length, 400)
            axis.plot(fine_x, problem.exact(fine_x, t[index]), color=line.get_color(), linewidth=1)
    axis.set(xlabel="x", ylabel="u(x,t)", title="Линии — аналитическое, точки — численное" if exact_available
             else "Численное решение")
    axis.grid(True)
    axis.legend(fontsize=8)
    save_plot(figure, args.output, "solution.png")
    if exact_available:
        error = max_error(problem, x, t, u)
        print(f"delta(T)={error[-1]:.6e}; max delta(t)={error.max():.6e}")
        figure, axis = plt.subplots(figsize=(8, 4))
        axis.plot(t, error)
        axis.set(xlabel="t", ylabel="δ(t)", title="Максимальная ошибка по пространству")
        axis.grid(True)
        save_plot(figure, args.output, "error.png")
        if args.study:
            comparison_plots(problem, args.n, args.time, args.steps, args.sigma, args.output)
            convergence_study(problem, args.time, args.sigma, [10, 20, 40, 80], args.output)
    if args.study:
        stability_experiment(problem, 40, 0.3, args.output)
    if args.show:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
