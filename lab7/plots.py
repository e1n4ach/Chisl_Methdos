"""Общие графики для Streamlit и пакетного запуска."""

import numpy as np
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from lab7.analysis import AXIS_NAMES
from lab7.boundaries import BOUNDARY_NAMES
from lab7.solver import METHOD_NAMES, Config, Solution


def settings(config: Config, *, grid: bool = True, method: bool = True) -> str:
    parts = [BOUNDARY_NAMES[config.boundary], f"ε={config.epsilon:g}, max_iter={config.max_iter}"]
    if method:
        parts.insert(0, METHOD_NAMES[config.method])
    if config.method == "sor" or not method:
        parts.append(f"ω(SOR)={config.omega:g}")
    if grid:
        parts.append(f"Nx={config.nx}, Ny={config.ny} · hx={config.hx:.4g}, hy={config.hy:.4g}")
    return " · ".join(parts)


def figure_with_title(title: str, parameters: str, height: float = 4.3) -> Figure:
    fig = Figure(figsize=(11, height), layout="constrained")
    fig.suptitle(f"{title}\n{parameters}", fontsize=11)
    return fig


def finish(axis, xlabel: str, ylabel: str) -> None:
    axis.set(xlabel=xlabel, ylabel=ylabel)
    axis.legend(fontsize=8)
    axis.grid(True, alpha=0.25)


def positive(values):
    values = np.asarray(values)
    return np.where((values > 0) & np.isfinite(values), values, np.nan)


def solution_figures(result: Solution, i: int, j: int) -> dict[str, Figure]:
    x, y = np.meshgrid(result.x, result.y)
    error = result.absolute_error
    parameters = settings(result.config)
    surfaces = figure_with_title("Решение задачи Лапласа", parameters, 5.1)
    numerical = np.ma.masked_invalid(result.values)
    norm = Normalize(min(float(result.exact.min()), float(numerical.min())),
                     max(float(result.exact.max()), float(numerical.max())))
    for index, (values, label, cmap, color, zlabel) in enumerate((
        (numerical, "Численное решение", "viridis", "#21918c", "u(x,y)"),
        (result.exact, "Точное решение", "plasma", "#cc4778", "U(x,y)"),
    ), 1):
        axis = surfaces.add_subplot(1, 2, index, projection="3d")
        axis.plot_surface(x, y, values, cmap=cmap, norm=norm)
        axis.set(xlabel="x", ylabel="y", zlabel=zlabel, zlim=(norm.vmin, norm.vmax))
        axis.legend(handles=[Patch(facecolor=color, label=label)], loc="upper left", fontsize=8)

    heatmap = figure_with_title("Абсолютная ошибка на сетке", parameters)
    axis = heatmap.subplots()
    mesh = axis.pcolormesh(x, y, error, shading="nearest", cmap="magma")
    axis.axvline(result.x[i], color="cyan", ls=":", label=f"x={result.x[i]:.5g}")
    axis.axhline(result.y[j], color="white", ls="--", label=f"y={result.y[j]:.5g}")
    axis.set(xlabel="x", ylabel="y", xlim=(result.x[0], result.x[-1]),
             ylim=(result.y[0], result.y[-1]))
    axis.legend(loc="upper right", fontsize=8)
    heatmap.colorbar(mesh, ax=axis, label="|u−U|")

    sections = figure_with_title("Сечения в узлах сетки", parameters, 6)
    axes = sections.subplots(2, 2, sharex="col")
    for column, (coords, cut, coordinate, value, xlabel) in enumerate((
        (result.x, np.s_[j, :], "y", result.y[j], "x"),
        (result.y, np.s_[:, i], "x", result.x[i], "y"),
    )):
        axis, error_axis = axes[:, column]
        axis.set_title(f"{coordinate}={value:.6g}")
        axis.plot(coords, result.values[cut], ".-", label="Численное u")
        axis.plot(coords, result.exact[cut], "--", label="Точное U")
        error_axis.plot(coords, error[cut], ".-", color="tab:red", label="Абсолютная ошибка")
        finish(axis, xlabel, "u, U")
        finish(error_axis, xlabel, "|u−U|")

    history = figure_with_title("История итераций", parameters)
    axes = history.subplots(1, 2)
    axes[0].semilogy(positive(result.error_history), label=r"$E_k=\|u^k-U\|_\infty$")
    axes[1].semilogy(positive(result.residual_history), label=r"$\|b-Au^k\|_\infty$")
    axes[1].axhline(result.config.epsilon, color="tab:red", ls="--", label="Порог ε")
    finish(axes[0], "Итерация k", "Максимальная ошибка Eₖ")
    finish(axes[1], "Итерация k", "Норма невязки")
    return {"surfaces": surfaces, "error_map": heatmap, "sections": sections, "history": history}


def comparison_figure(results: tuple[Solution, ...]) -> Figure:
    config = results[0].config
    fig = figure_with_title("Сравнение методов на одной сетке", settings(config, method=False), 7)
    error_axis, residual_axis = fig.subplots(2, 1)
    for result in results:
        label = f"{METHOD_NAMES[result.config.method]} · k={result.iterations}"
        if result.status != "converged":
            label += " (порог не достигнут)"
        error_axis.semilogy(positive(result.error_history), label=label)
        residual_axis.semilogy(positive(result.residual_history), label=label)
    residual_axis.axhline(config.epsilon, color="gray", ls="--", label="Порог ε")
    finish(error_axis, "Итерация k", "Максимальная ошибка Eₖ")
    finish(residual_axis, "Итерация k", "Норма невязки")
    return fig


def iterations_figure(results: tuple[Solution, ...]) -> Figure:
    fig = figure_with_title("Число итераций", settings(results[0].config, method=False))
    axis = fig.subplots()
    labels = [METHOD_NAMES[result.config.method] for result in results]
    counts = [result.iterations for result in results]
    colors = ["tab:blue" if result.status == "converged" else "tab:orange" for result in results]
    bars = axis.bar(labels, counts, color=colors)
    axis.bar_label(bars, padding=3)
    axis.margins(y=0.17)
    axis.set(xlabel="Метод", ylabel="Число итераций")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(handles=[Patch(facecolor=color, label=label) for color, label in (
        ("tab:blue", "Порог достигнут"), ("tab:orange", "Порог не достигнут"),
    )], fontsize=8)
    return fig


def comparison_error_figure(results: tuple[Solution, ...]) -> Figure:
    config = results[0].config
    fig = figure_with_title("Абсолютная ошибка методов в общей цветовой шкале",
                            settings(config, method=False))
    axes = fig.subplots(1, len(results), sharex=True, sharey=True)
    errors = [np.ma.masked_invalid(result.absolute_error) for result in results]
    limit = max(float(error.max()) for error in errors)
    norm = Normalize(0, limit)
    for axis, result, error in zip(axes, results, errors):
        mesh = axis.pcolormesh(result.x, result.y, error, cmap="magma", norm=norm,
                              shading="nearest")
        label = METHOD_NAMES[result.config.method]
        if result.status != "converged":
            label += "\nПорог не достигнут"
        axis.set(title=label, xlabel="x", ylabel="y", xlim=(result.x[0], result.x[-1]),
                 ylim=(result.y[0], result.y[-1]))
    fig.colorbar(mesh, ax=axes, label="|u−U|")
    return fig


def comparison_figures(results: tuple[Solution, ...]) -> dict[str, Figure]:
    return {"comparison": comparison_figure(results), "iterations": iterations_figure(results),
            "comparison_errors": comparison_error_figure(results)}


def convergence_figure(table, config: Config, direction: str) -> Figure:
    first = table.iloc[0]
    if direction == "both":
        fixed = f"Ny/Nx={first['Ny']/first['Nx']:.3g}, сгущение 2×"
        xlabel = "hx (hy уменьшается в той же пропорции)"
    else:
        fixed_column = "Ny" if direction == "hx" else "Nx"
        fixed_step = "hy" if direction == "hx" else "hx"
        fixed = f"{fixed_column}={first[fixed_column]}, {fixed_step}={first[fixed_step]:.4g}"
        xlabel = direction
    fig = figure_with_title(
        f"Сгущение сетки · {AXIS_NAMES[direction]} · {fixed}", settings(config, grid=False), 4.8
    )
    error_axis, order_axis = fig.subplots(1, 2)
    error_axis.loglog(table["h"], positive(table["E"]), "o-", label="E к точному U")
    order_axis.semilogx(table["h"], table["p"], "o-", label="p по E")
    if "E_ref" in table:
        error_axis.loglog(table["h"], positive(table["E_ref"]), "s-", label="E_ref к сеточному эталону")
        order_axis.semilogx(table["h"], table["p_ref"], "s-", label="p_ref по E_ref")
    order_axis.axhline(2, color="gray", ls="--", label="Второй порядок")
    finish(error_axis, xlabel, "Максимальная ошибка")
    finish(order_axis, xlabel, "Наблюдаемый порядок p")
    return fig
