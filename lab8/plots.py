"""Графики из численных массивов и таблиц расчёта."""

import numpy as np
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from lab8.analysis import AXIS_NAMES
from lab8.solver import METHOD_NAMES

COLORS = {"adi": "tab:blue", "fractional": "tab:orange"}
SHORT_NAMES = {"adi": "ADI", "fractional": "Дробные шаги"}


def settings(config, *, method=True):
    prefix = f"{METHOD_NAMES[config.method]} · " if method else ""
    return (f"{prefix}Nx={config.nx}, Ny={config.ny}, M={config.steps}, T={config.final_time:g}"
            f" · hx={config.hx:.4g}, hy={config.hy:.4g}, τ={config.tau:.4g}")


def figure(title, subtitle, height=4.2):
    fig = Figure(figsize=(11, height), layout="constrained")
    fig.suptitle(f"{title}\n{subtitle}", fontsize=11)
    return fig


def finish(axis, xlabel, ylabel):
    axis.set(xlabel=xlabel, ylabel=ylabel)
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8)


def solution_figures(result, tables, k, i, j):
    config = result.config
    t = result.time[k]
    x, y = result.problem.x, result.problem.y
    values, exact = result.values[k], result.problem.exact(t)
    xx, yy = np.meshgrid(x, y)
    subtitle = f"{settings(config)} · k={k}, t={t:.5g}"

    surfaces = figure("Численное и аналитическое решения", subtitle, 5.1)
    low, high = min(values.min(), exact.min()), max(values.max(), exact.max())
    if low == high:
        low, high = low-1e-15, high+1e-15
    norm = Normalize(low, high)
    for position, (z, label, symbol) in enumerate((
        (values, "Численное решение", "u"), (exact, "Точное решение", "U"),
    ), start=1):
        axis = surfaces.add_subplot(1, 2, position, projection="3d")
        axis.plot_surface(xx, yy, z, cmap="viridis", norm=norm)
        axis.set(xlabel="x", ylabel="y", zlabel=f"{symbol}(x,y,t)", zlim=(low, high))
        axis.legend(handles=[Patch(facecolor="#21918c", label=label)], loc="upper left", fontsize=8)

    heatmap = figure("Абсолютная ошибка в выбранный момент", subtitle)
    axis = heatmap.subplots()
    mesh = axis.pcolormesh(x, y, np.abs(values-exact), cmap="magma", shading="nearest")
    axis.axvline(x[i], color="cyan", ls=":", label=f"Сечение x={x[i]:.4g}")
    axis.axhline(y[j], color="white", ls="--", label=f"Сечение y={y[j]:.4g}")
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="x", ylabel="y", aspect="equal")
    axis.legend(fontsize=8)
    heatmap.colorbar(mesh, ax=axis, label="|u−U|")

    sections = figure("Сечения через узлы сетки", subtitle, 6.0)
    axes = sections.subplots(2, 2, sharex="col")
    for col, (name, xlabel, label) in enumerate((
        ("section_y", "x", f"y = {y[j]:.5g}"), ("section_x", "y", f"x = {x[i]:.5g}"),
    )):
        section = tables[name]
        ax, error_ax = axes[:, col]
        ax.set_title(label)
        ax.plot(section[xlabel], section["u"], ".-", label="Численное u")
        ax.plot(section[xlabel], section["U"], "--", label="Точное U")
        error_ax.plot(section[xlabel], section["|u−U|"], ".-", color="tab:red", label="Абсолютная ошибка")
        ax.ticklabel_format(axis="y", useOffset=False)
        finish(ax, xlabel, "u, U")
        finish(error_ax, xlabel, "|u−U|")

    history = figure("Максимальная ошибка по пространству", settings(config))
    axis = history.subplots()
    errors = tables["error_time"]
    axis.plot(errors["t"], errors["E_max(t)"], label="Максимум по всем узлам")
    axis.axvline(t, color="gray", ls=":", label=f"Выбранный слой k={k}, t={t:.5g}")
    finish(axis, "t", "E_max(t)")

    points = figure("Решение в фиксированных точках", settings(config), 5.8)
    ax, err_ax = points.subplots(2, 1, sharex=True)
    for (px, py), group in tables["point_history"].groupby(["x", "y"], sort=False):
        label = f"({px:.4g}; {py:.4g})"
        line, = ax.plot(group["t"], group["u"], label=f"u {label}")
        ax.plot(group["t"], group["U"], "--", label=f"U {label}", color=line.get_color())
        err_ax.plot(group["t"], group["|u−U|"], label=label, color=line.get_color())
    finish(ax, "t", "u, U")
    finish(err_ax, "t", "|u−U|")
    return {"surfaces": surfaces, "error_map": heatmap, "sections": sections,
            "error_time": history, "point_history": points}


def comparison_figures(comparison, config):
    rows = comparison.summary
    fig = figure("Точность на одной сетке", settings(config, method=False))
    axis = fig.subplots()
    for method, name in METHOD_NAMES.items():
        group = comparison.histories[comparison.histories["Метод"] == name]
        axis.plot(group["t"], group["E_max(t)"], color=COLORS[method], label=SHORT_NAMES[method])
    axis.set_yscale("symlog", linthresh=max(rows["max_t E_max"].min()/10, np.finfo(float).eps))
    finish(axis, "t", "E_max(t)")

    timing = figure("Время расчёта со сборкой и записью слоёв",
                    f"{settings(config, method=False)} · повторов на метод: {rows['Повторы'].iloc[0]}")
    axis = timing.subplots()
    median = rows["Медиана, с"]
    axis.bar(list(SHORT_NAMES.values()), median, color=list(COLORS.values()), label="Медиана")
    axis.errorbar(list(SHORT_NAMES.values()), median,
                  yerr=[median-rows["Минимум, с"], rows["Максимум, с"]-median],
                  fmt="none", ecolor="black", capsize=4, label="Минимум и максимум")
    finish(axis, "Метод", "Время, с")
    return {"comparison": fig, "comparison_timings": timing}


def convergence_figure(table, direction):
    row = table.iloc[0]
    if direction == "hx":
        fixed = f"Ny={row['Ny']}, hy={row['hy']:.4g}, M={row['M']}, τ={row['τ']:.4g}"
    elif direction == "hy":
        fixed = f"Nx={row['Nx']}, hx={row['hx']:.4g}, M={row['M']}, τ={row['τ']:.4g}"
    else:
        fixed = f"Nx={row['Nx']}, Ny={row['Ny']}, hx={row['hx']:.4g}, hy={row['hy']:.4g}"
    label = "τ" if direction == "tau" else direction
    fig = figure(f"Сгущение · {AXIS_NAMES[direction]}",
                 f"{fixed} · T={row['T']:g} · вариант 7 и контрольная задача" if direction != "tau"
                 else f"{fixed} · T={row['T']:g} · вариант 7", 4.9)
    error_axis, order_axis = fig.subplots(1, 2, sharex=True)
    for method, name in METHOD_NAMES.items():
        data = table[table["Метод"] == name]
        color, short = COLORS[method], SHORT_NAMES[method]
        error_axis.loglog(data["s"], data["E(T)"].where(data["E(T)"] > 0),
                          "o-", label=f"{short} · вариант 7", color=color)
        order_axis.semilogx(data["s"], data["p"], "o-", label=f"{short} · вариант 7", color=color)
        if "p_ref" in data:
            error_axis.loglog(data["s"], data["E_ref(T)"].where(data["E_ref(T)"] > 0),
                              "s--", label=f"{short} · контроль к эталону", color=color)
            order_axis.semilogx(data["s"], data["p_ref"], "s--",
                                label=f"{short} · контроль к эталону", color=color)
    finish(error_axis, label, "Максимальная ошибка при T")
    finish(order_axis, label, "Наблюдаемый порядок p")
    return fig
