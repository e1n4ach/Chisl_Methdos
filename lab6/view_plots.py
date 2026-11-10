"""Панели Streamlit: только отображение рассчитанных данных, без чтения PNG."""

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
import pandas as pd

from analysis import method_name
from boundaries import BOUNDARY_NAMES

PANEL_TITLES = {
    "first": "Две точки · 1-й порядок",
    "three": "Три точки · 2-й порядок",
    "pde": "Две точки с уравнением · 2-й порядок",
}
METHOD_STYLES = {
    ("explicit", 1): ("#2563eb", "--", "o"),
    ("explicit", 2): ("#ea580c", "-", "s"),
    ("implicit", 1): ("#059669", "--", "o"),
    ("implicit", 2): ("#9333ea", "-", "s"),
}


def short_method(scheme: str, order: int) -> str:
    return f"{'Крест' if scheme == 'explicit' else 'Неявная'}, старт {order}"


def finish_axis(axis, xlabel: str, ylabel: str) -> None:
    axis.set_xlabel(xlabel, fontsize=12)
    axis.set_ylabel(ylabel, fontsize=12)
    axis.grid(True, which="major", alpha=0.25)
    axis.tick_params(labelsize=10)
    if axis.lines:
        axis.legend(fontsize=11, framealpha=0.9)
    else:
        axis.text(0.5, 0.5, "Нет рассчитанных комбинаций", ha="center", va="center",
                  transform=axis.transAxes, color="#64748b")


def comparison_figure(histories: pd.DataFrame) -> Figure:
    """Три независимых масштаба по y: ошибки разных ГУ сильно различаются."""
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    for axis, boundary in zip(axes, BOUNDARY_NAMES):
        axis.set_title(PANEL_TITLES[boundary], fontsize=12, pad=12)
        for (scheme, order), (color, style, _) in METHOD_STYLES.items():
            column = method_name(scheme, order, boundary)
            if column in histories:
                axis.plot(histories.index, histories[column], color=color, linestyle=style,
                          linewidth=2, label=short_method(scheme, order))
        finish_axis(axis, "Время t", "Максимальная ошибка ε(t)")
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useMathText=True)
    return figure


def convergence_figure(table: pd.DataFrame, direction: str) -> Figure:
    """Сверху ошибки, снизу измеренные порядки; по четыре метода на панель."""
    figure, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    step = "h" if direction == "h" else "τ"
    error = "ε(T)" if direction == "h" else "Разность с временным эталоном"
    order_column = "p(T)" if direction == "h" else "p(время)"
    for column, boundary in enumerate(BOUNDARY_NAMES):
        rows = table[table["boundary"] == boundary]
        axes[0, column].set_title(PANEL_TITLES[boundary], fontsize=12, pad=12)
        for (scheme, order), (color, style, marker) in METHOD_STYLES.items():
            group = rows[(rows["scheme"] == scheme) & (rows["initial_order"] == order)]
            if group.empty:
                continue
            label = short_method(scheme, order)
            axes[0, column].loglog(group[step], group[error], color=color, linestyle=style,
                                   marker=marker, linewidth=2, markersize=5, label=label)
            measured = group.dropna(subset=[order_column])
            axes[1, column].semilogx(measured[step], measured[order_column], color=color,
                                     linestyle=style, marker=marker, linewidth=2,
                                     markersize=5, label=label)
        for row, ylabel in enumerate(("ε(T) относительно U" if direction == "h"
                                     else "maxᵢ |uᵢ(T) − uᵢ,эталон(T)|", "Измеренный порядок p")):
            axis = axes[row, column]
            finish_axis(axis, f"Шаг {step} (логарифмическая шкала)", ylabel)
            if not rows.empty:
                axis.xaxis.set_major_locator(FixedLocator(sorted(rows[step].unique())))
                axis.xaxis.set_major_formatter(FuncFormatter(lambda value, position: f"{value:.1e}"))
                axis.xaxis.set_minor_formatter(NullFormatter())
                axis.tick_params(axis="x", labelsize=9)
    return figure
