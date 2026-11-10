"""Воспроизводимые графики по фактическим таблицам checks.py."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
import pandas as pd

from analysis import method_name
from boundaries import BOUNDARY_NAMES


def short_name(scheme: str, order: int) -> str:
    return f"{'Крест' if scheme == 'explicit' else 'Неявная'}, старт {order}"


def save_report_plots(histories: pd.DataFrame, space_table: pd.DataFrame,
                      time_table: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for column, (boundary, name) in enumerate(BOUNDARY_NAMES.items()):
        rows = time_table[time_table["boundary"] == boundary]
        for (scheme, order), group in rows.groupby(["scheme", "initial_order"], sort=False):
            label = short_name(scheme, order)
            style = "o--" if order == 1 else "s-"
            axes[0, column].loglog(group["τ"], group["Разность с временным эталоном"], style, label=label)
            axes[1, column].semilogx(group["τ"], group["p(время)"], style, label=label)
        axes[0, column].set(title=name, xlabel="τ", ylabel="max |u(T) − u эталон(T)|")
        axes[1, column].set(xlabel="τ", ylabel="Измеренный p по соседним сеткам")
        for axis in axes[:, column]:
            axis.xaxis.set_major_locator(FixedLocator(sorted(rows["τ"].unique())))
            axis.xaxis.set_major_formatter(FuncFormatter(lambda value, position: f"{value:.1e}"))
            axis.xaxis.set_minor_formatter(NullFormatter())
            axis.tick_params(axis="x", labelsize=8)
            axis.grid(True, which="both", alpha=0.3)
            axis.legend(fontsize=9)
    figure.suptitle("Сходимость по времени: фиксировано N=60, T=1; шаги указаны в convergence_tau.csv")
    figure.savefig(output / "convergence_tau.png", dpi=150)
    plt.close(figure)

    for kind, ylabel in (("h", "ε(T) относительно U"), ("comparison", "ε(t) относительно U")):
        figure, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
        for axis, (boundary, name) in zip(axes, BOUNDARY_NAMES.items()):
            for scheme in ("explicit", "implicit"):
                for order in (1, 2):
                    label = short_name(scheme, order)
                    if kind == "h":
                        group = space_table[(space_table["boundary"] == boundary)
                                            & (space_table["scheme"] == scheme)
                                            & (space_table["initial_order"] == order)]
                        axis.loglog(group["h"], group["ε(T)"], "o--" if order == 1 else "s-", label=label)
                    else:
                        axis.plot(histories.index, histories[method_name(scheme, order, boundary)],
                                  "--" if order == 1 else "-", label=label)
            axis.set(title=name, xlabel="h" if kind == "h" else "t", ylabel=ylabel)
            if kind == "h":
                axis.xaxis.set_major_locator(FixedLocator(sorted(group["h"].unique())))
                axis.xaxis.set_major_formatter(FuncFormatter(lambda value, position: f"{value:.3f}"))
                axis.xaxis.set_minor_formatter(NullFormatter())
            axis.grid(True, which="both", alpha=0.3)
            axis.legend(fontsize=9)
        figure.suptitle("Сгущение по h при M=20000, T=1" if kind == "h" else "Все 12 комбинаций: N=40, M=200, T=1")
        figure.savefig(output / ("convergence_h.png" if kind == "h" else "comparison.png"), dpi=150)
        plt.close(figure)
