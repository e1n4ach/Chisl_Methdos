"""Streamlit-интерфейс для расчёта и исследования ЛР7."""

import streamlit as st

from lab7.analysis import (
    AXIS_NAMES, compare, comparison_tables, convergence, result_tables,
)
from lab7.boundaries import BOUNDARY_NAMES
from lab7.export import RESULTS_DIR, csv_bytes, save
from lab7.plots import comparison_figures, convergence_figure, settings, solution_figures
from lab7.solver import METHOD_NAMES, STATUS_NAMES, Config, solve


@st.cache_data(show_spinner=False, max_entries=16)
def calculate(config):
    return solve(config)


@st.cache_data(show_spinner=False, max_entries=8)
def calculate_comparison(config):
    return compare(config)


@st.cache_data(show_spinner=False, max_entries=16)
def calculate_convergence(config, direction, base, fixed, levels):
    return convergence(config, direction, base, fixed, levels)


def sidebar_config() -> Config:
    defaults = Config()
    with st.sidebar:
        st.header("Параметры расчёта")
        with st.form("parameters", border=False):
            method = st.selectbox("Метод", list(METHOD_NAMES), index=2, format_func=METHOD_NAMES.get, key="method")
            boundary = st.selectbox("Граничные условия Неймана", list(BOUNDARY_NAMES),
                                    format_func=BOUNDARY_NAMES.get, key="boundary")
            nx = st.number_input("Nx — интервалы по x", 2, 120, defaults.nx, key="nx")
            ny = st.number_input("Ny — интервалы по y", 3, 120, defaults.ny, key="ny")
            epsilon = st.number_input("ε — порог нормы невязки", min_value=1e-12,
                                      value=defaults.epsilon, format="%.1e", key="epsilon")
            omega = st.number_input("ω для SOR", 1.0, 1.999, defaults.omega, step=0.01,
                                    format="%.3f", key="omega")
            max_iter = st.number_input("max_iter — предел итераций", 1, 200_000,
                                       defaults.max_iter, step=1000, key="max_iter")
            st.caption("При ω=1 SOR совпадает с методом Зейделя.")
            if st.form_submit_button("Рассчитать"):
                st.session_state["config"] = Config(nx, ny, method, boundary, epsilon, omega, max_iter)
    return st.session_state.get("config", defaults)


def show_table(table, name, label="Скачать CSV"):
    numeric = ("E", "E_ref", "||r||∞", "Оценка итерационной ошибки", "Разность эталонов",
               "Норма невязки", "|u−U|", "ε")
    columns = {column: st.column_config.NumberColumn(format="%.4e") for column in numeric if column in table}
    st.dataframe(table, hide_index=True, width="stretch", column_config=columns)
    st.download_button(label, csv_bytes(table), file_name=f"{name}.csv", mime="text/csv",
                       key=f"download_{name}", on_click="ignore")


def save_button(group, tables, figures):
    if st.button("Сохранить PNG и CSV", key=f"save_{group}"):
        try:
            save(RESULTS_DIR / "ui", tables, figures)
            st.success("Файлы сохранены в lab7/results/ui.")
        except OSError as exc:
            st.error(f"Ошибка записи файлов ({exc}).")


def show_solution(result):
    config = result.config
    if result.status != "converged":
        st.warning(f"{STATUS_NAMES[result.status]}. Графики показывают последнее приближение.")
    columns = st.columns(5)
    for column, (label, value) in zip(columns, (
        ("hx", f"{config.hx:.6g}"), ("hy", f"{config.hy:.6g}"),
        ("Максимальная ошибка E", f"{result.error_history[-1]:.5e}"),
        ("Норма невязки", f"{result.residual_history[-1]:.5e}"),
        ("Итерации", result.iterations),
    )):
        column.metric(label, value)
    st.caption(settings(config))
    left, right = st.columns(2)
    # Ключи зависят от размера, чтобы после смены сетки индекс оставался допустимым.
    i = left.slider("Индекс i для сечения x = const", 0, config.nx, config.nx//2, key=f"section_i_{config.nx}")
    j = right.slider("Индекс j для сечения y = const", 0, config.ny, config.ny//2, key=f"section_j_{config.ny}")
    figures = solution_figures(result, i, j)
    for figure in figures.values():
        st.pyplot(figure, width="stretch")

    tables = result_tables(result, i, j)
    st.subheader("Таблицы расчёта")
    show_table(tables["summary"], "summary", "Скачать сводку CSV")
    for label, name in (
        ("Значения во всех узлах", "values"), ("Сечение y = const", "section_y"),
        ("Сечение x = const", "section_x"), ("История итераций", "history"),
        ("Невязки граничных условий", "boundaries"),
    ):
        with st.expander(label):
            show_table(tables[name], name)
    save_button("solution", tables, figures)


def show_comparison(config):
    st.caption("Все три метода стартуют с одинакового приближения и используют одну сетку, "
               "одну схему границы, общий порог невязки и общий предел итераций.")
    if st.button("Сравнить методы", key="compare"):
        with st.spinner("Расчёт трёх методов…"):
            st.session_state["comparison"] = (config, calculate_comparison(config))
    saved = st.session_state.get("comparison")
    if saved is None or saved[0] != config:
        return
    results = saved[1]
    if any(result.status != "converged" for result in results):
        st.warning("Для некоторых методов порог ε не достигнут. Статусы приведены в таблице.")
    figures = comparison_figures(results)
    tables = comparison_tables(results)
    for figure in figures.values():
        st.pyplot(figure, width="stretch")
    show_table(tables["comparison"], "comparison", "Скачать сравнение CSV")
    with st.expander("Истории трёх методов"):
        show_table(tables["comparison_history"], "comparison_history")
    save_button("comparison", tables, figures)


def show_convergence(config):
    st.caption("Число интервалов по сгущаемой оси удваивается на каждой следующей сетке.")
    direction = st.radio("Направление сгущения", list(AXIS_NAMES), format_func=AXIS_NAMES.get,
                         horizontal=True, key="direction")
    left, middle, right = st.columns(3)
    base_label = {"hx": "Начальное Nx", "hy": "Начальное Ny", "both": "Начальное Nx"}
    fixed_label = {"hx": "Фиксированное Ny", "hy": "Фиксированное Nx", "both": "Начальное Ny"}
    base = left.number_input(base_label[direction], 3, 16, 6, key=f"base_{direction}")
    fixed = middle.number_input(fixed_label[direction], 3, 48, 8 if direction == "both" else 24,
                                key=f"fixed_{direction}")
    levels = right.number_input("Количество сеток", 2, 4, 3, key="levels")
    if direction != "both":
        st.caption("E измеряется к аналитическому U, E_ref — к сеточному эталону при том же "
                   "фиксированном шаге. Сравнение эталонов проверяет точность E_ref.")
    st.latex(r"p=\frac{\log(E_{\mathrm{coarse}}/E_{\mathrm{fine}})}"
             r"{\log(h_{\mathrm{coarse}}/h_{\mathrm{fine}})}")
    key = f"convergence_{direction}"
    signature = (config, direction, base, fixed, levels)
    if st.button("Провести исследование", key="research"):
        with st.spinner("Расчёт серии сеток…"):
            st.session_state[key] = (signature, calculate_convergence(*signature))
    saved = st.session_state.get(key)
    if saved is None or saved[0] != signature:
        return

    table = saved[1]
    if (table["Статус"] != STATUS_NAMES["converged"]).any():
        st.warning("Порядок вычислен для пар сеток, на которых достигнут порог ε.")
    errors = table["E"]
    if "E_ref" in table:
        errors = errors.combine(table["E_ref"], min)
        if (table["Разность эталонов"] > 0.05*table["E_ref"]).any():
            st.warning("Разность эталонов превышает 5% от E_ref на части сеток.")
    if (table["Оценка итерационной ошибки"] > 0.05*errors).any():
        st.warning("Оценка вклада итераций превышает 5% от ошибки на части сеток. "
                   "Для оценки порядка уменьшите ε и при необходимости увеличьте max_iter.")
    figures = {key: convergence_figure(table, config, direction)}
    st.pyplot(figures[key], width="stretch")
    show_table(table, key, "Скачать серию CSV")
    save_button(key, {key: table}, figures)


def main():
    st.set_page_config(page_title="Лабораторная 7 — вариант 3", layout="wide")
    st.title("Лабораторная 7 · Вариант 3")
    st.latex(r"u_{xx}+u_{yy}=0,\qquad 0\le x\le 1,\quad 0\le y\le\pi/2")
    st.latex(r"u(0,y)=\cos y,\quad u(1,y)=e\cos y,\quad "
             r"u_y(x,0)=0,\quad u_y(x,\pi/2)=-e^x,\quad U(x,y)=e^x\cos y")
    config = sidebar_config()
    solution_tab, comparison_tab, study_tab = st.tabs(["Решение", "Сравнение методов", "Сгущение сетки"])
    with solution_tab:
        with st.spinner("Расчёт решения…"):
            result = calculate(config)
        show_solution(result)
        with st.expander("Разностная схема и остановка"):
            st.latex(r"\frac{u_{i-1,j}-2u_{i,j}+u_{i+1,j}}{h_x^2}+"
                     r"\frac{u_{i,j-1}-2u_{i,j}+u_{i,j+1}}{h_y^2}=0")
            st.latex(r"\|b-Au^k\|_\infty\le\varepsilon,\qquad E_k=\|u^k-U\|_\infty")
            st.markdown("Норма невязки проверяется после каждого прохода. Углы задаются из условий "
                        "Дирихле. Вывод формул для обеих схем приведён в lab7/README.md.")
    with comparison_tab:
        show_comparison(config)
    with study_tab:
        show_convergence(config)


if __name__ == "__main__":
    main()
