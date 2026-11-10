"""Интерфейс по структуре ЛР5: параметры, срезы, ошибки, отдельные исследования."""

from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from analysis import COMBINATIONS, compare, convergence, max_errors, method_name
from boundaries import BOUNDARY_NAMES
from problem import exact_solution
from solver import Config, SCHEME_NAMES, START_NAMES, solve, stability, steps_for_tau
from view_plots import comparison_figure, convergence_figure, short_method

CACHE_VERSION = "discrete-start-v2"


@st.cache_data(show_spinner=False, max_entries=24)
def calculate(config: Config, version: str):
    """version отделяет результаты до и после изменения численного алгоритма."""
    return solve(config)


@st.cache_data(show_spinner=False, max_entries=8)
def calculate_comparison(config: Config, combinations: tuple, version: str):
    bar = st.progress(0.0, text="Сравнение методов")
    try:
        return compare(config, combinations, lambda fraction, text: bar.progress(fraction, text=text))
    finally:
        bar.empty()


@st.cache_data(show_spinner=False, max_entries=8)
def calculate_research(final_time: float, direction: str, base: int, fixed: int, version: str):
    """Серии как в checks.py: один T, сгущение одного шага и все 12 методов."""
    tables, diagnostics = [], []
    bar = st.progress(0.0, text="Подготовка 12 серий сгущения")
    try:
        for index, (scheme, order, boundary) in enumerate(COMBINATIONS):
            config = Config(final_time=final_time, scheme=scheme, initial_order=order,
                            boundary=boundary)
            # Такое же уточнение переходного режима, как в checks.py.
            first_count = 4*base if direction == "tau" and (scheme, order, boundary) == ("implicit", 2, "first") else base
            grids = tuple(first_count*2**power for power in range(4))
            name = method_name(scheme, order, boundary)
            try:
                table = convergence(config, direction, grids, fixed,
                                    lambda fraction, text: bar.progress(
                                        (index+fraction)/len(COMBINATIONS),
                                        text=f"{index+1}/12 · {short_method(scheme, order)} · {BOUNDARY_NAMES[boundary]} · {text}"))
                attributes = dict(table.attrs)
                if direction == "tau":
                    attributes["reference_effect"] = attributes["reference_difference"] / max(
                        table["Разность с временным эталоном"].iloc[-1], 1e-300)
                table = table.assign(scheme=scheme, initial_order=order, boundary=boundary,
                                     Метод=short_method(scheme, order))
                table.attrs = {}
                tables.append(table)
                diagnostics.append({"Метод": name, "Статус": "OK", **attributes})
            except (ValueError, FloatingPointError) as exc:
                diagnostics.append({"Метод": name, "Статус": str(exc)})
            bar.progress((index+1)/len(COMBINATIONS), text=f"Завершено {index+1} из 12 серий")
    finally:
        bar.empty()
    return (pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(),
            pd.DataFrame(diagnostics))


def show_plot(x, series: dict, xlabel: str, ylabel: str, *, logarithmic: bool = False):
    figure, axis = plt.subplots(figsize=(8, 3.5))
    for name, y in series.items():
        axis.plot(x, y, label=name)
    if logarithmic:
        axis.set_xscale("log")
        axis.set_yscale("log")
    axis.set(xlabel=xlabel, ylabel=ylabel)
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    st.pyplot(figure)
    plt.close(figure)


def show_table(table: pd.DataFrame, key: str):
    st.dataframe(table, hide_index=True, width="stretch")
    buffer = StringIO()
    table.to_csv(buffer, index=False)
    st.download_button("Скачать таблицу CSV", buffer.getvalue().encode("utf-8-sig"),
                       file_name=f"{key}.csv", mime="text/csv", key=f"download_{key}")


def show_comparison(config: Config):
    st.subheader("Сравнение комбинаций")
    st.caption(f"Одна сетка для всех методов: N={config.n}, M={config.steps}, T={config.final_time:g}; "
               f"h={config.h:.6g}, τ={config.tau:.6g}. На каждой панели — свой способ границы, "
               "не более четырёх линий. Цвета и обозначения одинаковы на всех графиках.")
    labels = {method_name(*combination): combination for combination in COMBINATIONS}
    selected = st.multiselect("Комбинации для сравнения", list(labels),
                             default=[method_name("explicit", 2, "pde"), method_name("implicit", 2, "pde")])
    left, right = st.columns(2)
    selected_run = left.button("Сравнить выбранные", key="compare_selected")
    all_run = right.button("Сравнить все 12", key="compare_all")
    if selected_run or all_run:
        combinations = COMBINATIONS if all_run else tuple(labels[name] for name in selected)
        if not combinations:
            st.warning("Выберите хотя бы одну комбинацию.")
        else:
            st.session_state["comparison"] = (config, calculate_comparison(config, combinations, CACHE_VERSION), CACHE_VERSION)
    saved = st.session_state.get("comparison")
    if saved is not None and len(saved) == 3 and saved[0] == config and saved[2] == CACHE_VERSION:
        table, histories = saved[1]
        if not histories.empty:
            figure = comparison_figure(histories)
            st.pyplot(figure, width="stretch")
            plt.close(figure)
        if (table["Статус"] != "OK").any():
            st.warning("Некоторые комбинации не рассчитаны. Причины указаны в таблице.")
        show_table(table, "comparison")


def show_convergence(config: Config):
    st.subheader("Влияние h и τ по отдельности")
    st.caption("Это отдельные серии сгущения для всех 12 комбинаций, как в checks.py. "
               f"Используется текущее T={config.final_time:g}; N и M выбранного расчёта выше не задают сетки исследования. "
               "Сверху — ошибки, снизу — порядки, вычисленные по соседним строкам таблицы.")
    axis = st.radio("Исследуемый шаг", ["h", "tau"], format_func=lambda value: "Пространственный h" if value == "h" else "Временной τ", horizontal=True)
    if axis == "h":
        base = st.number_input("Начальное N для исследования", min_value=12, max_value=50, value=12, step=1)
        fixed = st.number_input("Фиксированное M", min_value=100, max_value=40000, value=20000, step=100)
        st.caption(f"N={base}, {2*base}, {4*base}, {8*base}; во всех четырёх расчётах M={fixed}, "
                   f"τ={config.final_time/fixed:.6g}. Ошибка ε(T) оценивается относительно аналитического U.")
    else:
        base = st.number_input("Начальное M для исследования", min_value=100, max_value=1000, value=160, step=10)
        fixed = st.number_input("Фиксированное N", min_value=10, max_value=80, value=60, step=10)
        st.caption(f"Для всех серий N={fixed}, h={1/fixed:.6g}; M={base}, {2*base}, {4*base}, {8*base}. "
                   f"Для неявной схемы со вторым стартом и первым ГУ: M={4*base}, {8*base}, {16*base}, {32*base}, как в checks.py.")
        st.info("Временной порядок оценивается относительно расчёта с очень малым τ при фиксированном N: "
                "Δτ = maxᵢ |uᵢ(T) − uᵢ,эталон(T)|. В эталоне M в 32 раза больше максимального M серии. "
                "Он проверяется сравнением с эталоном в 16 раз мельче. Ошибка относительно U сохраняется в таблицах: "
                "она может выйти на пространственный предел точности и не показывать порядок по времени.")
    signature = (config.final_time, axis, int(base), int(fixed), CACHE_VERSION)
    state_key = f"convergence_{axis}"
    if st.button("Провести исследование", key="research"):
        try:
            with st.spinner("12 серий сгущения и контроль фиксированного шага…"):
                result = calculate_research(config.final_time, axis, int(base), int(fixed), CACHE_VERSION)
            st.session_state[state_key] = (signature, result)
        except (ValueError, FloatingPointError) as exc:
            st.error(str(exc))
    saved = st.session_state.get(state_key)
    if saved is not None and saved[0] == signature:
        table, diagnostics = saved[1]
        if not table.empty:
            figure = convergence_figure(table, axis)
            st.pyplot(figure, width="stretch")
            plt.close(figure)
            order_column = "p(T)" if axis == "h" else "p(время)"
            st.markdown("**Измеренные порядки на последней паре сеток**")
            last = table.groupby(["boundary", "Метод"], sort=False).tail(1)
            summary = last.pivot(index="boundary", columns="Метод", values=order_column)
            summary = summary.reindex(BOUNDARY_NAMES).rename(index=BOUNDARY_NAMES)
            summary.index.name = "Граница"
            st.dataframe(summary.style.format("{:.3f}"), width="stretch")
            st.markdown("**Таблицы серий сгущения**")
            tabs = st.tabs(list(BOUNDARY_NAMES.values()))
            for tab, boundary in zip(tabs, BOUNDARY_NAMES):
                with tab:
                    columns = ["Метод", "N", "M", "h", "τ", "ε(T)", "p(T)", "max ε(t)", "p(max)"]
                    if axis == "tau":
                        columns += ["Разность с временным эталоном", "p(время)"]
                    show_table(table.loc[table["boundary"] == boundary, columns], f"{axis}_{boundary}")
            with st.expander("Полная таблица всех 12 серий и CSV"):
                show_table(table, f"convergence_{axis}")
        if (diagnostics["Статус"] != "OK").any():
            st.warning("Часть серий не рассчитана. Параметры и причины указаны ниже.")
        with st.expander("Контроль фиксированного шага и точности эталона"):
            st.caption("Контроль уменьшает фиксированный шаг вдвое. Для времени дополнительно сравниваются два эталона.")
            names = {"fixed_step_effect": "Изменение ε(T), доля", "control_error": "ε(T) после контроля",
                     "reference_steps": "M эталона", "reference_difference": "Разность эталонов",
                     "reference_effect": "Вклад эталона, доля"}
            show_table(diagnostics.rename(columns=names), f"diagnostics_{axis}")
            if "fixed_step_effect" in diagnostics and (diagnostics["fixed_step_effect"] > 0.1).any():
                st.warning("В некоторых сериях фиксированный шаг заметно влияет на ε(T). "
                           "Порядок p(T) нельзя считать изолированным временным порядком; для τ смотрите p(время).")
            if "reference_effect" in diagnostics and (diagnostics["reference_effect"] > 0.1).any():
                st.warning("В некоторых сериях временной эталон недостаточно точен для уверенной оценки p(время).")
        st.caption("Изменение знака ведущей ошибки, её компенсация и близость к нулю cos(T) могут искажать p(T). "
                   "Смотрите также max ε(t), p(max) и контроль фиксированного шага.")


def main():
    st.set_page_config(page_title="Лабораторная 6 — вариант 4", layout="wide")
    st.title("Лабораторная 6 · Вариант 4")
    st.latex(r"u_{tt}=u_{xx}-5u,\quad x\in[0,1],\quad u_x(0,t)-2u(0,t)=u_x(1,t)-2u(1,t)=0")
    st.latex(r"u(x,0)=e^{2x},\quad u_t(x,0)=0,\quad U(x,t)=e^{2x}\cos t")
    with st.sidebar:
        st.header("Параметры расчёта")
        scheme = st.selectbox("Схема", list(SCHEME_NAMES), format_func=SCHEME_NAMES.get, key="scheme")
        order = st.selectbox("Первый временной слой", list(START_NAMES), index=1, format_func=START_NAMES.get, key="initial_order")
        boundary = st.selectbox("Граничные условия", list(BOUNDARY_NAMES), index=2, format_func=BOUNDARY_NAMES.get, key="boundary")
        n = st.number_input("N — число интервалов по x", min_value=3, max_value=500, value=40, step=1, key="n")
        final_time = st.number_input("T — конечное время", min_value=0.01, max_value=20.0, value=1.0, step=0.1, key="final_time")
        mode = st.radio("Способ задания времени", ["Число шагов M", "Шаг τ"], key="time_mode")
        if mode == "Число шагов M":
            steps = int(st.number_input("M — число шагов", min_value=2, max_value=50000, value=200, step=10, key="steps"))
        else:
            requested_tau = st.number_input("Желаемый τ", min_value=0.00001, max_value=10.0, value=0.005, step=0.001, format="%.5f", key="requested_tau")
            steps = steps_for_tau(float(final_time), float(requested_tau))
            st.caption("M=ceil(T/τ), фактический τ=T/M.")
    config = Config(int(n), float(final_time), steps, scheme, order, boundary)
    try:
        diagnostic = stability(config)
        columns = st.columns(3)
        columns[0].metric("h", f"{config.h:.6g}")
        columns[1].metric("τ", f"{config.tau:.6g}")
        columns[2].metric("S = τ²·max(−λ)/4", f"{diagnostic.parameter:.6g}")
        st.caption(f"N={config.n}; M={config.steps}; τ/h={config.tau/config.h:.4g}; "
                   f"λmin={diagnostic.lambda_min:.6g}; λmax={diagnostic.lambda_max:.6g}. "
                   f"Для «креста» требуется λmax<0 и S<1; τ<{diagnostic.tau_limit:.6g}.")
        if scheme == "implicit":
            st.info("Правая часть уравнения берётся на новом слое k+1. Неявная схема имеет первый порядок по времени; ограничение S<1 относится к явной схеме.")
        with st.spinner("Расчёт решения…"):
            result = calculate(config, CACHE_VERSION)
    except (ValueError, FloatingPointError) as exc:
        st.error(str(exc))
        st.info("Уменьшите τ / увеличьте M. При λmax≥0 увеличьте N.")
        return
    errors = max_errors(result)
    st.caption(f"ε(T)={errors[-1]:.6e}; max ε(t)={errors.max():.6e}. Начальный слой задан точно; разностные ГУ выполняются со слоя 1.")
    layer = st.slider("Временной слой k", 0, config.steps, config.steps, key="layer")
    time = result.time[layer]
    exact = exact_solution(result.x, time)
    st.subheader(f"Решение при t={time:.6g}")
    show_plot(result.x, {"Численное решение": result.values[layer], "Аналитическое решение": exact}, "x", "u(x,t)")
    show_plot(result.x, {"Абсолютная ошибка": np.abs(result.values[layer]-exact)}, "x", "|u−U|")
    st.subheader("Максимальная ошибка по времени")
    st.latex(r"\varepsilon(t_k)=\max_{0\leq i\leq N}|u_i^k-U(x_i,t_k)|")
    show_plot(result.time, {"ε(t)": errors}, "t", "ε(t)")
    with st.expander("Таблица значений выбранного слоя"):
        show_table(pd.DataFrame({"x": result.x, "t": time, "u": result.values[layer], "U": exact,
                                 "|u−U|": np.abs(result.values[layer]-exact)}), "values")
    with st.expander("Формулы и алгоритм"):
        st.latex(r"\text{Крест: }u^{k+1}=2u^k-u^{k-1}+\tau^2Lu^k")
        st.latex(r"\text{Неявная: }(I-\tau^2L)u^{k+1}=2u^k-u^{k-1}")
        st.latex(r"v^1=v^0+\tau\dot v^0\quad\text{или}\quad v^1=v^0+\tau\dot v^0+\frac{\tau^2}{2}L_hv^0")
        st.caption("Старт и дальнейшие шаги используют один Lₕ. Для PDE-границы Taylor-формула применяется ко всем узлам, включая крайние.")
        st.markdown("Для первых двух ГУ крайние узлы исключаются до решения системы. "
                    "Для ГУ с уравнением они входят в трёхдиагональный оператор. "
                    "LU с выбором главного элемента готовится один раз. Полный вывод формул приведён в METHODS.md.")
    show_comparison(config)
    show_convergence(config)


if __name__ == "__main__":
    main()
