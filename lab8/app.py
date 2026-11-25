"""Интерфейс расчёта, сравнения методов и исследования сходимости."""

import streamlit as st

from lab8.analysis import AXIS_NAMES, compare, convergence, solution_tables
from lab8.export import RESULTS_DIR, csv_bytes, save
from lab8.plots import comparison_figures, convergence_figure, settings, solution_figures
from lab8.solver import METHOD_NAMES, Config, solve, steps_for_tau


@st.cache_data(show_spinner=False, max_entries=4)
def calculate(config):
    return solve(config)


@st.cache_data(show_spinner=False, max_entries=12)
def calculate_convergence(config, direction, base, levels):
    return convergence(config, direction, base, levels)


def sidebar_config():
    defaults = Config()
    with st.sidebar:
        st.header("Параметры расчёта")
        time_mode = st.radio("Как задать временную сетку", ["Число шагов M", "Шаг τ"], key="time_mode")
        with st.form("parameters", border=False):
            method = st.selectbox("Метод", list(METHOD_NAMES), format_func=METHOD_NAMES.get, key="method")
            nx = st.number_input("Nx — интервалы по x", 2, 150, defaults.nx, key="nx")
            ny = st.number_input("Ny — интервалы по y", 2, 150, defaults.ny, key="ny")
            final_time = st.number_input("T — конечное время", 0.001, 100.0, defaults.final_time,
                                         step=0.1, format="%.4f", key="final_time")
            if time_mode == "Число шагов M":
                steps = st.number_input("M — число шагов", 1, 10_000, defaults.steps, key="steps")
            else:
                requested_tau = st.number_input("Желаемый шаг τ", 0.000001, 100.0, 0.01,
                                                step=0.001, format="%.6f", key="requested_tau")
                st.caption("M = ceil(T/τ), фактический шаг равен T/M.")
            if st.form_submit_button("Рассчитать"):
                try:
                    if time_mode == "Шаг τ":
                        steps = steps_for_tau(final_time, requested_tau)
                        if steps > 10_000:
                            raise ValueError("Требуется M ≤ 10000. Увеличьте желаемый шаг τ.")
                    config = Config(nx, ny, steps, final_time, method)
                    with st.spinner("Расчёт временных слоёв…"):
                        calculate(config)
                    st.session_state["config"] = config
                except (ValueError, FloatingPointError) as exc:
                    st.error(str(exc))
        config = st.session_state.get("config", defaults)
        st.caption(f"Текущая сетка: M={config.steps}, T={config.final_time:g}, τ={config.tau:.6g}.")
        st.subheader("Отображение")
        moment = st.slider("Момент времени t", 0.0, config.final_time, config.final_time,
                           step=config.tau, format="%.6f", key=f"moment_{config.final_time}_{config.steps}")
        i = st.slider("Индекс i для x = const", 0, config.nx, config.nx//2, key=f"section_i_{config.nx}")
        j = st.slider("Индекс j для y = const", 0, config.ny, config.ny//2, key=f"section_j_{config.ny}")
    return config, moment, i, j


def show_table(table, key, label="Скачать CSV"):
    numeric = ("E_max(t)", "|u−U|", "E(T)", "max_t E_max", "E_ref(T)", "Разность эталонов",
               "Медиана, с", "Минимум, с", "Максимум, с", "Время, с")
    columns = {name: st.column_config.NumberColumn(format="%.5e") for name in numeric if name in table}
    st.dataframe(table, hide_index=True, width="stretch", column_config=columns)
    st.download_button(label, csv_bytes(table), file_name=f"{key}.csv", mime="text/csv",
                       key=f"download_{key}", on_click="ignore")


def show_figures_and_save(key, tables, figures):
    for figure in figures.values():
        st.pyplot(figure, width="stretch")
    if st.button("Сохранить PNG и CSV", key=f"save_{key}"):
        try:
            save(RESULTS_DIR / "ui", tables, figures)
            st.success("Файлы сохранены в lab8/results/ui.")
        except OSError as exc:
            st.error(f"Ошибка записи файлов ({exc}).")


def show_solution(result, k, i, j):
    tables = solution_tables(result, k, i, j)
    row = tables["summary"].iloc[0]
    columns = st.columns(5)
    for column, (label, value) in zip(columns, (
        ("hx", f"{row['hx']:.5g}"), ("hy", f"{row['hy']:.5g}"), ("τ", f"{row['τ']:.5g}"),
        ("Фактический t", f"{row['t']:.6g}"), ("E_max(t)", f"{row['E_max(t)']:.5e}"),
    )):
        column.metric(label, value)
    st.caption(f"{settings(result.config)} · слой k={k} · пересечение сечений "
               f"({result.problem.x[i]:.5g}; {result.problem.y[j]:.5g}).")
    st.caption("Сплошные линии показывают численное решение, штриховые — аналитическое. "
               "Временные графики охватывают все слои от 0 до T.")
    show_figures_and_save("solution", tables, solution_figures(result, tables, k, i, j))
    st.subheader("Таблицы расчёта")
    show_table(tables["summary"], "summary", "Скачать сводку CSV")
    for label, key in (
        ("Поле на выбранном слое", "values"), ("Сечение y = const", "section_y"),
        ("Сечение x = const", "section_x"), ("Максимальная ошибка по времени", "error_time"),
        ("Решение в фиксированных точках", "point_history"),
    ):
        with st.expander(label):
            show_table(tables[key], key)
    with st.expander("Разностные схемы и промежуточная граница"):
        st.latex(r"\mathrm{ADI}:\ (I-\tfrac{\tau}{2}D_x)w=(I+\tfrac{\tau}{2}D_y)u^n+\tfrac{\tau}{2}f^{n+1/2},"
                 r"\quad (I-\tfrac{\tau}{2}D_y)u^{n+1}=(I+\tfrac{\tau}{2}D_x)w+\tfrac{\tau}{2}f^{n+1/2}")
        st.latex(r"\mathrm{FS}:\ (I-\tau D_x)w=u^n+\tfrac{\tau}{2}f^{n+1},"
                 r"\quad (I-\tau D_y)u^{n+1}=w+\tfrac{\tau}{2}f^{n+1}")
        st.markdown("Вертикальные края промежуточного слоя выводятся из уравнений соответствующей схемы. "
                    "На полном слое задаются исходные условия Дирихле при новом времени.")


def show_comparison(config):
    st.caption("Обе схемы рассчитываются на текущей сетке. Замеры включают подготовку прогонок и запись "
               "временных слоёв. На графике времени показаны медиана и диапазон замеров.")
    repeats = st.number_input("Количество повторов на метод", 1, 10, 3, key="repeats")
    signature = (config, repeats)
    if st.button("Сравнить методы", key="compare"):
        with st.spinner("Расчёт и замер времени…"):
            try:
                st.session_state["comparison"] = (signature, compare(config, repeats))
            except (ValueError, FloatingPointError) as exc:
                st.error(str(exc))
    saved = st.session_state.get("comparison")
    if saved is None or saved[0] != signature:
        return
    comparison = saved[1]
    show_figures_and_save("comparison", comparison.tables(), comparison_figures(comparison, config))
    show_table(comparison.summary, "comparison", "Скачать сравнение CSV")
    with st.expander("Истории ошибок и отдельные замеры"):
        show_table(comparison.histories, "comparison_history")
        show_table(comparison.timings, "comparison_timings")


def show_convergence(config):
    direction = st.radio("Исследуемый шаг", list(AXIS_NAMES), format_func=AXIS_NAMES.get,
                         horizontal=True, key="direction")
    left, right = st.columns(2)
    label = {"hx": "Начальное Nx", "hy": "Начальное Ny", "tau": "Начальное M"}[direction]
    base = left.number_input(label, 2, 16 if direction != "tau" else 400,
                             6 if direction != "tau" else 60, key=f"base_{direction}")
    levels = right.number_input("Количество сеток", 2, 4, 3, key="levels")
    st.caption("В серии удваивается только выбранное число интервалов. Остальные параметры "
               "берутся из запущенного расчёта, конечное время всегда одинаково.")
    if direction != "tau":
        st.info("Для варианта 7 центральные пространственные разности точны на U=xy cos(t). "
                "Пространственный порядок оценивается по E_ref на контрольной задаче с добавленной синусоидой.")
        st.caption("Эталон сгущён по выбранной оси в 8 раз относительно последней сетки серии. "
                   "Столбец «Разность эталонов» сравнивает его с 4-кратным сгущением.")
    st.latex(r"p=\log(E_c/E_f)/\log(s_c/s_f)")
    key = f"convergence_{direction}"
    signature = (config, direction, base, levels)
    if st.button("Провести исследование", key="research"):
        with st.spinner("Решение на последовательно сгущаемых сетках…"):
            try:
                st.session_state[key] = (signature, calculate_convergence(*signature))
            except (ValueError, FloatingPointError) as exc:
                st.error(str(exc))
    saved = st.session_state.get(key)
    if saved is None or saved[0] != signature:
        return
    table = saved[1]
    if "E_ref(T)" in table and (table["Разность эталонов"] > 0.1*table["E_ref(T)"]).any():
        st.warning("Разность эталонов превышает 10% от E_ref на части сеток.")
    show_figures_and_save(key, {key: table}, {key: convergence_figure(table, direction)})
    show_table(table, key, "Скачать исследование CSV")


def main():
    st.set_page_config(page_title="Лабораторная 8 — вариант 7", layout="wide")
    st.title("Лабораторная 8 · Вариант 7")
    st.latex(r"u_t=u_{xx}+u_{yy}-xy\sin t,\quad (x,y)\in[0,1]^2,\quad U=xy\cos t")
    st.latex(r"u(0,y,t)=u(x,0,t)=0,\quad u(1,y,t)=y\cos t,\quad "
             r"u(x,1,t)=x\cos t,\quad u(x,y,0)=xy")
    config, moment, i, j = sidebar_config()
    solution_tab, comparison_tab, study_tab = st.tabs(["Решение", "Сравнение методов", "Сгущение сетки"])
    with solution_tab:
        with st.spinner("Подготовка временных слоёв…"):
            result = calculate(config)
        k = int(abs(result.time-moment).argmin())
        show_solution(result, k, i, j)
    with comparison_tab:
        show_comparison(config)
    with study_tab:
        show_convergence(config)


if __name__ == "__main__":
    main()
