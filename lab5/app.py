"""Локальный интерфейс: python -m streamlit run lab5/app.py."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from lab5 import (Problem, SCHEMES, BOUNDARIES, SCHEME_NAMES, BC_NAMES, solve, max_error,
                  comparison_plots, convergence_study, stability_experiment)


RESEARCH_PLOTS = {
    "comparison.png": ("Сравнение трёх схем и трёх способов задания левого ГУ",
                       "На каждой панели — один способ ГУ. Линии показывают ошибку "
                       "трёх схем во времени: чем ниже линия, тем точнее решение."),
    "error_vs_h.png": ("Как уменьшается ошибка при сгущении сетки",
                       "По горизонтали — шаг h, по вертикали — ошибка в момент T. "
                       "Шаг времени уменьшается вместе с h², сохраняя одинаковую σ."),
    "convergence.png": ("Оценка порядка точности",
                        "Те же ошибки в координатах ln h и ln ε. Наклон p в легенде "
                        "показывает наблюдаемый порядок: p ≈ 2 означает второй порядок."),
    "stability.png": ("Устойчивость явной схемы",
                      "Сравниваются два расчёта с почти одинаковыми начальными данными. "
                      "Рост их разности означает усиление малого возмущения. "
                      "Вертикальная шкала логарифмическая."),
}


@st.cache_data(show_spinner=False, max_entries=3)
def calculate(problem, n, final_time, sigma, scheme, boundary):
    """Кешируем расчёт: движение ползунка не запускает метод повторно."""
    x, t, u = solve(problem, n=n, final_time=final_time, sigma=sigma,
                    scheme=scheme, boundary=boundary)
    error = max_error(problem, x, t, u) if problem.has_exact() else None
    return x, t, u, error


def conduct_research(problem, n, final_time, sigma):
    """Запускаем штатные исследования и забираем только что построенные рисунки."""
    images, errors = {}, []
    jobs = []
    if problem.has_exact():
        jobs.extend([
            ("Сравнение девяти комбинаций", lambda output:
             comparison_plots(problem, n, final_time, None, sigma, output),
             ("comparison.png",)),
            ("Сгущение сетки", lambda output:
             convergence_study(problem, final_time, sigma, [n, 2*n, 4*n], output),
             ("error_vs_h.png", "convergence.png")),
        ])
    jobs.append(("Эксперимент устойчивости", lambda output:
                 stability_experiment(problem, n, final_time, output), ("stability.png",)))

    # Функции lab5.py сохраняют построенные ими графики в указанную папку.
    # Временная папка исключает чтение старых PNG от другого запуска или параметров.
    with TemporaryDirectory(prefix="lab5_research_") as directory:
        output = Path(directory)
        for title, calculation, filenames in jobs:
            existing_figures = set(plt.get_fignums())
            try:
                calculation(output)
                for filename in filenames:
                    images[filename] = (output/filename).read_bytes()
            except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
                # Неудача одного исследования не скрывает остальные результаты.
                errors.append(f"{title}: {exc}")
            finally:
                for number in set(plt.get_fignums()) - existing_figures:
                    plt.close(number)
    return images, errors


def show_research(problem, n, final_time, sigma, parameters):
    st.header("Исследование схем")
    st.caption(f"Сравнение: n = {n}, T = {final_time:g}, σ = {sigma:g}. "
               f"Сгущение: n = {n}, {2*n}, {4*n} при постоянной фактической σ. "
               f"Устойчивость: n = {n}, T = {final_time:g}, σ около 0.4 и 0.6.")
    if not problem.has_exact():
        st.info("Без применимого точного решения доступен только эксперимент устойчивости.")

    # Результаты принадлежат текущим настройкам и сохраняются при движении ползунка.
    if st.session_state.get("research_parameters") != parameters:
        st.session_state.pop("research_result", None)
        st.session_state["research_parameters"] = parameters
    if st.button("Провести исследование", key="run_research"):
        with st.spinner("Сравниваем схемы, сгущаем сетку и проверяем устойчивость…"):
            st.session_state["research_result"] = conduct_research(problem, n, final_time, sigma)

    result = st.session_state.get("research_result")
    if result is None:
        st.info("Нажмите «Провести исследование», чтобы построить дополнительные графики.")
        return
    images, errors = result
    for message in errors:
        st.warning(message)
    for filename, (title, explanation) in RESEARCH_PLOTS.items():
        if filename in images:
            st.subheader(title)
            st.caption(explanation)
            st.image(images[filename])
    if problem.has_exact():
        st.caption("В варианте 8 даже первое ГУ может дать p ≈ 2: у точного решения "
                   "Uₓₓ(0,t) = 0 и ведущая ошибка этой граничной формулы исчезает.")


def main():
    st.set_page_config(page_title="Лабораторная 5 — вариант 8", layout="centered")
    st.title("Лабораторная 5 · Вариант 8")
    st.latex(r"u_t = a u_{xx} + cu,\quad x\in[0,\pi/2],\quad u(x,0)=\sin x")
    st.caption("Слева: uₓ(0,t) = exp((c−a)t). Справа: u(π/2,t) = exp((c−a)t).")

    with st.sidebar:
        st.header("Параметры расчёта")
        scheme = st.selectbox("Схема", list(SCHEMES), index=2,
                              format_func=SCHEME_NAMES.get, key="scheme")
        boundary = st.selectbox("Левое ГУ", BOUNDARIES, index=2,
                                format_func=BC_NAMES.get, key="boundary")
        a = st.number_input("a", min_value=0.001, value=1.0, step=0.1, key="a")
        c = st.number_input("c", max_value=-0.001, value=-1.0, step=0.1, key="c")
        n = st.number_input("n — число интервалов", min_value=3, value=40, step=1, key="n")
        final_time = st.number_input("T — конечное время", min_value=0.001,
                                     value=1.0, step=0.1, key="final_time")
        sigma = st.number_input("σ = aτ/h²", min_value=0.001, value=0.4,
                                 step=0.05, key="sigma")
        st.caption("Шаг времени выбирается по σ; число шагов округляется вверх до целого.")

    problem = Problem(a=a, c=c)
    try:
        with st.spinner("Расчёт решения…"):
            x, t, u, error = calculate(problem, n, final_time, sigma, scheme, boundary)
    except (ValueError, FloatingPointError, OverflowError) as exc:
        st.error(str(exc))
        st.info("Для явной схемы попробуйте уменьшить σ. Проверьте параметры сетки.")
        st.stop()

    h, tau = float(x[1]-x[0]), float(t[1]-t[0])
    columns = st.columns(3)
    columns[0].metric("h", f"{h:.6g}")
    columns[1].metric("τ", f"{tau:.6g}")
    columns[2].metric("δ(T)", f"{error[-1]:.6e}" if error is not None else "—")
    st.caption(f"Фактическая σ = {problem.a*tau/h**2:.6g}; временных шагов: {len(t)-1}.")

    # Ползунок задаёт время; показываем ближайший рассчитанный слой без интерполяции.
    selected_time = st.slider("Момент времени t", min_value=0.0,
                               max_value=float(t[-1]), value=float(t[-1]),
                               step=float(t[-1])/1000,
                               format="%.4f", key="selected_time")
    index = int(np.argmin(np.abs(t-selected_time)))
    st.subheader("Решение в выбранный момент")
    st.caption("Численная кривая сравнивается с точной при одном и том же времени."
               if error is not None else "Показано численное решение на выбранном слое.")
    st.caption(f"Показан слой k = {index}, t = {t[index]:.6g}.")
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(x, u[index], "o-", markersize=3, label="Численное решение")
    if error is not None:
        fine_x = np.linspace(0, problem.length, 400)
        axis.plot(fine_x, problem.exact(fine_x, t[index]), "--", label="Точное решение")
    else:
        st.info("Для этих параметров точное решение варианта 8 неприменимо. "
                "Сравнение и оценка ошибки отключены.")
    axis.set(xlabel="x", ylabel="u(x,t)", title=f"Решение при t = {t[index]:.6g}")
    axis.grid(True)
    axis.legend()
    figure.tight_layout()
    st.pyplot(figure)
    plt.close(figure)

    with st.expander("Решение в нескольких моментах времени — solution.png"):
        st.caption("Каждый цвет — свой момент времени: точки показывают численный "
                   "результат." + (" Линии показывают точное решение."
                                   if error is not None else ""))
        figure, axis = plt.subplots(figsize=(8, 4))
        for layer in np.unique(np.round(np.linspace(0, len(t)-1, 5)).astype(int)):
            line, = axis.plot(x, u[layer], "o", markersize=3, label=f"t = {t[layer]:.3g}")
            if error is not None:
                fine_x = np.linspace(0, problem.length, 400)
                axis.plot(fine_x, problem.exact(fine_x, t[layer]), color=line.get_color())
        axis.set(xlabel="x", ylabel="u(x,t)")
        axis.grid(True)
        axis.legend()
        figure.tight_layout()
        st.pyplot(figure)
        plt.close(figure)

    if error is not None:
        st.subheader("Максимальная ошибка по пространству")
        st.caption("Для каждого времени берём наибольшее отклонение от точного решения "
                   "среди всех узлов. Вертикальная линия отмечает время на ползунке.")
        st.latex(r"\delta(t_k)=\max_i |u_i^k-U(x_i,t_k)|")
        figure, axis = plt.subplots(figsize=(8, 3))
        axis.plot(t, error)
        axis.axvline(t[index], color="gray", linestyle=":", label="Выбранное время")
        axis.set(xlabel="t", ylabel="δ(t)")
        axis.grid(True)
        axis.legend()
        figure.tight_layout()
        st.pyplot(figure)
        plt.close(figure)

    parameters = (a, c, n, final_time, sigma, scheme, boundary, problem.has_exact())
    show_research(problem, n, final_time, sigma, parameters)


if __name__ == "__main__":
    main()
