"""Интеграционная проверка настоящего app.py средствами Streamlit AppTest."""

from pathlib import Path
import sys

from streamlit.testing.v1 import AppTest
import matplotlib.pyplot as plt

from analysis import COMBINATIONS
from view_plots import comparison_figure, convergence_figure


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    app = AppTest.from_file(str(Path(__file__).parent / "app.py"), default_timeout=240).run()
    assert not app.exception
    assert [box.key for box in app.sidebar.selectbox] == ["scheme", "initial_order", "boundary"]
    assert [box.key for box in app.sidebar.number_input] == ["n", "final_time", "steps"]
    for scheme, order, boundary in COMBINATIONS:
        app.selectbox(key="scheme").set_value(scheme)
        app.selectbox(key="initial_order").set_value(order)
        app.selectbox(key="boundary").set_value(boundary).run()
        assert not app.exception, (scheme, order, boundary)
        assert not app.error, (scheme, order, boundary)
    app.slider(key="layer").set_value(0).run()
    assert not app.exception
    app.radio(key="time_mode").set_value("Шаг τ").run()
    assert not app.exception
    app.button(key="compare_all").click().run()
    assert not app.exception
    assert len(app.session_state["comparison"][1][0]) == 12
    figure = comparison_figure(app.session_state["comparison"][1][1])
    assert len(figure.axes) == 3
    assert all(len(axis.lines) == 4 for axis in figure.axes)
    plt.close(figure)
    app.button(key="research").click().run()
    assert not app.exception and not app.error
    space_table, diagnostics = app.session_state["convergence_h"][1]
    assert len(space_table) == 48 and (diagnostics["Статус"] == "OK").all()
    assert set(space_table["N"]) == {12, 24, 48, 96}
    assert set(space_table["M"]) == {20000}
    figure = convergence_figure(space_table, "h")
    assert len(figure.axes) == 6 and all(len(axis.lines) == 4 for axis in figure.axes)
    plt.close(figure)
    # Радиокнопка исследуемого шага не имеет key и выбирается по подписи.
    next(radio for radio in app.radio if radio.label == "Исследуемый шаг").set_value("tau").run()
    app.button(key="research").click().run()
    assert not app.exception and not app.error
    time_table, diagnostics = app.session_state["convergence_tau"][1]
    assert len(time_table) == 48 and "p(время)" in time_table
    assert (diagnostics["Статус"] == "OK").all()
    assert set(time_table["N"]) == {60}
    assert set(time_table.loc[(time_table["scheme"] == "implicit") &
                              (time_table["initial_order"] == 2) &
                              (time_table["boundary"] == "first"), "M"]) == {640, 1280, 2560, 5120}
    figure = convergence_figure(time_table, "tau")
    assert len(figure.axes) == 6 and all(len(axis.lines) == 4 for axis in figure.axes)
    plt.close(figure)
    # Оба исследования сохраняются независимо; смена слоя не сбрасывает их.
    app.slider(key="layer").set_value(100).run()
    assert not app.exception and "convergence_h" in app.session_state
    assert "convergence_tau" in app.session_state
    app.selectbox(key="scheme").set_value("explicit")
    app.number_input(key="requested_tau").set_value(0.1).run()
    assert not app.exception and app.error
    assert "Неустойчивая" in app.error[0].value
    print("PASS Streamlit: левая панель, 12 комбинаций, ползунок, ввод τ, 3 панели по 4 линии; "
          "оба исследования по 12 серий, таблицы порядков, сохранение результатов, отказ неустойчивой сетки.")


if __name__ == "__main__":
    main()
