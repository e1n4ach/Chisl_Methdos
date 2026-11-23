"""Проверка Streamlit через AppTest с вычислениями и записью файлов."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest


class InterfaceChecks(unittest.TestCase):
    def test_sidebar_plots_studies_and_export(self):
        path = Path(__file__).with_name("app.py")
        with TemporaryDirectory() as temp, patch("lab8.export.RESULTS_DIR", Path(temp)):
            app = AppTest.from_file(str(path), default_timeout=60).run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.tabs), 3)
            self.assertEqual([box.key for box in app.sidebar.selectbox], ["method"])
            self.assertEqual([box.key for box in app.sidebar.number_input], ["nx", "ny", "final_time", "steps"])
            self.assertEqual(len(app.sidebar.slider), 3)

            app.sidebar.radio(key="time_mode").set_value("Шаг τ").run()
            app.sidebar.selectbox(key="method").set_value("fractional")
            app.sidebar.number_input(key="nx").set_value(7)
            app.sidebar.number_input(key="ny").set_value(9)
            app.sidebar.number_input(key="final_time").set_value(0.9)
            app.sidebar.number_input(key="requested_tau").set_value(0.16)
            app.sidebar.button[0].click().run()
            self.assertFalse(app.exception)
            config = app.session_state["config"]
            self.assertEqual((config.method, config.nx, config.ny, config.steps), ("fractional", 7, 9, 6))
            self.assertAlmostEqual(config.tau, 0.15)

            app.sidebar.slider(key="moment_0.9_6").set_value(0.3)
            app.sidebar.slider(key="section_i_7").set_value(3)
            app.sidebar.slider(key="section_j_9").set_value(4)
            app.button(key="save_solution").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.get("download_button")), 6)
            output = Path(temp) / "ui"
            for name in ("surfaces", "error_map", "sections", "error_time", "point_history"):
                self.assertGreater((output / f"{name}.png").stat().st_size, 1000)
            self.assertEqual(len(pd.read_csv(output / "values.csv")), 8*10)
            self.assertAlmostEqual(pd.read_csv(output / "summary.csv")["t"][0], 0.3)
            self.assertAlmostEqual(pd.read_csv(output / "section_y.csv")["y"][0], 4/9)
            self.assertAlmostEqual(pd.read_csv(output / "point_history.csv")["t"].max(), 0.9)

            app.number_input(key="repeats").set_value(1)
            app.button(key="compare").click().run()
            self.assertFalse(app.exception)
            result = app.session_state["comparison"][1]
            self.assertEqual(len(result.summary), 2)
            self.assertEqual(len(result.timings), 2)

            for direction, fixed, value, order in (("hx", "Ny", 9, "p_ref"), ("hy", "Nx", 7, "p_ref"),
                                                    ("tau", "Nx", 7, "p")):
                app.radio(key="direction").set_value(direction)
                app.button(key="research").click().run()
                self.assertFalse(app.exception)
                table = app.session_state[f"convergence_{direction}"][1]
                self.assertEqual(set(table[fixed]), {value})
                self.assertEqual(table[order].notna().sum(), 4)
            app.button(key="save_convergence_tau").click().run()
            self.assertFalse(app.exception)
            self.assertTrue((output / "convergence_tau.png").is_file())
            self.assertTrue((output / "convergence_tau.csv").is_file())
