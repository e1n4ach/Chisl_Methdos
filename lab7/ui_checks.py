"""Интеграционная проверка приложения через Streamlit AppTest."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest


class InterfaceChecks(unittest.TestCase):
    def test_sidebar_calculations_and_export(self):
        app_path = Path(__file__).with_name("app.py")
        with TemporaryDirectory() as temporary, patch("lab7.export.RESULTS_DIR", Path(temporary)):
            app = AppTest.from_file(str(app_path), default_timeout=60).run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.tabs), 3)
            self.assertEqual([item.key for item in app.sidebar.selectbox], ["method", "boundary"])
            self.assertEqual([item.key for item in app.sidebar.number_input],
                             ["nx", "ny", "epsilon", "omega", "max_iter"])

            app.sidebar.selectbox(key="method").set_value("jacobi")
            app.sidebar.selectbox(key="boundary").set_value("ghost")
            app.sidebar.number_input(key="nx").set_value(8)
            app.sidebar.number_input(key="ny").set_value(10)
            app.sidebar.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["config"].method, "jacobi")
            self.assertEqual(app.session_state["config"].boundary, "ghost")
            app.slider(key="section_i_8").set_value(3)
            app.slider(key="section_j_10").set_value(7)
            app.button(key="save_solution").click().run()
            self.assertFalse(app.exception)
            self.assertTrue(app.success)
            folder = Path(temporary) / "ui"
            self.assertGreater((folder / "surfaces.png").stat().st_size, 1000)
            self.assertEqual(len(pd.read_csv(folder / "values.csv")), 9*11)
            self.assertAlmostEqual(pd.read_csv(folder / "section_y.csv")["y"][0],
                                   7*app.session_state["config"].hy)
            self.assertEqual(len(app.get("download_button")), 6)

            app.button(key="compare").click().run()
            self.assertFalse(app.exception)
            results = app.session_state["comparison"][1]
            self.assertEqual(len(results), 3)
            self.assertTrue(all(result.status == "converged" for result in results))

            for direction, column in (("hx", "Ny"), ("hy", "Nx")):
                app.radio(key="direction").set_value(direction)
                app.button(key="research").click().run()
                self.assertFalse(app.exception)
                table = app.session_state[f"convergence_{direction}"][1]
                self.assertEqual(set(table[column]), {24})
                self.assertAlmostEqual(table["p_ref"].iloc[-1], 2, delta=0.15)
            app.button(key="save_convergence_hy").click().run()
            self.assertFalse(app.exception)
            self.assertTrue((folder / "convergence_hy.png").is_file())
            self.assertTrue((folder / "convergence_hy.csv").is_file())
            self.assertIn("convergence_hx", app.session_state)

            app.sidebar.number_input(key="max_iter").set_value(1)
            app.sidebar.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertIn("max_iter", app.warning[0].value)
