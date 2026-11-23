"""Запись таблиц и рисунков из Streamlit и CLI."""

from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def csv_bytes(table):
    return table.to_csv(index=False).encode("utf-8-sig")


def save(path, tables, figures):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        (path / f"{name}.csv").write_bytes(csv_bytes(table))
    for name, figure in figures.items():
        figure.savefig(path / f"{name}.png", dpi=140)
