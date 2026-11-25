"""Общий экспорт расчётов и графиков из интерфейса и командной строки."""

from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def csv_bytes(table) -> bytes:
    return table.to_csv(index=False).encode("utf-8-sig")


def save(output, tables, figures) -> None:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        (output / f"{name}.csv").write_bytes(csv_bytes(table))
    for name, figure in figures.items():
        figure.savefig(output / f"{name}.png", dpi=140)
