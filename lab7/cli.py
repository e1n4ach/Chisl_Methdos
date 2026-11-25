"""Повторный расчёт и экспорт графиков без Streamlit."""

from argparse import ArgumentParser
from pathlib import Path

from lab7.analysis import AXIS_NAMES, compare, comparison_tables, convergence, result_tables
from lab7.boundaries import BOUNDARY_NAMES
from lab7.export import RESULTS_DIR, save
from lab7.plots import comparison_figures, convergence_figure, solution_figures
from lab7.solver import METHOD_NAMES, Config, solve


def main() -> None:
    defaults = Config()
    parser = ArgumentParser(description="ЛР7 · Вариант 3 · Задача Лапласа")
    parser.add_argument("--nx", type=int, default=defaults.nx)
    parser.add_argument("--ny", type=int, default=defaults.ny)
    parser.add_argument("--method", choices=METHOD_NAMES, default=defaults.method)
    parser.add_argument("--boundary", choices=BOUNDARY_NAMES, default=defaults.boundary)
    parser.add_argument("--epsilon", type=float, default=defaults.epsilon)
    parser.add_argument("--omega", type=float, default=defaults.omega)
    parser.add_argument("--max-iter", type=int, default=defaults.max_iter)
    parser.add_argument("--section-i", type=int)
    parser.add_argument("--section-j", type=int)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR)
    parser.add_argument("--compare", action="store_true", help="Сравнить три метода.")
    parser.add_argument("--study", choices=(*AXIS_NAMES, "all"),
                        help="Пересчитать серии сгущения выбранного направления или все серии.")
    parser.add_argument("--base", type=int, default=6, help="Начальное число интервалов по сгущаемой оси.")
    parser.add_argument("--fixed", type=int, help="Интервалы по фиксированной оси (начальное Ny для both).")
    parser.add_argument("--levels", type=int, default=3, help="Количество сеток серии.")
    args = parser.parse_args()
    try:
        config = Config(args.nx, args.ny, args.method, args.boundary, args.epsilon, args.omega, args.max_iter)
        i = args.section_i if args.section_i is not None else config.nx // 2
        j = args.section_j if args.section_j is not None else config.ny // 2
        if not (0 <= i <= config.nx and 0 <= j <= config.ny):
            raise ValueError("Индексы сечений должны принадлежать сетке.")
        result = solve(config)
        tables = result_tables(result, i, j)
        save(args.output, tables, solution_figures(result, i, j))
        print(tables["summary"].to_string(index=False))
        if args.compare:
            results = compare(config)
            tables = comparison_tables(results)
            save(args.output, tables, comparison_figures(results))
            print(tables["comparison"].to_string(index=False))
        directions = AXIS_NAMES if args.study == "all" else ([args.study] if args.study else [])
        for direction in directions:
            fixed = args.fixed if args.fixed is not None else (8 if direction == "both" else 24)
            table = convergence(config, direction, args.base, fixed, args.levels)
            name = f"convergence_{direction}"
            save(args.output, {name: table}, {name: convergence_figure(table, config, direction)})
            print(name)
            print(table.filter(items=["Nx", "Ny", "E", "p", "E_ref", "p_ref", "Статус"]).to_string(index=False))
    except ValueError as exc:
        parser.error(str(exc))
    print(f"PNG и CSV сохранены в {args.output.resolve()}")


if __name__ == "__main__":
    main()
