"""Пакетный расчёт и повторное построение графиков."""

from argparse import ArgumentParser
from pathlib import Path

from lab8.analysis import AXIS_NAMES, compare, convergence, solution_tables
from lab8.export import RESULTS_DIR, save
from lab8.plots import comparison_figures, convergence_figure, solution_figures
from lab8.solver import METHOD_NAMES, Config, solve, steps_for_tau


def main():
    defaults = Config()
    parser = ArgumentParser(description="ЛР8 · Вариант 7 · Двумерное параболическое уравнение")
    parser.add_argument("--method", choices=METHOD_NAMES, default=defaults.method)
    parser.add_argument("--nx", type=int, default=defaults.nx)
    parser.add_argument("--ny", type=int, default=defaults.ny)
    parser.add_argument("--time", type=float, default=defaults.final_time, help="Конечное время T.")
    time_grid = parser.add_mutually_exclusive_group()
    time_grid.add_argument("--steps", type=int, default=defaults.steps, help="Число шагов M.")
    time_grid.add_argument("--tau", type=float, help="Желаемый шаг, M = ceil(T/τ).")
    parser.add_argument("--at", type=float, help="Момент для графиков, отображается ближайший слой.")
    parser.add_argument("--section-i", type=int)
    parser.add_argument("--section-j", type=int)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--study", choices=(*AXIS_NAMES, "all"))
    parser.add_argument("--space-base", type=int, default=6)
    parser.add_argument("--time-base", type=int, default=60)
    parser.add_argument("--levels", type=int, default=3)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    try:
        steps = steps_for_tau(args.time, args.tau) if args.tau is not None else args.steps
        config = Config(args.nx, args.ny, steps, args.time, args.method)
        i = config.nx//2 if args.section_i is None else args.section_i
        j = config.ny//2 if args.section_j is None else args.section_j
        instant = config.final_time if args.at is None else args.at
        if not (0 <= i <= config.nx and 0 <= j <= config.ny and 0 <= instant <= config.final_time):
            raise ValueError("Момент и индексы сечений должны принадлежать сетке расчёта.")
        result = solve(config)
        k = int(abs(result.time - instant).argmin())
        tables = solution_tables(result, k, i, j)
        save(args.output, tables, solution_figures(result, tables, k, i, j))
        print(tables["summary"].to_string(index=False))
        if args.compare:
            comparison = compare(config, args.repeats)
            save(args.output, comparison.tables(), comparison_figures(comparison, config))
            print(comparison.summary.to_string(index=False))
        directions = AXIS_NAMES if args.study == "all" else ([args.study] if args.study else [])
        for direction in directions:
            base = args.time_base if direction == "tau" else args.space_base
            table = convergence(config, direction, base, args.levels)
            name = f"convergence_{direction}"
            save(args.output, {name: table}, {name: convergence_figure(table, direction)})
            print(name)
            print(table.filter(items=["Метод", "s", "E(T)", "p", "E_ref(T)", "p_ref"]).to_string(index=False))
    except (ValueError, FloatingPointError, OSError) as exc:
        parser.error(str(exc))
    print(f"PNG и CSV сохранены в {args.output.resolve()}")


if __name__ == "__main__":
    main()
