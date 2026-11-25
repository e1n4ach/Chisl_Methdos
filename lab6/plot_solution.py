"""Отдельная генерация results/solutions_xt.png за заданное время T."""

import argparse
from math import ceil, isfinite, pi
from pathlib import Path
import re

from plots import save_solution_plot
from solver import Config, stability, steps_for_tau


def parse_time(value: str) -> float:
    multiple = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(?:pi|π)", value.strip().lower())
    if multiple:
        result = float(multiple.group(1))*pi
    else:
        try:
            result = float(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("Укажите положительное число или кратное π, например 6pi.") from exc
    if not isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("T должно быть конечным положительным числом.")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="График u(x,t) для явного креста, старта 2 и PDE-ГУ.")
    parser.add_argument("--T", type=parse_time, default=6*pi,
                        help="Конечное время; по умолчанию 6pi.")
    args = parser.parse_args()
    periods = round(args.T/(2*pi))
    divisor = 4*periods if periods > 0 and abs(args.T-2*pi*periods) < 1e-10 else 4
    steps = divisor*ceil(steps_for_tau(args.T, 0.005)/divisor)
    config = Config(n=40, steps=steps, final_time=args.T, scheme="explicit",
                    initial_order=2, boundary="pde")
    diagnostic = stability(config)
    if not diagnostic.stable:
        parser.error(f"Явная схема неустойчива: S={diagnostic.parameter:.6g}, "
                     f"λmax={diagnostic.lambda_max:.6g}; предел τ={diagnostic.tau_limit:.6g}.")
    output = Path(__file__).parent / "results"
    output.mkdir(exist_ok=True)
    save_solution_plot(config, output)
    print(f"Сохранено: {output / 'solutions_xt.png'}")
    print(f"N={config.n}, M={config.steps}, T={config.final_time:.9g}, "
          f"τ={config.tau:.9g} ≤ 0.005; S={diagnostic.parameter:.6g}<1, "
          f"λmax={diagnostic.lambda_max:.6g}<0 — явная схема устойчива.")


if __name__ == "__main__":
    main()
