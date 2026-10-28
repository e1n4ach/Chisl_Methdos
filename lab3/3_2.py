from typing import List, Tuple


def solve_linear_system_gauss(a: List[List[float]], b: List[float]) -> List[float]:
    """
    Решение СЛАУ методом Гаусса с частичным выбором главного элемента.
    Используется только для решения системы (3.13).
    """
    n = len(a)

    # Формируем расширенную матрицу
    aug = [row[:] + [b[i]] for i, row in enumerate(a)]

    # Прямой ход
    for k in range(n):
        # Выбор главного элемента в столбце k
        pivot_row = k
        max_value = abs(aug[k][k])
        for i in range(k + 1, n):
            if abs(aug[i][k]) > max_value:
                max_value = abs(aug[i][k])
                pivot_row = i

        if max_value == 0:
            raise ValueError("Система не имеет единственного решения: нулевой ведущий элемент.")

        if pivot_row != k:
            aug[k], aug[pivot_row] = aug[pivot_row], aug[k]

        # Обнуление элементов ниже главного
        for i in range(k + 1, n):
            factor = aug[i][k] / aug[k][k]
            for j in range(k, n + 1):
                aug[i][j] -= factor * aug[k][j]

    # Обратный ход
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = 0.0
        for j in range(i + 1, n):
            s += aug[i][j] * x[j]
        if aug[i][i] == 0:
            raise ValueError("Система не имеет единственного решения: нулевой диагональный элемент.")
        x[i] = (aug[i][n] - s) / aug[i][i]

    return x


def build_natural_cubic_spline(x: List[float], f: List[float]) -> Tuple[List[float], List[float], List[float], List[float], List[List[float]], List[float], List[float]]:
    """
    Построение кубического сплайна по формулам (3.13) и (3.14)
    для случая нулевой кривизны на концах отрезка.

    Возвращает коэффициенты a_i, b_i, c_i, d_i для интервалов:
    S_i(x) = a_i + b_i (x - x_{i-1}) + c_i (x - x_{i-1})^2 + d_i (x - x_{i-1})^3,
    i = 1, ..., n.
    """
    if len(x) != len(f):
        raise ValueError("Списки x и f должны иметь одинаковую длину.")

    n = len(x) - 1  # число интервалов
    if n < 1:
        raise ValueError("Для построения сплайна нужно минимум две точки.")

    # Проверка строгого возрастания узлов и вычисление h_i
    h = []
    for i in range(1, len(x)):
        hi = x[i] - x[i - 1]
        if hi <= 0:
            raise ValueError("Узлы x должны быть строго возрастающими.")
        h.append(hi)

    # Если интервал один, строим линейный сплайн как частный случай
    if n == 1:
        a = [f[0]]
        b = [(f[1] - f[0]) / h[0]]
        c = [0.0]
        d = [0.0]
        return a, b, c, d, A, rhs, c_full, [], [], [0.0, 0.0]

    # По условию естественного сплайна:
    # c1 = 0, c_{n+1} = 0
    # Неизвестные: c2, c3, ..., c_n
    size = n - 1
    A = [[0.0 for _ in range(size)] for _ in range(size)]
    rhs = [0.0 for _ in range(size)]

    # Система (3.13)
    for row in range(size):
        i = row + 1  # соответствует внутреннему узлу x_i, i = 1..n-1 в нумерации Python

        if row - 1 >= 0:
            A[row][row - 1] = h[i - 1]

        A[row][row] = 2.0 * (h[i - 1] + h[i])

        if row + 1 < size:
            A[row][row + 1] = h[i]

        rhs[row] = 3.0 * ((f[i + 1] - f[i]) / h[i] - (f[i] - f[i - 1]) / h[i - 1])

    internal_c = solve_linear_system_gauss(A, rhs)

    # Полный список c: [c1, c2, ..., c_n, c_{n+1}]
    c_full = [0.0] + internal_c + [0.0]

    # Восстановление коэффициентов по формулам (3.14)
    a = [0.0] * n
    b = [0.0] * n
    c = [0.0] * n
    d = [0.0] * n

    for i in range(1, n + 1):
        hi = h[i - 1]
        a[i - 1] = f[i - 1]
        c[i - 1] = c_full[i - 1]

        if i < n:
            b[i - 1] = (f[i] - f[i - 1]) / hi - hi * (c_full[i] + 2.0 * c_full[i - 1]) / 3.0
        else:
            b[i - 1] = (f[i] - f[i - 1]) / hi - 2.0 * hi * c_full[i - 1] / 3.0

        d[i - 1] = (c_full[i] - c_full[i - 1]) / (3.0 * hi)

    return a, b, c, d, A, rhs, c_full


def evaluate_spline(x_nodes: List[float], a: List[float], b: List[float], c: List[float], d: List[float], x_star: float) -> Tuple[float, int]:
    """
    Вычисление значения сплайна в точке x_star.
    Возвращает значение и номер интервала (нумерация с 1).
    """
    n = len(x_nodes) - 1

    if not (x_nodes[0] <= x_star <= x_nodes[-1]):
        raise ValueError("Точка x* должна принадлежать отрезку [x0, xn].")

    # Поиск интервала [x_{i-1}, x_i]
    interval = None
    for i in range(1, n + 1):
        if x_nodes[i - 1] <= x_star <= x_nodes[i]:
            interval = i
            break

    if interval is None:
        raise ValueError("Не удалось определить интервал для точки x*.")

    dx = x_star - x_nodes[interval - 1]
    value = a[interval - 1] + b[interval - 1] * dx + c[interval - 1] * dx ** 2 + d[interval - 1] * dx ** 3
    return value, interval


def print_table(x: List[float], a: List[float], b: List[float], c: List[float], d: List[float]) -> None:
    print("\nКоэффициенты кубического сплайна:")
    print("-" * 106)
    print(f"{'i':>2} | {'[x_(i-1), x_i]':>20} | {'a_i':>14} | {'b_i':>14} | {'c_i':>14} | {'d_i':>14}")
    print("-" * 106)
    for i in range(1, len(a) + 1):
        interval_str = f"[{x[i - 1]:.6f}, {x[i]:.6f}]"
        print(
            f"{i:>2} | {interval_str:>20} | "
            f"{a[i - 1]:>14.8f} | {b[i - 1]:>14.8f} | {c[i - 1]:>14.8f} | {d[i - 1]:>14.8f}"
        )
    print("-" * 106)


def main() -> None:
    print("ЛР 3.2, вариант 24")
    print("Построение кубического сплайна с нулевой кривизной на концах отрезка")

    # Данные варианта 24
    x = [0.1, 0.5, 0.9, 1.3, 1.7]
    f = [100.00, 4.0, 1.2346, 0.59172, 0.34602]
    x_star = 0.8

    print("\nИсходные данные:")
    print("x_i =", x)
    print("f_i =", f)
    print(f"x* = {x_star}")

    # Шаги h_i
    h = [x[i] - x[i - 1] for i in range(1, len(x))]
    print("\nШаги h_i:")
    for i, hi in enumerate(h, start=1):
        print(f"h{i} = x{i} - x{i - 1} = {x[i]} - {x[i - 1]} = {hi:.8f}")

    # Построение сплайна
    a, b, c, d, A, rhs, c_full = build_natural_cubic_spline(x, f)


    print("\nСистема для коэффициентов c_i по формуле (3.13):")
    if A:
        for i in range(len(A)):
            row_str = " + ".join(f"({A[i][j]:.8f})*c{j + 2}" for j in range(len(A[i])) if abs(A[i][j]) > 1e-15)
            print(f"{row_str} = {rhs[i]:.8f}")
    else:
        print("Внутренних коэффициентов нет, так как интервал только один.")

    print("\nКоэффициенты c с учетом условий нулевой кривизны:")
    for idx, value in enumerate(c_full, start=1):
        print(f"c{idx} = {value:.8f}")

    print_table(x, a, b, c, d)

    # Вычисление значения в точке x*
    y_star, interval = evaluate_spline(x, a, b, c, d, x_star)
    left = x[interval - 1]
    right = x[interval]

    print(f"\nТочка x* = {x_star} принадлежит интервалу [{left}, {right}] (i = {interval}).")
    print("На этом интервале сплайн имеет вид:")
    print(
        f"S_{interval}(x) = {a[interval - 1]:.8f}"
        f" + ({b[interval - 1]:.8f}) * (x - {left:.8f})"
        f" + ({c[interval - 1]:.8f}) * (x - {left:.8f})^2"
        f" + ({d[interval - 1]:.8f}) * (x - {left:.8f})^3"
    )

    print(f"\nЗначение сплайна в точке x* = {x_star}:")
    print(f"S({x_star}) ≈ {y_star:.10f}")


if __name__ == "__main__":
    main()
