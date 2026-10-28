# Построение интерполяционных многочленов Лагранжа и Ньютона

# fabs используется для вычисления абсолютной погрешности:
# |точное значение - приближённое значение|
from math import fabs

# 1. Исходная функция по варианту
def f(x: float) -> float:
    # По варианту 24 функция имеет вид y = 1 / x^2
    # Именно её значения мы считаем в узлах интерполяции и в точке X*
    return 1.0 / (x * x)

# 2. Проверки входных данных
def check_nodes(x_nodes):
    # Для интерполяции нужно хотя бы две точки.
    # Если точек меньше, построить интерполяционный многочлен нельзя.
    if len(x_nodes) < 2:
        raise ValueError("Ошибка: для интерполяции нужно минимум 2 узла.")

    # Проверка несовпадения узлов
    # Если какие-то x_i совпадают, то в формулах Лагранжа и Ньютона
    # появится деление на ноль, поэтому такие данные недопустимы
    for i in range(len(x_nodes)):
        for j in range(i + 1, len(x_nodes)):
            if x_nodes[i] == x_nodes[j]:
                raise ValueError("Ошибка: интерполяционные узлы должны быть различными.")

# 3. Формирование таблицы значений функции
def make_table(x_nodes):
    # Здесь по каждому узлу x_i вычисляется значение функции y_i = f(x_i).
    # По сути, мы получаем таблично заданную функцию:
    # (x_0, y_0), (x_1, y_1), ..., (x_n, y_n)
    y_nodes = []
    for x in x_nodes:
        y_nodes.append(f(x))
    return y_nodes

# 4. Многочлен Лагранжа
#    Формула вида:
#    L_n(x) = sum( f(x_i) * l_i(x) )
def lagrange_value(x_nodes, y_nodes, x_star):
    # n = количество узлов
    n = len(x_nodes)

    # В result постепенно будет накапливаться значение многочлена Лагранжа
    # в точке x_star
    result = 0.0

    # Внешний цикл перебирает каждое слагаемое суммы Лагранжа
    for i in range(n):
        # numerator  = произведение (x* - x_j), j != i
        # denominator = произведение (x_i - x_j), j != i
        # Их отношение даёт базисный многочлен l_i(x*)
        numerator = 1.0
        denominator = 1.0

        # Строим произведения для i-го базисного многочлена
        for j in range(n):
            if i != j:
                numerator *= (x_star - x_nodes[j])
                denominator *= (x_nodes[i] - x_nodes[j])

        # Добавляем очередное слагаемое:
        # y_i * l_i(x*)
        result += y_nodes[i] * numerator / denominator

    # После окончания цикла получаем итоговое значение L_n(x*)
    return result

# Красивый вывод  === === === === === === === === === === === === ===
def lagrange_coefficients_for_print(x_nodes, y_nodes):
    # Эта функция нужна не для вычисления значения в точке,
    # а для красивой записи самого многочлена Лагранжа в виде строки.

    # Здесь считаются коэффициенты вида:
    # f(x_i) / w'(x_i)
    # как в примере из методички.
    n = len(x_nodes)
    coeffs = []

    # Для каждого узла x_i считаем произведение
    # (x_i - x_j), j != i
    for i in range(n):
        w_derivative = 1.0
        for j in range(n):
            if i != j:
                w_derivative *= (x_nodes[i] - x_nodes[j])

        # Сохраняем коэффициент перед соответствующим произведением
        coeffs.append(y_nodes[i] / w_derivative)

    return coeffs

def lagrange_polynomial_string(x_nodes, y_nodes):
    # Получаем коэффициенты для записи многочлена
    coeffs = lagrange_coefficients_for_print(x_nodes, y_nodes)
    parts = []

    # Формируем текст каждого слагаемого многочлена Лагранжа
    for i in range(len(x_nodes)):
        term = f"{coeffs[i]:+.10f}"

        # К коэффициенту дописываем множители (x - x_j), j != i
        for j in range(len(x_nodes)):
            if i != j:
                term += f" * (x - {x_nodes[j]})"

        parts.append(term)

    # Склеиваем все слагаемые в одну строку
    return " ".join(parts)
# === === === === === === === === === === === === === === === === ===

# 5. Разделённые разности для многочлена Ньютона
def divided_differences_table(x_nodes, y_nodes):
    # n = число узлов
    n = len(x_nodes)

    # Создаём квадратную таблицу n x n, заполненную нулями.
    # В ней будут храниться разделённые разности:
    # 0-го, 1-го, 2-го, ... порядка
    table = [[0.0 for _ in range(n)] for _ in range(n)]

    # Нулевой порядок разделённых разностей —
    # это просто сами значения функции в узлах:
    # f[x_i] = y_i
    for i in range(n):
        table[i][0] = y_nodes[i]

    # Далее строим разделённые разности более высоких порядков.
    #
    # order = 1  -> первый порядок
    # order = 2  -> второй порядок
    # и так далее
    for order in range(1, n):
        for i in range(n - order):
            # Знаменатель формулы разделённой разности:
            # x_{i+order} - x_i
            denominator = x_nodes[i + order] - x_nodes[i]

            # Дополнительная защита от деления на ноль
            if denominator == 0:
                raise ValueError("Ошибка: деление на ноль при вычислении разделённых разностей.")

            # Формула разделённой разности:
            # f[x_i, ..., x_{i+order}] =
            # ( f[x_{i+1}, ..., x_{i+order}] - f[x_i, ..., x_{i+order-1}] )
            # / (x_{i+order} - x_i)
            table[i][order] = (table[i + 1][order - 1] - table[i][order - 1]) / denominator

    # Возвращаем готовую таблицу разделённых разностей
    return table


# 6. Многочлен Ньютона
#    P_n(x) = f[x0] + f[x0,x1](x-x0) + ...
def newton_value(x_nodes, diff_table, x_star):
    # n = число узлов
    n = len(x_nodes)

    # Первый член многочлена Ньютона — это f[x0]
    result = diff_table[0][0]

    # В этой переменной постепенно будет накапливаться произведение:
    # (x - x0), (x - x0)(x - x1), ...
    product = 1.0

    # Добавляем остальные члены многочлена Ньютона
    for order in range(1, n):
        # На каждом шаге домножаем ещё на один множитель
        product *= (x_star - x_nodes[order - 1])

        # Прибавляем очередной член:
        # f[x0, ..., x_order] * (x-x0)...(x-x_{order-1})
        result += diff_table[0][order] * product

    # Итог — значение P_n(x*) в точке x_star
    return result


def newton_polynomial_string(x_nodes, diff_table):
    # Начинаем запись многочлена Ньютона с первого члена f[x0]
    parts = [f"{diff_table[0][0]:.10f}"]

    # Формируем остальные слагаемые
    for order in range(1, len(x_nodes)):
        # Коэффициент — это разделённая разность нужного порядка
        term = f"{diff_table[0][order]:+.10f}"

        # Добавляем множители (x - x0)(x - x1)... по формуле Ньютона
        for j in range(order):
            term += f" * (x - {x_nodes[j]})"

        parts.append(term)

    # Возвращаем строковую запись многочлена
    return " ".join(parts)


# 7. Печать таблицы разделённых разностей
def print_divided_differences_table(x_nodes, diff_table):
    # Количество узлов
    n = len(x_nodes)

    print("Таблица разделённых разностей:")

    # Формируем заголовок таблицы.
    # Первый столбец — номер i,
    # второй — x_i,
    # третий — f[x_i],
    # далее столбцы с разделёнными разностями higher order.
    header = ["i", "x_i", "f[x_i]"]
    for order in range(1, n):
        if order == 1:
            header.append("f[x_i, x_{i+1}]")
        elif order == 2:
            header.append("f[x_i, x_{i+1}, x_{i+2}]")
        else:
            header.append("f[x_i, ..., x_{i+" + str(order) + "}]")

    print("-" * 120)
    print("{:<5} {:<12} {:<18} {:<22} {:<24} {:<24}".format(*header))
    print("-" * 120)

    # Построчно печатаем таблицу
    for i in range(n):
        # Начало строки: номер узла, сам узел и значение функции
        row = [str(i), f"{x_nodes[i]:.10f}", f"{diff_table[i][0]:.10f}"]

        # Добавляем разделённые разности.
        # Не все элементы в строке существуют, поэтому для недостающих
        # ячеек позже добавляем пустые строки.
        for order in range(1, n):
            if i < n - order:
                row.append(f"{diff_table[i][order]:.10f}")
            else:
                row.append("")

        # Чтобы форматирование print не ломалось, добиваем строку
        # до нужного количества столбцов
        while len(row) < 6:
            row.append("")

        print("{:<5} {:<12} {:<18} {:<22} {:<24} {:<24}".format(*row))

    print("-" * 120)


# 8. Решение одного пункта варианта
def solve_case(case_name, x_nodes, x_star):
    # Красивый заголовок для отдельного случая:
    # а) или б)
    print("=" * 120)
    print(f"СЛУЧАЙ {case_name}")
    print("=" * 120)

    # Шаг 1. Проверка узлов
    # Проверяем, что узлы корректны:
    # их достаточно и они не совпадают
    check_nodes(x_nodes)

    # Шаг 2. Вычисление значений функции в узлах
    # Получаем таблицу y_i = f(x_i)
    y_nodes = make_table(x_nodes)

    print("Таблица значений функции:")
    print("-" * 50)
    print("{:<5} {:<15} {:<20}".format("i", "x_i", "y_i = f(x_i)"))
    print("-" * 50)

    # Печатаем исходную таблицу узлов и значений
    for i in range(len(x_nodes)):
        print("{:<5} {:<15.10f} {:<20.10f}".format(i, x_nodes[i], y_nodes[i]))

    print("-" * 50)

    # Шаг 3. Построение многочлена Лагранжа
    # Получаем строковую запись многочлена и его значение в точке X*
    lagrange_poly_text = lagrange_polynomial_string(x_nodes, y_nodes)
    lagrange_at_x_star = lagrange_value(x_nodes, y_nodes, x_star)

    print("\nМногочлен Лагранжа:")
    print(f"L(x) = {lagrange_poly_text}")

    # Шаг 4. Построение таблицы разделённых разностей
    # Она нужна для построения многочлена Ньютона
    diff_table = divided_differences_table(x_nodes, y_nodes)
    print()
    print_divided_differences_table(x_nodes, diff_table)

    # Шаг 5. Построение многочлена Ньютона
    # Получаем строковую запись многочлена и его значение в точке X*
    newton_poly_text = newton_polynomial_string(x_nodes, diff_table)
    newton_at_x_star = newton_value(x_nodes, diff_table, x_star)

    print("\nМногочлен Ньютона:")
    print(f"P(x) = {newton_poly_text}")

    # Шаг 6. Вычисление точного значения и погрешности
    # Сначала считаем точное значение исходной функции в точке X*
    y_exact = f(x_star)

    # Затем считаем абсолютные погрешности для Лагранжа и Ньютона
    error_lagrange = fabs(y_exact - lagrange_at_x_star)
    error_newton = fabs(y_exact - newton_at_x_star)

    print("\nВычисления в точке X*:")
    print(f"X* = {x_star:.10f}")
    print(f"Точное значение y(X*) = f(X*) = {y_exact:.10f}")
    print(f"L(X*) = {lagrange_at_x_star:.10f}")
    print(f"P(X*) = {newton_at_x_star:.10f}")
    print(f"|y(X*) - L(X*)| = {error_lagrange:.10f}")
    print(f"|y(X*) - P(X*)| = {error_newton:.10f}")

    print("\nВывод:")
    print("Для одного и того же набора узлов многочлены Лагранжа и Ньютона")
    print("задают один и тот же интерполяционный многочлен, поэтому значения")
    print("L(X*) и P(X*) совпадают с точностью до округления.")
    print()


# 9. Главная программа
def main():
    # По условию точка, в которой нужно вычислить значение интерполяции
    # и погрешность
    x_star = 0.8

    # Вариант 24
    # а) первый набор узлов
    x_nodes_a = [0.1, 0.5, 0.9, 1.3]

    # б) второй набор узлов
    x_nodes_b = [0.1, 0.5, 1.1, 1.3]

    # Решаем пункт а)
    solve_case("а", x_nodes_a, x_star)

    # Решаем пункт б)
    solve_case("б", x_nodes_b, x_star)


# Стандартная точка входа в программу:
# если файл запускается напрямую, выполняется main()
if __name__ == "__main__":
    main()