# Лабораторная работа 3.4
# Численное дифференцирование таблично заданной функции
# Вариант 24
#
# Цель:
# Вычислить первую и вторую производную функции, заданной таблицей,
# в точке X* = 1.4.
#
# Решение выполняется строго по формулам:
# 1) формула (3.18) — первая производная на отрезке;
# 2) формула (3.20) — первая производная со вторым порядком точности;
# 3) формула (3.21) — вторая производная.


EPS = 1e-12


def check_input(xs, ys, x_star):
    """
    Проверка исходных данных.

    Эти проверки нужны, чтобы формулы из методички можно было применять корректно:
    - количество x и y должно совпадать;
    - должно быть минимум 3 точки, потому что для формул (3.20) и (3.21)
      нужен интерполяционный многочлен второй степени, то есть три соседних узла;
    - узлы x должны идти строго по возрастанию, иначе возможны нулевые
      или некорректные знаменатели;
    - точка X* должна совпадать с одним из внутренних узлов таблицы,
      как в примере из методички;
    - у точки X* должен быть сосед слева и сосед справа.
    """

    if len(xs) != len(ys):
        raise ValueError("Ошибка: количество значений x и y должно совпадать.")

    if len(xs) < 3:
        raise ValueError("Ошибка: для вычислений нужно минимум 3 табличные точки.")

    for i in range(len(xs) - 1):
        if xs[i] >= xs[i + 1]:
            raise ValueError("Ошибка: значения x должны быть строго возрастающими.")

    target_index = None

    for i in range(len(xs)):
        if abs(xs[i] - x_star) < EPS:
            target_index = i
            break

    if target_index is None:
        raise ValueError("Ошибка: точка X* должна совпадать с одним из узлов таблицы.")

    if target_index == 0 or target_index == len(xs) - 1:
        raise ValueError(
            "Ошибка: точка X* не должна быть крайней, "
            "так как нужны соседний узел слева и соседний узел справа."
        )

    return target_index


def check_uniform_grid(xs):
    """
    Проверка равномерности сетки.

    В методичке указано, что формула (3.20) при равностоящих точках
    обеспечивает второй порядок точности.
    Для варианта 24 сетка равномерная: шаг h = 0.2.
    """

    h = xs[1] - xs[0]

    for i in range(1, len(xs) - 1):
        current_h = xs[i + 1] - xs[i]
        if abs(current_h - h) > EPS:
            return False, h

    return True, h


def calculate_derivatives(xs, ys, x_star):
    """
    Вычисление первой и второй производной в точке X*.

    Пусть X* = x_{i+1}.
    Тогда для формул берем три соседние точки:

    x_i     — точка слева от X*
    x_{i+1} — сама точка X*
    x_{i+2} — точка справа от X*

    Для моего варианта:
    x_i     = 1.2
    x_{i+1} = 1.4
    x_{i+2} = 1.6
    """

    target_index = check_input(xs, ys, x_star)

    # Так как X* = x_{i+1}, то индекс i равен target_index - 1.
    i = target_index - 1

    # Берем три соседние точки для формул (3.20) и (3.21).
    x_i = xs[i]
    x_i1 = xs[i + 1]
    x_i2 = xs[i + 2]

    y_i = ys[i]
    y_i1 = ys[i + 1]
    y_i2 = ys[i + 2]

    # Формула (3.18)
    # Левая производная:
    # y'(X*) ≈ (y_{i+1} - y_i) / (x_{i+1} - x_i)
    # Для моего варианта это производная на отрезке [1.2; 1.4].
    left_derivative = (y_i1 - y_i) / (x_i1 - x_i)

    # Правая производная:
    # y'(X*) ≈ (y_{i+2} - y_{i+1}) / (x_{i+2} - x_{i+1})
    #
    # Для моего варианта это производная на отрезке [1.4; 1.6].
    right_derivative = (y_i2 - y_i1) / (x_i2 - x_i1)

    # Формула (3.20)
    # Первая производная со вторым порядком точности.
    # В методичке формула имеет вид:
    # y'(x) ≈ (y_{i+1} - y_i) / (x_{i+1} - x_i)
    #       + [ ((y_{i+2} - y_{i+1}) / (x_{i+2} - x_{i+1})
    #          - (y_{i+1} - y_i) / (x_{i+1} - x_i))
    #          / (x_{i+2} - x_i) ]
    #       * (2x - x_i - x_{i+1})
    #
    # Здесь x = X*.
    first_derivative_second_order = left_derivative + (
        (right_derivative - left_derivative) / (x_i2 - x_i)
    ) * (2 * x_star - x_i - x_i1)

    # Формула (3.21)
    # Вторая производная.
    # y''(x) ≈ 2 * [ ((y_{i+2} - y_{i+1}) / (x_{i+2} - x_{i+1})
    #              - (y_{i+1} - y_i) / (x_{i+1} - x_i))
    #              / (x_{i+2} - x_i) ]
    second_derivative = 2 * (
        (right_derivative - left_derivative) / (x_i2 - x_i)
    )

    return {
        "i": i,
        "x_i": x_i,
        "x_i1": x_i1,
        "x_i2": x_i2,
        "y_i": y_i,
        "y_i1": y_i1,
        "y_i2": y_i2,
        "left_derivative": left_derivative,
        "right_derivative": right_derivative,
        "first_derivative_second_order": first_derivative_second_order,
        "second_derivative": second_derivative,
    }


def main():
    x_star = 1.4

    xs = [1.0, 1.2, 1.4, 1.6, 1.8]
    ys = [2.0, 2.1344, 2.4702, 2.9506, 3.5486]

    print("Лабораторная работа 3.4")
    print("Тема: численное дифференцирование")
    print("Вариант: 24")
    print()

    print("Исходные данные:")
    print(f"X* = {x_star}")
    print()

    print("Таблица значений функции:")
    print("i        x_i        y_i")
    print("--------------------------")
    for i in range(len(xs)):
        print(f"{i:<8} {xs[i]:<10.4f} {ys[i]:<10.4f}")

    print()

    is_uniform, h = check_uniform_grid(xs)

    if is_uniform:
        print(f"Проверка сетки: сетка равномерная, h = {h:.4f}")
    else:
        print("Проверка сетки: сетка неравномерная")

    print()

    result = calculate_derivatives(xs, ys, x_star)

    print("Для вычислений выбраны три соседние точки:")
    print(f"x_i     = {result['x_i']:.4f}, y_i     = {result['y_i']:.4f}")
    print(f"x_i+1   = {result['x_i1']:.4f}, y_i+1   = {result['y_i1']:.4f}")
    print(f"x_i+2   = {result['x_i2']:.4f}, y_i+2   = {result['y_i2']:.4f}")
    print()

    print("Вычисление по формуле (3.18):")
    print(
        "Левосторонняя производная на отрезке "
        f"[{result['x_i']:.4f}; {result['x_i1']:.4f}]: "
        f"{result['left_derivative']:.6f}"
    )
    print(
        "Правосторонняя производная на отрезке "
        f"[{result['x_i1']:.4f}; {result['x_i2']:.4f}]: "
        f"{result['right_derivative']:.6f}"
    )
    print()

    print("Вычисление первой производной со вторым порядком точности по формуле (3.20):")
    print(f"y'({x_star}) ≈ {result['first_derivative_second_order']:.6f}")
    print()

    print("Вычисление второй производной по формуле (3.21):")
    print(f"y''({x_star}) ≈ {result['second_derivative']:.6f}")


if __name__ == "__main__":
    main()