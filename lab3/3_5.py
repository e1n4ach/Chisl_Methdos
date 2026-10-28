
import math


# ============================================================
# Лабораторная работа 3.5
# Численное интегрирование
# Вариант 24
#
# Нужно вычислить:
#     integral from -2 to 2 of sqrt(16 - x^2) dx
#
# Нужно:
# 1) метод прямоугольников;
# 2) метод трапеций;
# 3) метод Симпсона;
# 4) оценку/уточнение по методу Рунге-Ромберга.
# ============================================================


def f(x):
    """
    Подынтегральная функция варианта 24:
        y = sqrt(16 - x^2)
    """
    return math.sqrt(16 - x ** 2)


def build_grid(x0, xk, h):
    """
    Строим равномерную сетку x_i = x0 + i*h.

    В методичке для формул численного интегрирования используется
    постоянный шаг h, поэтому проверяем, что отрезок [x0, xk]
    делится на целое число частей.
    """
    n_float = (xk - x0) / h
    n = int(round(n_float))

    # Техническая проверка корректности шага:
    # количество интервалов должно быть целым.
    if abs(n_float - n) > 1e-12:
        raise ValueError("Шаг h не делит отрезок [X0, Xk] на целое число частей")

    return [x0 + i * h for i in range(n + 1)]


def rectangle_method(x0, xk, h):
    """
    Метод прямоугольников по формуле:

        F ≈ h * sum f((x_{i-1} + x_i) / 2)

    То есть на каждом отрезке [x_{i-1}, x_i]
    функция заменяется значением в середине отрезка.
    """
    x = build_grid(x0, xk, h)
    n = len(x) - 1

    result = 0.0

    # Идем по всем частичным отрезкам [x_{i-1}, x_i]
    for i in range(1, n + 1):
        middle = (x[i - 1] + x[i]) / 2
        result += h * f(middle)

    return result


def trapezoid_method(x0, xk, h):
    """
    Метод трапеций по формуле:

        F ≈ h * (f_0 / 2 + f_1 + ... + f_{N-1} + f_N / 2)

    То есть на каждом отрезке [x_{i-1}, x_i]
    функция заменяется прямой, проходящей через концы отрезка.
    """
    x = build_grid(x0, xk, h)
    n = len(x) - 1

    result = f(x[0]) / 2 + f(x[n]) / 2

    # Внутренние узлы входят с коэффициентом 1
    for i in range(1, n):
        result += f(x[i])

    result *= h
    return result


def simpson_method(x0, xk, h):
    """
    Метод Симпсона по формуле:

        F ≈ h/3 * (f_0 + 4f_1 + 2f_2 + 4f_3 + ... + 2f_{N-2} + 4f_{N-1} + f_N)

    Важно:
    метод Симпсона требует четное количество интервалов,
    потому что парабола строится на каждой паре соседних отрезков.
    """
    x = build_grid(x0, xk, h)
    n = len(x) - 1

    # Проверка из условия применимости метода Симпсона:
    # число интервалов должно быть четным.
    if n % 2 != 0:
        raise ValueError("Для метода Симпсона число интервалов должно быть четным")

    result = f(x[0]) + f(x[n])

    # Внутренние узлы:
    # нечетные индексы имеют коэффициент 4,
    # четные индексы имеют коэффициент 2.
    for i in range(1, n):
        if i % 2 == 1:
            result += 4 * f(x[i])
        else:
            result += 2 * f(x[i])

    result *= h / 3
    return result


def runge_romberg(F_h, F_kh, p, k):
    """
    Метод Рунге-Ромберга по формуле методички:

        F ≈ F_h + (F_h - F_kh) / (k^p - 1)

    Здесь:
        F_h  — значение на более мелкой сетке;
        F_kh — значение на более крупной сетке;
        k    — во сколько раз крупный шаг больше мелкого;
        p    — порядок точности метода.

    Для нашей работы:
        k = h1 / h2 = 1.0 / 0.5 = 2

    Порядки точности:
        p = 2 для метода прямоугольников;
        p = 2 для метода трапеций;
        p = 4 для метода Симпсона.
    """
    return F_h + (F_h - F_kh) / (k ** p - 1)


def exact_integral_for_check():
    """
    Точное значение интеграла используем только для проверки результата.

    integral sqrt(16 - x^2) dx =
        x/2 * sqrt(16 - x^2) + 8 * arcsin(x / 4)

    На отрезке [-2, 2]:
        F_exact = 4*sqrt(3) + 8*pi/3
    """
    return 4 * math.sqrt(3) + 8 * math.pi / 3


def print_table_for_step(x0, xk, h):
    """
    Печатаем таблицу, как таблицу из примера методички:
    x_i, y_i и накопленные значения интеграла от X0 до x_i
    для трех методов.

    Для метода Симпсона значение печатается только в четных узлах,
    потому что формула Симпсона работает на парах интервалов.
    """
    x = build_grid(x0, xk, h)
    n = len(x) - 1

    print()
    print(f"Таблица вычислений для h = {h}")
    print("-" * 86)
    print(f"{'i':>2} {'x_i':>12} {'y_i':>14} {'прямоуг.':>14} {'трапеций':>14} {'Симпсона':>14}")
    print("-" * 86)

    rect_sum = 0.0
    trap_sum = 0.0

    for i in range(n + 1):
        xi = x[i]
        yi = f(xi)

        # В нулевом узле интеграл от X0 до X0 равен 0.
        if i == 0:
            rect_str = f"{0.0:.8f}"
            trap_str = f"{0.0:.8f}"
            simp_str = f"{0.0:.8f}"
        else:
            # Добавляем вклад текущего отрезка [x_{i-1}, x_i]
            middle = (x[i - 1] + x[i]) / 2
            rect_sum += h * f(middle)
            trap_sum += h * (f(x[i - 1]) + f(x[i])) / 2

            rect_str = f"{rect_sum:.8f}"
            trap_str = f"{trap_sum:.8f}"

            # Симпсон считается только после каждой пары интервалов.
            if i % 2 == 0:
                simp_value = simpson_method(x0, xi, h)
                simp_str = f"{simp_value:.8f}"
            else:
                simp_str = "—"

        print(f"{i:>2} {xi:>12.6f} {yi:>14.8f} {rect_str:>14} {trap_str:>14} {simp_str:>14}")

    print("-" * 86)


def main():
    # Исходные данные варианта 24
    x0 = -2.0
    xk = 2.0
    h1 = 1.0
    h2 = 0.5

    # Проверяем, что для метода Симпсона количество интервалов четное
    # при обоих шагах.
    for h in (h1, h2):
        n = int(round((xk - x0) / h))
        if n % 2 != 0:
            raise ValueError(f"При h = {h} число интервалов N = {n}, оно нечетное")

    print("ЛР 3.5, вариант 24")
    print("Интеграл: integral from -2 to 2 of sqrt(16 - x^2) dx")
    print(f"X0 = {x0}, Xk = {xk}, h1 = {h1}, h2 = {h2}")

    # Таблицы для двух шагов, как в примере методички
    print_table_for_step(x0, xk, h1)
    print_table_for_step(x0, xk, h2)

    # Вычисления тремя методами на крупном и мелком шагах
    rect_h1 = rectangle_method(x0, xk, h1)
    rect_h2 = rectangle_method(x0, xk, h2)

    trap_h1 = trapezoid_method(x0, xk, h1)
    trap_h2 = trapezoid_method(x0, xk, h2)

    simp_h1 = simpson_method(x0, xk, h1)
    simp_h2 = simpson_method(x0, xk, h2)

    # В методе Рунге-Ромберга:
    # F_h  — значение на мелком шаге h2,
    # F_kh — значение на крупном шаге h1,
    # k = h1 / h2.
    k = h1 / h2

    rect_rr = runge_romberg(rect_h2, rect_h1, p=2, k=k)
    trap_rr = runge_romberg(trap_h2, trap_h1, p=2, k=k)
    simp_rr = runge_romberg(simp_h2, simp_h1, p=4, k=k)

    # Оценка погрешности по Рунге-Ромбергу:
    # это модуль поправки, то есть |F_RR - F_h|.
    rect_error_estimate = abs(rect_rr - rect_h2)
    trap_error_estimate = abs(trap_rr - trap_h2)
    simp_error_estimate = abs(simp_rr - simp_h2)

    exact = exact_integral_for_check()

    print()
    print("Итоговые значения интеграла")
    print("-" * 78)
    print(f"{'Метод':<22} {'h1 = 1.0':>14} {'h2 = 0.5':>14} {'Рунге-Ромберг':>18}")
    print("-" * 78)
    print(f"{'Прямоугольников':<22} {rect_h1:>14.8f} {rect_h2:>14.8f} {rect_rr:>18.8f}")
    print(f"{'Трапеций':<22} {trap_h1:>14.8f} {trap_h2:>14.8f} {trap_rr:>18.8f}")
    print(f"{'Симпсона':<22} {simp_h1:>14.8f} {simp_h2:>14.8f} {simp_rr:>18.8f}")
    print("-" * 78)

    print()
    print("Оценка погрешности по Рунге-Ромбергу")
    print("-" * 78)
    print(f"{'Метод':<22} {'|F_RR - F_h2|':>20}")
    print("-" * 78)
    print(f"{'Прямоугольников':<22} {rect_error_estimate:>20.8f}")
    print(f"{'Трапеций':<22} {trap_error_estimate:>20.8f}")
    print(f"{'Симпсона':<22} {simp_error_estimate:>20.8f}")
    print("-" * 78)

    print()
    print("Проверка по точному значению")
    print("-" * 78)
    print(f"Точное значение: {exact:.8f}")
    print(f"Абсолютная ошибка метода прямоугольников после Рунге-Ромберга: {abs(rect_rr - exact):.8f}")
    print(f"Абсолютная ошибка метода трапеций после Рунге-Ромберга:       {abs(trap_rr - exact):.8f}")
    print(f"Абсолютная ошибка метода Симпсона после Рунге-Ромберга:       {abs(simp_rr - exact):.8f}")
    print("-" * 78)


if __name__ == "__main__":
    main()
