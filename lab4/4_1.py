import math

# ------------------------------------------------------------
# Лабораторная работа 4.1
# Вариант 24
# Численные методы решения задачи Коши для ОДУ 2-го порядка:
#
#     x^2 * y'' + (x + 1) * y' - y = 0
#     y(1) = 2 + e
#     y'(1) = 1
#     x in [1, 2], h = 0.1
#
# Точное решение:
#     y(x) = x + 1 + x * e^(1/x)
#
# По методичке ОДУ 2-го порядка сводим к системе двух ОДУ 1-го порядка:
#     z = y'
#     y' = z
#     z' = (y - (x + 1) * z) / x^2
# ------------------------------------------------------------


def exact_y(x):
    """Точное решение y(x) для оценки погрешности."""
    return x + 1.0 + x * math.exp(1.0 / x)


def exact_z(x):
    """Точная производная y'(x), нужна только для контроля z."""
    return 1.0 + math.exp(1.0 / x) - math.exp(1.0 / x) / x


def f(x, y, z):
    """Правая часть первого уравнения системы: y' = z."""
    return z


def g(x, y, z):
    """Правая часть второго уравнения системы: z' = y''."""
    return (y - (x + 1.0) * z) / (x * x)


def euler_method(a, b, h, y0, z0):
    """
    Явный метод Эйлера для системы двух ОДУ 1-го порядка.

    По методичке для одного уравнения:
        y_{k+1} = y_k + h * f(x_k, y_k)

    Для системы применяем ту же формулу к каждой компоненте:
        y_{k+1} = y_k + h * f(x_k, y_k, z_k)
        z_{k+1} = z_k + h * g(x_k, y_k, z_k)
    """
    n = int(round((b - a) / h))
    x_values = [a]
    y_values = [y0]
    z_values = [z0]

    for k in range(n):
        xk = x_values[k]
        yk = y_values[k]
        zk = z_values[k]

        # Сначала считаем приращения строго из левой точки интервала.
        dy = h * f(xk, yk, zk)
        dz = h * g(xk, yk, zk)

        # Затем переходим в следующий узел сетки.
        x_values.append(xk + h)
        y_values.append(yk + dy)
        z_values.append(zk + dz)

    return x_values, y_values, z_values


def rk4_step(xk, yk, zk, h):
    """
    Один шаг метода Рунге-Кутты 4-го порядка для системы.

    Обозначения соответствуют формулам методички для системы:
        K_i — приращения для y,
        L_i — приращения для z.
    """
    K1 = h * f(xk, yk, zk)
    L1 = h * g(xk, yk, zk)

    K2 = h * f(xk + h / 2.0, yk + K1 / 2.0, zk + L1 / 2.0)
    L2 = h * g(xk + h / 2.0, yk + K1 / 2.0, zk + L1 / 2.0)

    K3 = h * f(xk + h / 2.0, yk + K2 / 2.0, zk + L2 / 2.0)
    L3 = h * g(xk + h / 2.0, yk + K2 / 2.0, zk + L2 / 2.0)

    K4 = h * f(xk + h, yk + K3, zk + L3)
    L4 = h * g(xk + h, yk + K3, zk + L3)

    dy = (K1 + 2.0 * K2 + 2.0 * K3 + K4) / 6.0
    dz = (L1 + 2.0 * L2 + 2.0 * L3 + L4) / 6.0

    theta_y = None
    theta_z = None

    # Контроль правильности выбора шага h по методичке для РК4:
    # theta_1 = (K3 - K2) / (K2 - K1), theta_2 = (L3 - L2) / (L2 - L1).
    # Если знаменатель близок к нулю, параметр не считаем, чтобы не делить на ноль.
    if abs(K2 - K1) > 1e-15:
        theta_y = abs((K3 - K2) / (K2 - K1))
    if abs(L2 - L1) > 1e-15:
        theta_z = abs((L3 - L2) / (L2 - L1))

    return dy, dz, theta_y, theta_z


def runge_kutta_4_method(a, b, h, y0, z0):
    """Метод Рунге-Кутты 4-го порядка для системы двух ОДУ."""
    n = int(round((b - a) / h))
    x_values = [a]
    y_values = [y0]
    z_values = [z0]
    theta_values = []

    for k in range(n):
        xk = x_values[k]
        yk = y_values[k]
        zk = z_values[k]

        dy, dz, theta_y, theta_z = rk4_step(xk, yk, zk, h)

        x_values.append(xk + h)
        y_values.append(yk + dy)
        z_values.append(zk + dz)
        theta_values.append((theta_y, theta_z))

    return x_values, y_values, z_values, theta_values


def adams_4_method(a, b, h, y0, z0):
    """
    Явный метод Адамса 4-го порядка для системы.

    Метод Адамса не является самостартующим, поэтому первые 4 узла
    x0, x1, x2, x3 получаем методом Рунге-Кутты 4-го порядка.

    Формула Адамса 4-го порядка из методички:
        y_{k+1} = y_k + h/24 * (55 f_k - 59 f_{k-1} + 37 f_{k-2} - 9 f_{k-3})

    Для системы отдельно применяем ее к y и z:
        для y используем f = y' = z,
        для z используем g = z'.
    """
    n = int(round((b - a) / h))

    if n < 4:
        raise ValueError("Для метода Адамса 4-го порядка нужно минимум 4 шага сетки.")

    x_values = [a]
    y_values = [y0]
    z_values = [z0]

    # Стартовые значения y1, y2, y3 получаем методом Рунге-Кутты 4-го порядка.
    for k in range(3):
        xk = x_values[k]
        yk = y_values[k]
        zk = z_values[k]

        dy, dz, theta_y, theta_z = rk4_step(xk, yk, zk, h)

        x_values.append(xk + h)
        y_values.append(yk + dy)
        z_values.append(zk + dz)

    # Далее применяем многошаговую формулу Адамса.
    for k in range(3, n):
        xk = x_values[k]
        yk = y_values[k]
        zk = z_values[k]

        f_k = f(x_values[k], y_values[k], z_values[k])
        f_k1 = f(x_values[k - 1], y_values[k - 1], z_values[k - 1])
        f_k2 = f(x_values[k - 2], y_values[k - 2], z_values[k - 2])
        f_k3 = f(x_values[k - 3], y_values[k - 3], z_values[k - 3])

        g_k = g(x_values[k], y_values[k], z_values[k])
        g_k1 = g(x_values[k - 1], y_values[k - 1], z_values[k - 1])
        g_k2 = g(x_values[k - 2], y_values[k - 2], z_values[k - 2])
        g_k3 = g(x_values[k - 3], y_values[k - 3], z_values[k - 3])

        dy = h / 24.0 * (55.0 * f_k - 59.0 * f_k1 + 37.0 * f_k2 - 9.0 * f_k3)
        dz = h / 24.0 * (55.0 * g_k - 59.0 * g_k1 + 37.0 * g_k2 - 9.0 * g_k3)

        x_values.append(xk + h)
        y_values.append(yk + dy)
        z_values.append(zk + dz)

    return x_values, y_values, z_values


def runge_romberg_errors(y_h, y_h2, p):
    """
    Оценка погрешности по методу Рунге-Ромберга.

    y_h  — решение на сетке с шагом h,
    y_h2 — решение на сетке с шагом h/2,
    p    — порядок точности метода.

    Сравниваем значения в одинаковых узлах:
        x_k на крупной сетке соответствует x_{2k} на мелкой сетке.

    Оценка главного члена погрешности:
        R = |(y_{h/2} - y_h) / (2^p - 1)|
    """
    result = []
    denominator = 2.0 ** p - 1.0

    for k in range(len(y_h)):
        result.append(abs((y_h2[2 * k] - y_h[k]) / denominator))

    return result


def print_table(method_name, x_values, y_values, z_values, rr_errors=None, theta_values=None):
    """Печать итоговой таблицы для выбранного метода."""
    print("\n" + method_name)
    print("-" * 118)

    if rr_errors is None:
        print(f"{'k':>2} {'x_k':>8} {'y_k':>16} {'z_k':>16} {'y точн.':>16} {'|y-yточн|':>16}")
    else:
        print(f"{'k':>2} {'x_k':>8} {'y_k':>16} {'z_k':>16} {'y точн.':>16} {'|y-yточн|':>16} {'R_R':>16}")

    for k in range(len(x_values)):
        xk = x_values[k]
        y_exact = exact_y(xk)
        exact_error = abs(y_values[k] - y_exact)

        if rr_errors is None:
            print(f"{k:2d} {xk:8.4f} {y_values[k]:16.9f} {z_values[k]:16.9f} {y_exact:16.9f} {exact_error:16.9e}")
        else:
            print(f"{k:2d} {xk:8.4f} {y_values[k]:16.9f} {z_values[k]:16.9f} {y_exact:16.9f} {exact_error:16.9e} {rr_errors[k]:16.9e}")

    if theta_values is not None:
        print("\nКонтроль шага для метода Рунге-Кутты 4-го порядка:")
        print(f"{'k':>2} {'theta_y':>16} {'theta_z':>16} {'вывод':>35}")
        for k in range(len(theta_values)):
            theta_y, theta_z = theta_values[k]

            # По методичке: несколько сотых — нормально, >0.1 — шаг уменьшить, <0.01 — можно увеличить.
            max_theta = 0.0
            if theta_y is not None:
                max_theta = max(max_theta, theta_y)
            if theta_z is not None:
                max_theta = max(max_theta, theta_z)

            if max_theta > 0.1:
                conclusion = "шаг лучше уменьшить"
            elif max_theta < 0.01:
                conclusion = "шаг можно увеличить"
            else:
                conclusion = "шаг приемлем"

            theta_y_text = "-" if theta_y is None else f"{theta_y:.9f}"
            theta_z_text = "-" if theta_z is None else f"{theta_z:.9f}"
            print(f"{k:2d} {theta_y_text:>16} {theta_z_text:>16} {conclusion:>35}")


def print_max_errors(method_name, x_values, y_values, rr_errors):
    """Печать максимальной ошибки по точному решению и по Рунге-Ромбергу."""
    exact_errors = []
    for k in range(len(x_values)):
        exact_errors.append(abs(y_values[k] - exact_y(x_values[k])))

    print(f"\n{method_name}")
    print(f"Максимальная погрешность по точному решению: {max(exact_errors):.9e}")
    print(f"Максимальная оценка по Рунге-Ромбергу:     {max(rr_errors):.9e}")


def main():
    # Исходные данные варианта 24.
    a = 1.0
    b = 2.0
    h = 0.1
    h2 = h / 2.0
    y0 = 2.0 + math.e
    z0 = 1.0

    print("Лабораторная работа 4.1. Вариант 24")
    print("ОДУ: x^2*y'' + (x + 1)*y' - y = 0")
    print("Начальные условия: y(1)=2+e, y'(1)=1")
    print("Отрезок: [1, 2]")
    print("Основной шаг h =", h)
    print("Шаг для Рунге-Ромберга h/2 =", h2)
    print("Точное решение: y = x + 1 + x*e^(1/x)")

    # Решения с основным шагом h.
    x_e, y_e, z_e = euler_method(a, b, h, y0, z0)
    x_rk, y_rk, z_rk, theta_rk = runge_kutta_4_method(a, b, h, y0, z0)
    x_ad, y_ad, z_ad = adams_4_method(a, b, h, y0, z0)

    # Решения с половинным шагом h/2 для оценки Рунге-Ромберга.
    x_e2, y_e2, z_e2 = euler_method(a, b, h2, y0, z0)
    x_rk2, y_rk2, z_rk2, theta_rk2 = runge_kutta_4_method(a, b, h2, y0, z0)
    x_ad2, y_ad2, z_ad2 = adams_4_method(a, b, h2, y0, z0)

    # Порядки точности методов.
    rr_e = runge_romberg_errors(y_e, y_e2, p=1)
    rr_rk = runge_romberg_errors(y_rk, y_rk2, p=4)
    rr_ad = runge_romberg_errors(y_ad, y_ad2, p=4)

    print_table("Метод Эйлера", x_e, y_e, z_e, rr_e)
    print_table("Метод Рунге-Кутты 4-го порядка", x_rk, y_rk, z_rk, rr_rk, theta_rk)
    print_table("Метод Адамса 4-го порядка", x_ad, y_ad, z_ad, rr_ad)

    print("\nИтоговые максимальные погрешности")
    print("=" * 60)
    print_max_errors("Метод Эйлера", x_e, y_e, rr_e)
    print_max_errors("Метод Рунге-Кутты 4-го порядка", x_rk, y_rk, rr_rk)
    print_max_errors("Метод Адамса 4-го порядка", x_ad, y_ad, rr_ad)


if __name__ == "__main__":
    main()
