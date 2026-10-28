import math

# Лабораторная работа 4.2, вариант 24
# Численные методы решения краевой задачи для ОДУ второго порядка.
# Требуется реализовать:
# 1) метод стрельбы;
# 2) конечно-разностный метод;
# 3) оценку погрешности по Рунге--Ромбергу;
# 4) сравнение с точным решением.

# -----------------------------------------------------------------------------
# Вариант 24:
#     (x^2 + 1) * y'' - 2y = 0,
#     y'(0) = 2,
#     y(1) = 3 + pi/2.
#
# Точное решение:
#     y(x) = x^2 + x + 1 + (x^2 + 1) * arctg(x).
#
# Для метода стрельбы сводим ОДУ второго порядка к системе первого порядка:
#     y' = z,
#     z' = 2y / (x^2 + 1).
# Так как на левом конце задано y'(0), а не y(0), неизвестным параметром
# стрельбы является eta = y(0). Подбираем eta методом секущих так, чтобы
# выполнялось правое граничное условие y(1) = 3 + pi/2.
# -----------------------------------------------------------------------------

A = 0.0
B = 1.0
Y_DERIV_A = 2.0
Y_B = 3.0 + math.pi / 2.0
EPS = 1e-6


def exact_solution(x):
    """Точное решение варианта 24."""
    return x * x + x + 1.0 + (x * x + 1.0) * math.atan(x)


def f_for_y(x, y, z):
    """Правая часть первого уравнения системы: y' = z."""
    return z


def f_for_z(x, y, z):
    """Правая часть второго уравнения системы: z' = 2y / (x^2 + 1)."""
    return 2.0 * y / (x * x + 1.0)


def runge_kutta_4_system(eta, h):
    """
    Решение задачи Коши методом Рунге-Кутты 4-го порядка по формулам методички.

    Начальные условия для текущего значения параметра стрельбы:
        y(0) = eta,
        z(0) = y'(0) = 2.

    Возвращает список точек (x_k, y_k, z_k).
    """
    n = int(round((B - A) / h))
    x = A
    y = eta
    z = Y_DERIV_A

    result = [(x, y, z)]

    for _ in range(n):
        # Формулы Рунге-Кутты 4-го порядка для системы:
        # K_i отвечают за приращение y, L_i отвечают за приращение z.
        k1 = h * f_for_y(x, y, z)
        l1 = h * f_for_z(x, y, z)

        k2 = h * f_for_y(x + h / 2.0, y + k1 / 2.0, z + l1 / 2.0)
        l2 = h * f_for_z(x + h / 2.0, y + k1 / 2.0, z + l1 / 2.0)

        k3 = h * f_for_y(x + h / 2.0, y + k2 / 2.0, z + l2 / 2.0)
        l3 = h * f_for_z(x + h / 2.0, y + k2 / 2.0, z + l2 / 2.0)

        k4 = h * f_for_y(x + h, y + k3, z + l3)
        l4 = h * f_for_z(x + h, y + k3, z + l3)

        delta_y = (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        delta_z = (l1 + 2.0 * l2 + 2.0 * l3 + l4) / 6.0

        y = y + delta_y
        z = z + delta_z
        x = x + h

        # Округление x защищает от накопления вида 0.30000000000000004.
        result.append((round(x, 12), y, z))

    return result


def phi_for_shooting(eta, h):
    """
    Алгоритмическая функция Phi(eta) из метода стрельбы.
    Нужно добиться Phi(eta) = 0, то есть y(b, eta) - y_b = 0.
    """
    solution = runge_kutta_4_system(eta, h)
    y_at_b = solution[-1][1]
    return y_at_b - Y_B


def shooting_method(h, eta0=0.0, eta1=2.0, eps=EPS, max_iter=100):
    """
    Метод стрельбы + метод секущих.

    В методичке для параметра стрельбы используется итерация секущих:
        eta_{j+2} = eta_{j+1} - (eta_{j+1} - eta_j) * Phi(eta_{j+1}) /
                    (Phi(eta_{j+1}) - Phi(eta_j)).

    Критерий остановки:
        |Phi(eta_j)| <= eps,
    то есть правое граничное условие выполнено с заданной точностью.
    """
    phi0 = phi_for_shooting(eta0, h)
    phi1 = phi_for_shooting(eta1, h)

    iterations = [
        (0, eta0, runge_kutta_4_system(eta0, h)[-1][1], phi0),
        (1, eta1, runge_kutta_4_system(eta1, h)[-1][1], phi1),
    ]

    if abs(phi0) <= eps:
        return eta0, runge_kutta_4_system(eta0, h), iterations
    if abs(phi1) <= eps:
        return eta1, runge_kutta_4_system(eta1, h), iterations

    for j in range(max_iter):
        denominator = phi1 - phi0
        if abs(denominator) < 1e-14:
            raise ZeroDivisionError("Метод секущих остановлен: Phi(eta_j+1) - Phi(eta_j) слишком мало.")

        eta2 = eta1 - (eta1 - eta0) * phi1 / denominator
        phi2 = phi_for_shooting(eta2, h)
        y_at_b = runge_kutta_4_system(eta2, h)[-1][1]
        iterations.append((j + 2, eta2, y_at_b, phi2))

        # Проверка критерия остановки строго после получения нового приближения.
        if abs(phi2) <= eps:
            return eta2, runge_kutta_4_system(eta2, h), iterations

        eta0, phi0 = eta1, phi1
        eta1, phi1 = eta2, phi2

    raise RuntimeError("Метод стрельбы не сошелся за заданное число итераций.")


def p_coef(x):
    """Коэффициент p(x) в форме y'' + p(x)y' + q(x)y = f(x)."""
    return 0.0


def q_coef(x):
    """Коэффициент q(x) после деления исходного уравнения на x^2 + 1."""
    return -2.0 / (x * x + 1.0)


def right_side(x):
    """Правая часть f(x) после деления исходного уравнения на x^2 + 1."""
    return 0.0


def thomas_algorithm(a, b, c, d):
    """
    Метод прогонки для трехдиагональной системы.

    a[i] -- коэффициент при y_{i-1};
    b[i] -- коэффициент при y_i;
    c[i] -- коэффициент при y_{i+1};
    d[i] -- правая часть.
    """
    n = len(d)

    # Проверка размерностей массивов.
    if not (len(a) == len(b) == len(c) == n):
        raise ValueError("Размеры массивов коэффициентов трехдиагональной системы не совпадают.")

    # Прямой ход прогонки.
    alpha = [0.0] * n
    beta = [0.0] * n

    if abs(b[0]) < 1e-14:
        raise ZeroDivisionError("Нулевой первый диагональный элемент в методе прогонки.")

    alpha[0] = -c[0] / b[0]
    beta[0] = d[0] / b[0]

    for i in range(1, n):
        denominator = b[i] + a[i] * alpha[i - 1]
        if abs(denominator) < 1e-14:
            raise ZeroDivisionError("Нулевой знаменатель в прямом ходе метода прогонки.")

        alpha[i] = -c[i] / denominator if i < n - 1 else 0.0
        beta[i] = (d[i] - a[i] * beta[i - 1]) / denominator

    # Обратный ход прогонки.
    y = [0.0] * n
    y[n - 1] = beta[n - 1]

    for i in range(n - 2, -1, -1):
        y[i] = alpha[i] * y[i + 1] + beta[i]

    return y


def check_diagonal_dominance(lower, diag, upper):
    """
    Проверка преобладания диагональных элементов.

    В методичке отдельно отмечено, что при достаточно малых h и q(x_k)<0
    для трехдиагональной системы выполняется условие преобладания диагональных
    элементов, что гарантирует корректное применение метода прогонки.
    Здесь проверяем его численно для построенной системы.
    """
    for i in range(len(diag)):
        left = abs(lower[i])
        middle = abs(diag[i])
        right = abs(upper[i])
        if middle + 1e-14 < left + right:
            return False
    return True


def finite_difference_method(h):
    """
    Конечно-разностный метод.

    Методичка приводит центральные разности для внутренних узлов:
        y'_k  ≈ (y_{k+1} - y_{k-1}) / (2h),
        y''_k ≈ (y_{k+1} - 2y_k + y_{k-1}) / h^2.

    Для условия второго рода y'(0)=2 используем одностороннюю разность 1-го порядка
    из методички, чтобы сохранить трехдиагональную матрицу:
        (y_1 - y_0) / h = 2.

    Правое условие первого рода:
        y_N = 3 + pi/2.
    """
    n = int(round((B - A) / h))
    xs = [A + i * h for i in range(n + 1)]

    # Система строится для всех неизвестных y_0, y_1, ..., y_N.
    lower = [0.0] * (n + 1)
    diag = [0.0] * (n + 1)
    upper = [0.0] * (n + 1)
    rhs = [0.0] * (n + 1)

    # Левая граница: (y_1 - y_0) / h = 2  =>  -y_0 + y_1 = 2h.
    diag[0] = -1.0
    upper[0] = 1.0
    rhs[0] = Y_DERIV_A * h

    # Внутренние узлы: формула (4.38) после умножения на h^2:
    # (1 - p_k h/2)y_{k-1} + (-2 + h^2 q_k)y_k + (1 + p_k h/2)y_{k+1} = h^2 f_k.
    for k in range(1, n):
        xk = xs[k]
        pk = p_coef(xk)
        qk = q_coef(xk)
        fk = right_side(xk)

        lower[k] = 1.0 - pk * h / 2.0
        diag[k] = -2.0 + h * h * qk
        upper[k] = 1.0 + pk * h / 2.0
        rhs[k] = h * h * fk

    # Правая граница: y_N = 3 + pi/2.
    diag[n] = 1.0
    rhs[n] = Y_B

    # Проверка условия преобладания диагональных элементов перед прогонкой.
    # Для данной задачи q(x)<0, поэтому на внутренних узлах условие выполняется;
    # граничные строки также остаются корректными для метода прогонки.
    if not check_diagonal_dominance(lower, diag, upper):
        print("Предупреждение: диагональное преобладание не выполнено для h =", h)

    ys = thomas_algorithm(lower, diag, upper, rhs)
    return list(zip(xs, ys))


def runge_romberg_error(coarse_solution, fine_solution, order):
    """
    Оценка погрешности по Рунге--Ромбергу на узлах крупной сетки.

    coarse_solution -- решение с шагом h;
    fine_solution   -- решение с шагом h/2;
    order           -- порядок метода.

    На одинаковых узлах берем каждую вторую точку решения с шагом h/2.
    Оценка: |y_{h/2} - y_h| / (2^p - 1).
    """
    result = []
    denominator = 2.0 ** order - 1.0

    for i in range(len(coarse_solution)):
        x = coarse_solution[i][0]
        y_coarse = coarse_solution[i][1]
        y_fine = fine_solution[2 * i][1]
        error = abs(y_fine - y_coarse) / denominator
        result.append((x, error))

    return result


def print_iterations(iterations):
    print("\nИтерации метода секущих в методе стрельбы")
    print("j        eta_j              y(b, eta_j)        Phi(eta_j)")
    print("-" * 70)
    for j, eta, yb, phi in iterations:
        print(f"{j:<3d} {eta:>18.10f} {yb:>18.10f} {phi:>18.10e}")


def print_solution_table(title, solution, rr_errors=None):
    print("\n" + title)
    if rr_errors is None:
        print("k      x_k          y_k              y_exact          |y_exact-y_k|")
        print("-" * 78)
        for k, (x, y) in enumerate(solution):
            y_exact = exact_solution(x)
            print(f"{k:<3d} {x:>8.4f} {y:>16.10f} {y_exact:>16.10f} {abs(y_exact - y):>16.10e}")
    else:
        print("k      x_k          y_k              y_exact          |exact-y|        RR")
        print("-" * 95)
        for k, (x, y) in enumerate(solution):
            y_exact = exact_solution(x)
            rr = rr_errors[k][1]
            print(f"{k:<3d} {x:>8.4f} {y:>16.10f} {y_exact:>16.10f} {abs(y_exact - y):>16.10e} {rr:>16.10e}")


def main():
    h = 0.1
    h_half = h / 2.0

    print("Лабораторная работа 4.2. Вариант 24")
    print("Краевая задача: (x^2 + 1)y'' - 2y = 0")
    print("Граничные условия: y'(0)=2, y(1)=3+pi/2")
    print("Точное решение: y=x^2+x+1+(x^2+1)arctg(x)")
    print(f"Основной шаг h = {h}")
    print(f"Половинный шаг h/2 = {h_half}")
    print(f"Точность для метода секущих eps = {EPS}")

    # ------------------------- Метод стрельбы -------------------------
    eta, shooting_solution_full, iterations = shooting_method(h)
    eta_half, shooting_solution_half_full, _ = shooting_method(h_half)

    shooting_solution = [(x, y) for x, y, _ in shooting_solution_full]
    shooting_solution_half = [(x, y) for x, y, _ in shooting_solution_half_full]

    # Метод Рунге-Кутты 4-го порядка, значит p = 4.
    rr_shooting = runge_romberg_error(shooting_solution, shooting_solution_half, order=4)

    print_iterations(iterations)
    print(f"\nНайденный параметр стрельбы eta = y(0) = {eta:.10f}")
    print_solution_table("Метод стрельбы", shooting_solution, rr_shooting)

    # --------------------- Конечно-разностный метод -------------------
    fd_solution = finite_difference_method(h)
    fd_solution_half = finite_difference_method(h_half)

    # Из-за односторонней аппроксимации y'(0) по формуле (4.40)
    # вся конечно-разностная схема имеет первый порядок на границе, поэтому p = 1.
    rr_fd = runge_romberg_error(fd_solution, fd_solution_half, order=1)

    print_solution_table("Конечно-разностный метод", fd_solution, rr_fd)


if __name__ == "__main__":
    main()
