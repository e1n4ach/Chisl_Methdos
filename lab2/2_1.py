import math


# ============================================================
# Лабораторная работа 2.1
# Вариант 24:
#     x^6 - 5x - 2 = 0
#
# Реализованы:
# 1) метод Ньютона
# 2) метод простой итерации
#
# Алгоритм и критерии остановки записаны по методичке.
# ============================================================


# ------------------------------------------------------------
# Исходная функция и ее производные
# ------------------------------------------------------------
def f(x: float) -> float:
    """Исходная функция f(x) = x^6 - 5x - 2."""
    return x**6 - 5 * x - 2


def df(x: float) -> float:
    """Первая производная f'(x) = 6x^5 - 5."""
    return 6 * x**5 - 5


def d2f(x: float) -> float:
    """Вторая производная f''(x) = 30x^4."""
    return 30 * x**4


# ------------------------------------------------------------
# Функция phi(x) для метода простой итерации
#
# Из уравнения:
#     x^6 - 5x - 2 = 0
# получаем:
#     x^6 = 5x + 2
#     x = (5x + 2)^(1/6) = phi(x)
# ------------------------------------------------------------
def phi(x: float) -> float:
    """Функция phi(x) = (5x + 2)^(1/6)."""
    return (5 * x + 2) ** (1 / 6)


def dphi(x: float) -> float:
    """Производная phi'(x) = 5 / (6 * (5x + 2)^(5/6))."""
    return 5 / (6 * (5 * x + 2) ** (5 / 6))


# ------------------------------------------------------------
# Вспомогательная функция:
# проверка сохранения знака функции g(x) на [a, b]
#
# В методичке требуется проверять знак производных на отрезке.
# Так как в программе невозможно проверить "для всех x" буквально,
# делаем численную проверку на плотной сетке внутри отрезка.
# Это именно программная реализация проверки условия методички.
# ------------------------------------------------------------
def sign_is_constant_on_segment(g, a: float, b: float, steps: int = 10000) -> tuple[bool, int]:
    """
    Проверяет, сохраняет ли функция g(x) знак на [a, b].

    Возвращает:
        (True, sign), если знак сохраняется,
        (False, 0), если знак меняется или где-то близок к нулю.

    sign = 1  -> функция положительна на отрезке
    sign = -1 -> функция отрицательна на отрезке
    """
    h = (b - a) / steps
    eps_zero = 1e-14

    first_value = g(a)
    if abs(first_value) < eps_zero:
        return False, 0

    sign = 1 if first_value > 0 else -1

    for i in range(steps + 1):
        x = a + i * h
        value = g(x)

        if abs(value) < eps_zero:
            return False, 0

        current_sign = 1 if value > 0 else -1
        if current_sign != sign:
            return False, 0

    return True, sign


# ------------------------------------------------------------
# Проверка условий локализации корня по методичке
# ------------------------------------------------------------
def check_root_isolation(a: float, b: float) -> None:
    """
    Проверяет условия отделения корня на [a, b]:
    1) f(a) * f(b) < 0
    2) f'(x) сохраняет знак на [a, b]  -> функция строго монотонна
    """
    fa = f(a)
    fb = f(b)

    if fa * fb >= 0:
        raise ValueError(
            "На отрезке [a, b] не выполняется условие f(a) * f(b) < 0. "
            "Корень не отделен."
        )

    monotonic, sign_df = sign_is_constant_on_segment(df, a, b)
    if not monotonic:
        raise ValueError(
            "На отрезке [a, b] первая производная f'(x) не сохраняет знак. "
            "Нельзя утверждать строгую монотонность по методичке."
        )

    print("ПРОВЕРКА ОТДЕЛЕНИЯ КОРНЯ")
    print(f"Отрезок: [{a}, {b}]")
    print(f"f(a) = {fa:.10f}")
    print(f"f(b) = {fb:.10f}")
    print(f"f(a) * f(b) = {fa * fb:.10f} < 0")
    print(
        "f'(x) сохраняет знак на [a, b]: "
        + ("положительный" if sign_df > 0 else "отрицательный")
    )
    print("Следовательно, на [a, b] содержится единственный корень.\n")


# ------------------------------------------------------------
# Метод Ньютона
# ------------------------------------------------------------
def choose_initial_newton(a: float, b: float) -> float:
    """
    Выбор начального приближения по условию методички:
        f(x0) * f''(x0) > 0
    """
    left_condition = f(a) * d2f(a)
    right_condition = f(b) * d2f(b)

    if left_condition > 0:
        return a
    if right_condition > 0:
        return b

    raise ValueError(
        "Не удалось выбрать начальное приближение для метода Ньютона "
        "по условию f(x0) * f''(x0) > 0."
    )


def newton_method(a: float, b: float, eps: float) -> list[dict]:
    """
    Метод Ньютона по формуле:
        x_(k+1) = x_k - f(x_k) / f'(x_k)

    Остановка:
        |x_(k+1) - x_k| < eps
    """
    # Проверка постоянства знаков f'(x) и f''(x) на [a, b]
    const_df, sign_df = sign_is_constant_on_segment(df, a, b)
    const_d2f, sign_d2f = sign_is_constant_on_segment(d2f, a, b)

    if not const_df:
        raise ValueError("На [a, b] f'(x) не сохраняет знак. Условие метода Ньютона не выполнено.")
    if not const_d2f:
        raise ValueError("На [a, b] f''(x) не сохраняет знак. Условие метода Ньютона не выполнено.")

    xk = choose_initial_newton(a, b)

    print("МЕТОД НЬЮТОНА")
    print(f"f'(x) сохраняет знак на [a, b]: {'+' if sign_df > 0 else '-'}")
    print(f"f''(x) сохраняет знак на [a, b]: {'+' if sign_d2f > 0 else '-'}")
    print(f"Начальное приближение x0 = {xk:.10f}")
    print("Останов: |x_(k+1) - x_k| < eps\n")

    rows = []
    k = 0

    while True:
        fx = f(xk)
        dfx = df(xk)

        if abs(dfx) < 1e-14:
            raise ZeroDivisionError("f'(x_k) = 0, метод Ньютона продолжить нельзя.")

        correction = fx / dfx
        x_next = xk - correction
        diff = abs(x_next - xk)

        rows.append({
            "k": k,
            "xk": xk,
            "f_xk": fx,
            "df_xk": dfx,
            "corr": correction,
            "x_next": x_next,
            "diff": diff
        })

        if diff < eps:
            break

        xk = x_next
        k += 1

    return rows


# ------------------------------------------------------------
# Метод простой итерации
# ------------------------------------------------------------
def check_simple_iteration_conditions(a: float, b: float) -> float:
    """
    Проверяет условия теоремы для метода простой итерации:
    1) phi(x) принадлежит [a, b] для всех x из [a, b]
    2) существует q < 1, такое что |phi'(x)| <= q на [a, b]

    Возвращает найденное q.
    """
    # Проверка phi([a, b]) ⊂ [a, b]
    steps = 10000
    h = (b - a) / steps

    min_phi = phi(a)
    max_phi = phi(a)

    for i in range(steps + 1):
        x = a + i * h
        value = phi(x)

        if value < min_phi:
            min_phi = value
        if value > max_phi:
            max_phi = value

    if min_phi < a or max_phi > b:
        raise ValueError(
            "Не выполнено условие phi([a, b]) ⊂ [a, b] для метода простой итерации."
        )

    # Проверка |phi'(x)| <= q < 1
    q = abs(dphi(a))
    for i in range(steps + 1):
        x = a + i * h
        value = abs(dphi(x))
        if value > q:
            q = value

    if q >= 1:
        raise ValueError(
            "Не выполнено условие |phi'(x)| <= q < 1 для метода простой итерации."
        )

    print("МЕТОД ПРОСТОЙ ИТЕРАЦИИ")
    print(f"phi([a, b]) ⊂ [a, b]: [{min_phi:.10f}, {max_phi:.10f}] ⊂ [{a}, {b}]")
    print(f"q = max|phi'(x)| на [a, b] = {q:.10f} < 1")
    print("Начальное приближение по методичке берем из [a, b].")
    print("Удобно взять середину отрезка:")
    print(f"x0 = (a + b) / 2 = {(a + b) / 2:.10f}")
    print("Останов: q/(1-q) * |x_(k+1) - x_k| <= eps\n")

    return q


def simple_iteration_method(a: float, b: float, eps: float) -> tuple[list[dict], float]:
    """
    Метод простой итерации по формуле:
        x_(k+1) = phi(x_k)

    Останов:
        q/(1-q) * |x_(k+1) - x_k| <= eps
    """
    q = check_simple_iteration_conditions(a, b)
    xk = (a + b) / 2

    rows = []
    k = 0

    while True:
        x_next = phi(xk)
        diff = abs(x_next - xk)
        estimate = q / (1 - q) * diff

        rows.append({
            "k": k,
            "xk": xk,
            "phi_xk": x_next,
            "diff": diff,
            "estimate": estimate
        })

        if estimate <= eps:
            break

        xk = x_next
        k += 1

    return rows, q


# ------------------------------------------------------------
# Вспомогательный расчет "точного" значения корня
# Нужен только для анализа зависимости погрешности от числа итераций.
# ------------------------------------------------------------
def reference_root() -> float:
    """
    Находим очень точное значение корня тем же методом Ньютона,
    но с гораздо меньшей точностью.
    Это нужно только для анализа уменьшения фактической погрешности.
    """
    x = 2.0
    eps_ref = 1e-15

    while True:
        x_next = x - f(x) / df(x)
        if abs(x_next - x) < eps_ref:
            return x_next
        x = x_next


# ------------------------------------------------------------
# Печать таблиц
# ------------------------------------------------------------
def print_newton_table(rows: list[dict], x_ref: float) -> None:
    print("Таблица итераций метода Ньютона")
    print(
        f"{'k':>3} "
        f"{'x_k':>16} "
        f"{'f(x_k)':>16} "
        f"{'f (x_k)':>16} "
        f"{'f/df':>16} "
        f"{'x_(k+1)':>16} "
        f"{'|x_(k+1)-x_k|':>18} "
        f"{'|x_k-x*|':>16}"
    )

    for row in rows:
        error = abs(row["xk"] - x_ref)
        print(
            f"{row['k']:3d} "
            f"{row['xk']:16.10f} "
            f"{row['f_xk']:16.10f} "
            f"{row['df_xk']:16.10f} "
            f"{row['corr']:16.10f} "
            f"{row['x_next']:16.10f} "
            f"{row['diff']:18.10e} "
            f"{error:16.10e}"
        )
    print()


def print_simple_iteration_table(rows: list[dict], x_ref: float) -> None:
    print("Таблица итераций метода простой итерации")
    print(
        f"{'k':>3} "
        f"{'x_k':>16} "
        f"{'phi(x_k)':>16} "
        f"{'|x_(k+1)-x_k|':>18} "
        f"{'Оценка погр.':>16} "
        f"{'|x_k-x*|':>16}"
    )

    for row in rows:
        error = abs(row["xk"] - x_ref)
        print(
            f"{row['k']:3d} "
            f"{row['xk']:16.10f} "
            f"{row['phi_xk']:16.10f} "
            f"{row['diff']:18.10e} "
            f"{row['estimate']:16.10e} "
            f"{error:16.10e}"
        )
    print()


# ------------------------------------------------------------
# Главная функция
# ------------------------------------------------------------
def main():
    # Входные данные: требуемая точность
    eps = float(input("Введите точность eps: "))

    # Положительный корень по графическому отделению берем на отрезке [1, 2]
    a = 1.0
    b = 2.0

    print("\nУРАВНЕНИЕ: x^6 - 5x - 2 = 0")
    print("Ищем положительный корень.\n")

    # 1. Проверка отделения корня
    check_root_isolation(a, b)

    # 2. Метод Ньютона
    newton_rows = newton_method(a, b, eps)

    # 3. Метод простой итерации
    simple_rows, q = simple_iteration_method(a, b, eps)

    # 4. Эталонный корень для анализа погрешности
    x_ref = reference_root()

    # 5. Вывод результатов
    print(f"Эталонное значение корня для анализа погрешности: x* = {x_ref:.15f}\n")

    print_newton_table(newton_rows, x_ref)
    print_simple_iteration_table(simple_rows, x_ref)

    # 6. Итоговые приближения
    x_newton = newton_rows[-1]["x_next"]
    x_simple = simple_rows[-1]["phi_xk"]

    print("ИТОГ")
    print(f"Метод Ньютона:           x ≈ {x_newton:.10f}, число итераций = {len(newton_rows)}")
    print(f"Метод простой итерации:  x ≈ {x_simple:.10f}, число итераций = {len(simple_rows)}")

    # 7. Краткий анализ зависимости погрешности от числа итераций
    print("\nАНАЛИЗ ПОГРЕШНОСТИ")
    print("По таблицам видно, что с ростом числа итераций погрешность уменьшается.")
    print("Метод Ньютона сходится быстрее, чем метод простой итерации.")
    print("У метода простой итерации сходимость медленнее, так как используется итерационная схема")
    print("x_(k+1) = phi(x_k) и теоретическая оценка погрешности через коэффициент q.")


if __name__ == "__main__":
    main()
