import math
import matplotlib.pyplot as plt

# ============================================================
# Лабораторная работа 2.2, вариант 24
# Система:000
#   3*x1^2 - x1 + x2^2 - 1 = 0
#   x2 - tan(x1) = 0
#
# В коде соблюдены шаги из методички:
# 1) начальное приближение выбирается графически;
# 2) метод Ньютона реализован по формулам (2.13)-(2.20);
# 3) метод простой итерации реализован в форме x = phi(x);
# 4) для простой итерации проверяются q < 1 и попадание всех итераций в область G;
# 5) критерии остановки соответствуют (2.18) и (2.26).
# ============================================================

DEFAULT_EPS = 1e-6
MAX_ITER = 100
GRID_STEPS_Q = 200
GRID_STEPS_G = 80

# По графику берем область положительного решения и начальное приближение
G_X1 = (0.58, 0.64)
G_X2 = (0.67, 0.73)
X0 = (0.60, 0.70)


# -----------------------------
# Исходная система и Якобиан
# -----------------------------
def f1(x1, x2):
    return 3.0 * x1 * x1 - x1 + x2 * x2 - 1.0


def f2(x1, x2):
    return x2 - math.tan(x1)


def jacobian(x1, x2):
    # J = [[df1/dx1, df1/dx2], [df2/dx1, df2/dx2]]
    return (
        (6.0 * x1 - 1.0, 2.0 * x2),
        (-(1.0 / (math.cos(x1) ** 2)), 1.0),
    )


# -----------------------------
# Вспомогательные функции
# -----------------------------
def det2(m):
    return m[0][0] * m[1][1] - m[0][1] * m[1][0]



def norm_inf_vec(v):
    return max(abs(v[0]), abs(v[1]))



def in_G(x1, x2):
    return G_X1[0] <= x1 <= G_X1[1] and G_X2[0] <= x2 <= G_X2[1]


# ----------------------------------------------------------
# Шаг 1. Графическое определение начального приближения
# ----------------------------------------------------------
def plot_system():
    x_values = []
    tan_values = []
    sqrt_values = []

    left = -0.6
    right = 0.8
    step = 0.001
    x = left
    while x <= right:
        x_values.append(x)
        tan_values.append(math.tan(x))

        rad = 1.0 + x - 3.0 * x * x
        if rad >= 0.0:
            sqrt_values.append(math.sqrt(rad))
        else:
            sqrt_values.append(float('nan'))
        x += step

    plt.figure(figsize=(8, 6))
    plt.plot(x_values, tan_values, label='x2 = tan(x1)')
    plt.plot(x_values, sqrt_values, label='x2 = sqrt(1 + x1 - 3*x1^2)')

    # Показываем прямоугольник G и начальную точку
    gx1 = [G_X1[0], G_X1[1], G_X1[1], G_X1[0], G_X1[0]]
    gx2 = [G_X2[0], G_X2[0], G_X2[1], G_X2[1], G_X2[0]]
    plt.plot(gx1, gx2, label='область G')
    plt.scatter([X0[0]], [X0[1]], label='x^(0)')

    plt.xlim(0.45, 0.75)
    plt.ylim(0.45, 0.95)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Графический выбор начального приближения')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('lab2_2_graph.png', dpi=150)
    plt.close()


# -----------------------------------------------------------------
# Шаг 2. Метод Ньютона по формулам (2.13)-(2.20) для системы 2x2
# -----------------------------------------------------------------
# Для двух уравнений используем форму из примера методички:
# x1^(k+1) = x1^(k) - det(A1) / det(J)
# x2^(k+1) = x2^(k) - det(A2) / det(J)

def newton_method(eps, max_iter=MAX_ITER):
    x1, x2 = X0
    history = []

    for k in range(max_iter):
        cur_f1 = f1(x1, x2)
        cur_f2 = f2(x1, x2)

        J = jacobian(x1, x2)
        det_j = det2(J)
        if abs(det_j) < 1e-14:
            raise ValueError(
                f'Метод Ньютона остановлен: det(J) = 0 на итерации k = {k}.'
            )

        # Матрицы A1 и A2 из формулы (2.20)
        A1 = (
            (cur_f1, J[0][1]),
            (cur_f2, J[1][1]),
        )
        A2 = (
            (J[0][0], cur_f1),
            (J[1][0], cur_f2),
        )

        det_a1 = det2(A1)
        det_a2 = det2(A2)

        x1_new = x1 - det_a1 / det_j
        x2_new = x2 - det_a2 / det_j

        diff = norm_inf_vec((x1_new - x1, x2_new - x2))

        history.append({
            'k': k,
            'x1': x1,
            'x2': x2,
            'f1': cur_f1,
            'f2': cur_f2,
            'detJ': det_j,
            'detA1': det_a1,
            'detA2': det_a2,
            'delta': diff,
            'x1_next': x1_new,
            'x2_next': x2_new,
        })

        x1, x2 = x1_new, x2_new

        # Критерий остановки (2.18)
        if diff <= eps:
            return (x1, x2), history

    raise ValueError('Метод Ньютона не сошелся за заданное число итераций.')


# --------------------------------------------------------------------------------------
# Шаг 3. Метод простой итерации: приводим систему к виду x = phi(x)
# --------------------------------------------------------------------------------------
# Берем эквивалентную систему x = phi(x) вида
#   x^(k+1) = x^(k) - C * f(x^(k)),
# где C = J(x^(0))^{-1} - постоянная матрица.
# Это все еще форма (2.21)-(2.24) из методички.

def inverse_2x2(m):
    d = det2(m)
    if abs(d) < 1e-14:
        raise ValueError('Невозможно обратить матрицу 2x2: det = 0.')
    return (
        (m[1][1] / d, -m[0][1] / d),
        (-m[1][0] / d, m[0][0] / d),
    )


J0 = jacobian(X0[0], X0[1])
C = inverse_2x2(J0)


def phi(x1, x2):
    cur_f1 = f1(x1, x2)
    cur_f2 = f2(x1, x2)

    p1 = x1 - (C[0][0] * cur_f1 + C[0][1] * cur_f2)
    p2 = x2 - (C[1][0] * cur_f1 + C[1][1] * cur_f2)
    return p1, p2



def dphi(x1, x2):
    # Производная phi(x) = I - C*J(x)
    sec2 = 1.0 / (math.cos(x1) ** 2)
    c11, c12 = C[0]
    c21, c22 = C[1]

    return (
        (
            1.0 - c11 * (6.0 * x1 - 1.0) + c12 * sec2,
            -(2.0 * c11 * x2 + c12),
        ),
        (
            -c21 * (6.0 * x1 - 1.0) + c22 * sec2,
            1.0 - 2.0 * c21 * x2 - c22,
        ),
    )



def compute_q_on_G(grid_steps=GRID_STEPS_Q):
    # Проверка условия (2.25) по формуле (2.27):
    # q = max_{x in G} max_i sum_j |d(phi_i)/d(x_j)|
    q = -1.0
    q_point = None

    for i in range(grid_steps + 1):
        x1 = G_X1[0] + (G_X1[1] - G_X1[0]) * i / grid_steps
        for j in range(grid_steps + 1):
            x2 = G_X2[0] + (G_X2[1] - G_X2[0]) * j / grid_steps
            d = dphi(x1, x2)
            row1 = abs(d[0][0]) + abs(d[0][1])
            row2 = abs(d[1][0]) + abs(d[1][1])
            current = max(row1, row2)
            if current > q:
                q = current
                q_point = (x1, x2)

    return q, q_point



def check_phi_maps_G_into_G(grid_steps=GRID_STEPS_G):
    # Дополнительная численная проверка: phi(G) ⊂ G
    for i in range(grid_steps + 1):
        x1 = G_X1[0] + (G_X1[1] - G_X1[0]) * i / grid_steps
        for j in range(grid_steps + 1):
            x2 = G_X2[0] + (G_X2[1] - G_X2[0]) * j / grid_steps
            p1, p2 = phi(x1, x2)
            if not in_G(p1, p2):
                return False, (x1, x2), (p1, p2)
    return True, None, None



def simple_iteration_method(eps, max_iter=MAX_ITER):
    q, q_point = compute_q_on_G()
    if q >= 1.0:
        raise ValueError(
            f'Метод простой итерации неприменим: q = {q:.10f} >= 1.'
        )

    maps_ok, src_point, mapped_point = check_phi_maps_G_into_G()
    if not maps_ok:
        raise ValueError(
            'Численная проверка показала, что phi(G) не содержится в G: '
            f'phi({src_point}) = {mapped_point}.'
        )

    x1, x2 = X0
    history = []

    for k in range(max_iter):
        x1_new, x2_new = phi(x1, x2)

        if not in_G(x1_new, x2_new):
            raise ValueError(
                f'Итерация k = {k + 1} вышла из области G: ({x1_new}, {x2_new}).'
            )

        diff = norm_inf_vec((x1_new - x1, x2_new - x2))
        err_est = q / (1.0 - q) * diff  # критерий из (2.26)

        history.append({
            'k': k,
            'x1': x1,
            'x2': x2,
            'phi1': x1_new,
            'phi2': x2_new,
            'delta': diff,
            'err_est': err_est,
            'q': q,
        })

        x1, x2 = x1_new, x2_new

        # Критерий остановки (2.26)
        if err_est <= eps:
            return (x1, x2), history, q, q_point

    raise ValueError('Метод простой итерации не сошелся за заданное число итераций.')


# ----------------------------------------------------------
# Вывод таблиц и анализ погрешности по числу итераций
# ----------------------------------------------------------
def print_newton_table(history):
    print('\nМЕТОД НЬЮТОНА')
    print('-' * 120)
    print(
        f"{'k':>2} | {'x1^(k)':>12} | {'x2^(k)':>12} | {'f1':>12} | {'f2':>12} | "
        f"{'detJ':>12} | {'detA1':>12} | {'detA2':>12} | {'||x^(k+1)-x^(k)||∞':>18}"
    )
    print('-' * 120)
    for row in history:
        print(
            f"{row['k']:>2} | {row['x1']:>12.8f} | {row['x2']:>12.8f} | "
            f"{row['f1']:>12.8f} | {row['f2']:>12.8f} | {row['detJ']:>12.8f} | "
            f"{row['detA1']:>12.8f} | {row['detA2']:>12.8f} | {row['delta']:>18.10f}"
        )



def print_simple_iteration_table(history, q, q_point):
    print('\nМЕТОД ПРОСТОЙ ИТЕРАЦИИ')
    print('-' * 120)
    print(f'q = {q:.10f}, точка максимума на проверочной сетке: ({q_point[0]:.6f}, {q_point[1]:.6f})')
    print(
        f"{'k':>2} | {'x1^(k)':>12} | {'x2^(k)':>12} | {'phi1':>12} | {'phi2':>12} | "
        f"{'||x^(k+1)-x^(k)||∞':>18} | {'оценка ошибки':>16}"
    )
    print('-' * 120)
    for row in history:
        print(
            f"{row['k']:>2} | {row['x1']:>12.8f} | {row['x2']:>12.8f} | "
            f"{row['phi1']:>12.8f} | {row['phi2']:>12.8f} | "
            f"{row['delta']:>18.10f} | {row['err_est']:>16.10f}"
        )



def newton_reference_solution():
    # Нужна только для анализа зависимости фактической ошибки от номера итерации
    sol, _ = newton_method(eps=1e-14, max_iter=200)
    return sol



def plot_error_analysis(newton_history, simple_history, reference_solution):
    newton_errors = []
    newton_k = []
    for row in newton_history:
        err = norm_inf_vec((row['x1_next'] - reference_solution[0], row['x2_next'] - reference_solution[1]))
        newton_errors.append(err)
        newton_k.append(row['k'] + 1)

    simple_errors = []
    simple_k = []
    for row in simple_history:
        err = norm_inf_vec((row['phi1'] - reference_solution[0], row['phi2'] - reference_solution[1]))
        simple_errors.append(err)
        simple_k.append(row['k'] + 1)

    plt.figure(figsize=(8, 6))
    plt.semilogy(newton_k, newton_errors, marker='o', label='Ньютон: фактическая ошибка')
    plt.semilogy(simple_k, simple_errors, marker='s', label='Простая итерация: фактическая ошибка')
    plt.xlabel('Номер итерации k')
    plt.ylabel('||x^(k) - x*||∞')
    plt.title('Зависимость погрешности от количества итераций')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('lab2_2_errors.png', dpi=150)
    plt.close()


# ----------------------------------------------------------
# Главная часть
# ----------------------------------------------------------
def main():
    print('ЛР 2.2, вариант 24')
    print('Система:')
    print('  3*x1^2 - x1 + x2^2 - 1 = 0')
    print('  x2 - tan(x1) = 0')
    try:
        raw_eps = input('Введите точность eps (например, 1e-6). Если оставить пусто, будет 1e-6: ').strip()
        eps = DEFAULT_EPS if raw_eps == '' else float(raw_eps)
    except ValueError:
        raise ValueError('Точность eps должна быть числом.')

    if eps <= 0:
        raise ValueError('Точность eps должна быть положительной.')

    print(f'Точность eps = {eps}')
    print(f'Область G: x1 in {G_X1}, x2 in {G_X2}')
    print(f'Начальное приближение x^(0) = {X0}')

    # 1. График для выбора начального приближения
    plot_system()

    # 2. Метод Ньютона
    newton_solution, newton_history = newton_method(eps=eps)
    print_newton_table(newton_history)
    print(f"\nРезультат метода Ньютона: x1 = {newton_solution[0]:.10f}, x2 = {newton_solution[1]:.10f}")

    # 3. Метод простой итерации
    simple_solution, simple_history, q, q_point = simple_iteration_method(eps=eps)
    print_simple_iteration_table(simple_history, q, q_point)
    print(f"\nРезультат метода простой итерации: x1 = {simple_solution[0]:.10f}, x2 = {simple_solution[1]:.10f}")

    # 4. Анализ погрешности по числу итераций
    ref = newton_reference_solution()
    plot_error_analysis(newton_history, simple_history, ref)

    print('\nСправочно: решение повышенной точности для анализа графика ошибок:')
    print(f'x* = ({ref[0]:.14f}, {ref[1]:.14f})')


if __name__ == '__main__':
    main()
