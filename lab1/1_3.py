import numpy as np


def vector_c_norm(x):
    """
    c-норма вектора:
        ||x||_c = max_i |x_i|

    В методичке именно эта норма используется как одна из основных
    для исследования сходимости и погрешности.
    """
    return np.max(np.abs(x))


def matrix_c_norm(A):
    """
    c-норма матрицы:
        ||A||_c = max_i sum_j |a_ij|

    Это максимум суммы модулей элементов по строкам.
    В методичке такая норма приведена как согласованная с c-нормой вектора.
    """
    return np.max(np.sum(np.abs(A), axis=1))


def check_diagonal_dominance(A):
    """
    Проверка диагонального преобладания по строкам.

    Для каждой строки сравниваем:
        |a_ii|  и  sum_{j != i} |a_ij|

    Возвращаем два логических значения:
    - strict_ok: строгое диагональное преобладание по всем строкам
    - weak_ok:   нестрогое диагональное преобладание по всем строкам
    """
    n = A.shape[0]
    strict_ok = True
    weak_ok = True

    for i in range(n):
        diag = abs(A[i, i])
        off_diag_sum = np.sum(np.abs(A[i])) - diag

        if diag < off_diag_sum:
            weak_ok = False
            strict_ok = False
        elif diag == off_diag_sum:
            strict_ok = False

    return strict_ok, weak_ok


def build_beta_alpha(A, b):
    """
    Строим эквивалентную систему вида:
        x = beta + alpha * x

    По методичке:
        beta_i  = b_i / a_ii
        alpha_ij = -a_ij / a_ii,  если i != j
        alpha_ii = 0

    Если на диагонали есть нулевой элемент, метод в таком виде применять нельзя:
    нужно переставить уравнения местами.
    """
    n = A.shape[0]
    beta = np.zeros(n, dtype=float)
    alpha = np.zeros((n, n), dtype=float)

    for i in range(n):
        if A[i, i] == 0:
            raise ValueError(
                f"Нулевой диагональный элемент A[{i}][{i}]. "
                "Нужно переставить уравнения местами."
            )

        beta[i] = b[i] / A[i, i]

        for j in range(n):
            if i != j:
                alpha[i, j] = -A[i, j] / A[i, i]

    return beta, alpha


def print_method_info(name, A, alpha):
    """
    Печатаем теоретическую информацию перед запуском метода:
    - есть ли диагональное преобладание;
    - чему равна ||alpha||_c;
    - какой критерий остановки будет использоваться.
    """
    strict_dom, weak_dom = check_diagonal_dominance(A)
    alpha_norm = matrix_c_norm(alpha)

    print(f"\n{name}")
    print("-" * len(name))

    if strict_dom:
        print("Диагональное преобладание: СТРОГОЕ.")
        print("Это достаточное условие сходимости.")
    elif weak_dom:
        print("Диагональное преобладание: НЕСТРОГОЕ.")
        print("Строгой гарантии по этому признаку нет.")
    else:
        print("Диагонального преобладания нет.")
        print("Гарантия сходимости по этому признаку отсутствует.")

    print(f"||alpha||_c = {alpha_norm:.6f}")

    if alpha_norm < 1:
        print("Условие ||alpha||_c < 1 выполнено.")
        print("Можно использовать теоретическую оценку погрешности.")
    else:
        print("Условие ||alpha||_c < 1 НЕ выполнено.")
        print("Используем практический критерий остановки по разности соседних приближений.")


def jacobi(A, b, eps=1e-6, max_iter=10000):
    """
    Метод простых итераций в форме Якоби.

    Сначала строим эквивалентную систему:
        x = beta + alpha * x

    Затем запускаем итерационный процесс:
        x^(k+1) = beta + alpha * x^(k)

    В методичке указано:
    - если ||alpha|| < 1, можно использовать оценку погрешности;
    - если это условие не выполнено, но метод все же сходится,
      можно завершать итерации по условию:
          ||x^(k+1) - x^(k)|| < eps
    """
    beta, alpha = build_beta_alpha(A, b)
    alpha_norm = matrix_c_norm(alpha)

    # В качестве начального приближения берем beta,
    # как это предлагается в методичке.
    x_old = beta.copy()

    print_method_info("Метод простых итераций (Якоби)", A, alpha)

    for k in range(1, max_iter + 1):
        # Формула метода Якоби:
        x_new = beta + alpha @ x_old
        # Если выполнено достаточное условие ||alpha|| < 1,
        # можно использовать оценку погрешности
        # Если она меньше заданной точности, процесс завершаем.
        if alpha_norm < 1:
            est_error = alpha_norm / (1 - alpha_norm) * vector_c_norm(x_new - x_old)
            if est_error < eps:
                return x_new, k
        else:
            # Если ||alpha|| >= 1, используем практический критерий:
            # разность соседних приближений стала меньше eps.
            if vector_c_norm(x_new - x_old) < eps:
                return x_new, k

        x_old = x_new.copy()

    raise ValueError("Метод простых итераций не сошелся за max_iter итераций")


def gauss_seidel(A, b, eps=1e-6, max_iter=10000):
    """
    Метод Зейделя.
    Так же, как и в методе Якоби, сначала строим эквивалентную систему:
        x = beta + alpha * x

    Но при вычислении i-й компоненты на текущей итерации:
    - для j < i используем уже обновленные значения x[j];
    - для j > i используем старые значения x_old[j].
    """
    beta, alpha = build_beta_alpha(A, b)
    alpha_norm = matrix_c_norm(alpha)

    # Начальное приближение также берем равным beta.
    x = beta.copy()

    print_method_info("Метод Зейделя", A, alpha)

    for k in range(1, max_iter + 1):
        x_old = x.copy()

        # Последовательно пересчитываем компоненты решения.
        for i in range(len(b)):
            # Формула Зейделя для i-й компоненты:
            # x_i^(k+1) = beta_i
            #            + sum_{j < i} alpha_ij * x_j^(k+1)
            #            + sum_{j > i} alpha_ij * x_j^(k)
            s1 = np.dot(alpha[i, :i], x[:i])          # уже обновленные значения
            s2 = np.dot(alpha[i, i + 1:], x_old[i + 1:])  # старые значения
            x[i] = beta[i] + s1 + s2

        if alpha_norm < 1:
            # В методичке для Зейделя при выполнении достаточных условий
            # тоже используют оценку погрешности.
            # Здесь для практической реализации берем ту же схему:
            est_error = alpha_norm / (1 - alpha_norm) * vector_c_norm(x - x_old)
            if est_error < eps:
                return x, k
        else:
            if vector_c_norm(x - x_old) < eps:
                return x, k

    raise ValueError("Метод Зейделя не сошелся за max_iter итераций")


def print_solution(title, x, iters):
    """
    Красивый вывод найденного решения и числа итераций.
    """
    print(f"\n{title}:")
    for i, value in enumerate(x, start=1):
        print(f"x{i} = {value:.6f}")
    print(f"Количество итераций: {iters}")


def main():
    # Матрица коэффициентов системы A * x = b
    A = np.array([
        [-18,  9, -1, -8],
        [  6, 22,  9,  0],
        [ -4,  2, -16,  9],
        [  1,  6, -1, -14]
    ], dtype=float)

    # Вектор правых частей
    b = np.array([-60, -109, -103, -33], dtype=float)

    # Заданная пользователем точность
    eps = float(input("Введите точность eps (например, 0.001): "))

    # Решаем систему двумя итерационными методами
    x_jacobi, iter_jacobi = jacobi(A, b, eps)
    x_seidel, iter_seidel = gauss_seidel(A, b, eps)

    # Выводим результаты
    print_solution("Результат метода простых итераций (Якоби)", x_jacobi, iter_jacobi)
    print_solution("Результат метода Зейделя", x_seidel, iter_seidel)


# Стандартная точка входа в программу
if __name__ == "__main__":
    main()