import math


def print_matrix(M, name="Matrix"):
    """
    Вывод матрицы на экран.
    """
    print(f"\n{name}:")
    for row in M:
        print(" ".join(f"{x:12.6f}" for x in row))


def identity_matrix(n):
    """
    Построение единичной матрицы размера n x n.
    """
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def transpose(A):
    """
    Транспонирование матрицы A.
    """
    n = len(A)
    m = len(A[0])
    return [[A[i][j] for i in range(n)] for j in range(m)]


def multiply_matrices(A, B):
    """
    Умножение двух матриц A и B.
    """
    n = len(A)
    m = len(B[0])
    k = len(B)
    C = [[0.0 for _ in range(m)] for _ in range(n)]

    for i in range(n):
        for j in range(m):
            for t in range(k):
                C[i][j] += A[i][t] * B[t][j]

    return C


def is_symmetric(A, tol=1e-12):
    """
    Проверка симметричности матрицы.
    Метод вращений Якоби применим только к симметрическим матрицам.
    """
    n = len(A)
    for i in range(n):
        for j in range(n):
            if abs(A[i][j] - A[j][i]) > tol:
                return False
    return True


def off_diagonal_sum_squares(A):
    """
    Сумма квадратов внедиагональных элементов матрицы.
    Этот критерий используется для остановки итераций.
    """
    n = len(A)
    s = 0.0

    for i in range(n):
        for j in range(n):
            if i != j:
                s += A[i][j] ** 2

    return s


def off_diagonal_norm(A):
    """
    Корень из суммы квадратов внедиагональных элементов.
    Используется для вывода погрешности по итерациям.
    """
    return math.sqrt(off_diagonal_sum_squares(A))


def max_offdiag_index(A):
    """
    Поиск максимального по модулю внедиагонального элемента матрицы.
    Возвращает его индексы (p, q).
    """
    n = len(A)
    max_val = abs(A[0][1])
    p, q = 0, 1

    for i in range(n):
        for j in range(i + 1, n):
            if abs(A[i][j]) > max_val:
                max_val = abs(A[i][j])
                p, q = i, j

    return p, q


def jacobi_rotation_method(A, eps=1e-6, max_iter=100):
    """
    Метод вращений Якоби для нахождения собственных значений
    и собственных векторов симметрической матрицы.
    """

    n = len(A)

    if not is_symmetric(A):
        raise ValueError(
            "Метод вращений Якоби применим только к симметрическим матрицам."
        )

    # Начальная матрица A^(0)
    A_k = [row[:] for row in A]

    # Накопленная матрица вращений
    U = identity_matrix(n)

    # История изменения погрешности
    history = []

    for iteration in range(max_iter):
        off_sq = off_diagonal_sum_squares(A_k)
        err = math.sqrt(off_sq)

        history.append((iteration, err, [A_k[i][i] for i in range(n)]))

        # Критерий остановки:
        # сумма квадратов внедиагональных элементов должна стать меньше eps^2
        if off_sq < eps ** 2:
            break

        # Выбираем максимальный по модулю внедиагональный элемент
        i, j = max_offdiag_index(A_k)

        # Вычисляем угол поворота
        if A_k[i][i] == A_k[j][j]:
            phi = math.pi / 4
        else:
            phi = 0.5 * math.atan((2 * A_k[i][j]) / (A_k[i][i] - A_k[j][j]))

        c = math.cos(phi)
        s = math.sin(phi)

        # Строим матрицу вращения U_k
        R = identity_matrix(n)
        R[i][i] = c
        R[j][j] = c
        R[i][j] = -s
        R[j][i] = s

        # Выполняем преобразование подобия:
        # A^(k+1) = U_k^T * A^(k) * U_k
        Rt = transpose(R)
        A_k = multiply_matrices(multiply_matrices(Rt, A_k), R)

        # Накапливаем произведение матриц вращений
        U = multiply_matrices(U, R)

    # Диагональные элементы итоговой матрицы - собственные значения
    eigenvalues = [A_k[i][i] for i in range(n)]

    # Столбцы матрицы U - собственные векторы
    eigenvectors = [[U[i][j] for i in range(n)] for j in range(n)]

    return eigenvalues, eigenvectors, history, A_k


def main():
    """
    Основная функция программы.
    """

    A = [
        [2.0, -9.0, 4.0],
        [-9.0, 0.0, 9.0],
        [4.0, 9.0, 6.0]
    ]

    eps = float(input("Введите точность eps (например, 0.001): "))

    eigenvalues, eigenvectors, history, A_final = jacobi_rotation_method(A, eps)

    print_matrix(A, "Исходная матрица")
    print_matrix(A_final, "Почти диагональная матрица после вращений")

    print("\nСобственные значения:")
    for i, lam in enumerate(eigenvalues, start=1):
        print(f"lambda{i} = {lam:.6f}")

    print("\nСобственные векторы:")
    for i, vec in enumerate(eigenvectors, start=1):
        print(f"x{i} = (" + ", ".join(f"{v:.6f}" for v in vec) + ")")

    print("\nЗависимость погрешности от числа итераций:")
    print(f"{'Итерация':>10} {'Погрешность':>15}")
    for it, err, diag in history:
        print(f"{it:10d} {err:15.8f}")

    print(f"\nКоличество итераций: {len(history) - 1}")


if __name__ == "__main__":
    main()