import math
import cmath


# =========================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =========================================================

def print_matrix(A, title="Матрица"):
    """Красивый вывод матрицы."""
    print(title + ":")
    for row in A:
        print(" ".join(f"{x:12.6f}" for x in row))
    print()


def identity_matrix(n):
    """Единичная матрица размера n x n."""
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def copy_matrix(A):
    """Копия матрицы."""
    return [row[:] for row in A]


def transpose(A):
    """Транспонирование матрицы."""
    n = len(A)
    m = len(A[0])
    return [[A[i][j] for i in range(n)] for j in range(m)]


def matmul(A, B):
    """Умножение матриц A * B."""
    n = len(A)
    m = len(B[0])
    k = len(B)

    C = [[0.0 for _ in range(m)] for _ in range(n)]
    for i in range(n):
        for j in range(m):
            s = 0.0
            for t in range(k):
                s += A[i][t] * B[t][j]
            C[i][j] = s
    return C


def vector_norm_2(v):
    """Евклидова норма вектора."""
    return math.sqrt(sum(x * x for x in v))


def is_square_matrix(A):
    """Проверка, что матрица квадратная."""
    if not isinstance(A, list) or len(A) == 0:
        return False
    n = len(A)
    return all(isinstance(row, list) and len(row) == n for row in A)


def sign(x):
    """
    Знак числа.
    В методичке используется sign(b1).
    """
    return 1.0 if x >= 0 else -1.0


# =========================================================
# ХАУСХОЛДЕР: ВЕКТОР И МАТРИЦА
# =========================================================

def build_householder_vector(b):
    """
    Построение вектора Хаусхолдера по формуле:

        v = b + sign(b1) * ||b|| * e1
    """
    norm_b = vector_norm_2(b)
    if norm_b < 1e-15:
        return None

    v = b[:]
    v[0] += sign(b[0]) * norm_b

    norm_v = vector_norm_2(v)
    if norm_v < 1e-15:
        return None

    return v


def build_householder_matrix(v, n, start):
    """
    Строим полную матрицу Хаусхолдера размера n x n.

    Локальная формула:
        H_local = E - 2 * (v v^T) / (v^T v)

    Затем встраиваем ее в общую матрицу с позиции 'start'.
    """
    H = identity_matrix(n)
    m = len(v)

    vt_v = sum(x * x for x in v)
    if abs(vt_v) < 1e-15:
        return H

    for i in range(m):
        for j in range(m):
            H[start + i][start + j] -= 2.0 * v[i] * v[j] / vt_v

    return H


# =========================================================
# QR-РАЗЛОЖЕНИЕ МЕТОДОМ ХАУСХОЛДЕРА
# =========================================================

def householder_qr(A):
    """
    QR-разложение матрицы A:
        A = Q * R

    По методичке:
    последовательно строим матрицы Хаусхолдера H1, H2, ...,
    зануляя элементы ниже главной диагонали в каждом столбце.
    """
    if not is_square_matrix(A):
        raise ValueError("Матрица должна быть квадратной.")

    n = len(A)
    R = copy_matrix(A)
    Q = identity_matrix(n)

    # На k-м шаге зануляем элементы ниже диагонали в k-м столбце
    for k in range(n - 1):
        # Берем подстолбец начиная с диагонального элемента
        b = [R[i][k] for i in range(k, n)]

        v = build_householder_vector(b)
        if v is None:
            continue

        H = build_householder_matrix(v, n, k)

        # Преобразуем текущую матрицу
        R = matmul(H, R)

        # Накапливаем итоговую ортогональную матрицу Q
        Q = matmul(Q, H)

    return Q, R


# =========================================================
# ПРИВЕДЕНИЕ К ВЕРХНЕЙ ФОРМЕ ХЕССЕНБЕРГА
# =========================================================

def hessenberg_reduction(A):
    """
    Предварительное приведение матрицы к верхней форме Хессенберга.

    По методичке перед QR-алгоритмом матрицу полезно привести
    к виду A^(0) = H^T * A * H, где A^(0) - верхняя хессенбергова.

    В верхней форме Хессенберга все элементы ниже первой поддиагонали равны нулю:
        a[i][j] = 0, если i > j + 1
    """
    if not is_square_matrix(A):
        raise ValueError("Матрица должна быть квадратной.")

    n = len(A)
    H_total = identity_matrix(n)
    A_h = copy_matrix(A)

    # Нужно занулять элементы ниже первой поддиагонали
    for k in range(n - 2):
        # Берем часть столбца ниже первой поддиагонали:
        # a[k+1][k], a[k+2][k], ..., a[n-1][k]
        b = [A_h[i][k] for i in range(k + 1, n)]

        v = build_householder_vector(b)
        if v is None:
            continue

        H = build_householder_matrix(v, n, k + 1)

        # Так как H - симметричная и ортогональная:
        # H^T = H
        A_h = matmul(matmul(H, A_h), H)
        H_total = matmul(H_total, H)

    return A_h, H_total


def is_hessenberg(A, tol=1e-10):
    """Проверка, что матрица верхняя хессенбергова."""
    n = len(A)
    for i in range(n):
        for j in range(n):
            if i > j + 1 and abs(A[i][j]) > tol:
                return False
    return True


# =========================================================
# ВСПОМОГАТЕЛЬНОЕ ДЛЯ QR-АЛГОРИТМА
# =========================================================

def solve_quadratic_for_2x2_block(a, b, c, d):
    """
    Собственные значения блока 2x2:
        | a  b |
        | c  d |

    Характеристическое уравнение:
        lambda^2 - (a+d) lambda + (ad - bc) = 0
    """
    trace = a + d
    det = a * d - b * c
    discriminant = trace * trace - 4.0 * det

    sqrt_disc = cmath.sqrt(discriminant)
    lambda1 = (trace + sqrt_disc) / 2.0
    lambda2 = (trace - sqrt_disc) / 2.0
    return lambda1, lambda2


# =========================================================
# QR-АЛГОРИТМ ДЛЯ СОБСТВЕННЫХ ЗНАЧЕНИЙ
# =========================================================

def qr_eigenvalues(A, eps=1e-6, max_iter=1000):
    """
    QR-алгоритм по методичке.

    Этапы:
    1. Приведение A к форме Хессенберга: A^(0)
    2. Далее итерации:
           A^(k) = Q^(k) R^(k)
           A^(k+1) = R^(k) Q^(k)
    3. Постепенно выделяем собственные значения:
       - если поддиагональный элемент a[n-1][n-2] мал,
         считаем a[n-1][n-1] вещественным собственным значением;
       - иначе работаем с нижним блоком 2x2, который может дать
         два вещественных или комплексно-сопряженных собственных значения.

    Возвращаем:
    - список собственных значений,
    - итоговую матрицу после итераций,
    - число итераций,
    - матрицу A^(0) после приведения к Хессенбергу.
    """
    if not is_square_matrix(A):
        raise ValueError("Матрица должна быть квадратной.")
    if eps <= 0:
        raise ValueError("eps должен быть > 0.")
    if max_iter <= 0:
        raise ValueError("max_iter должен быть > 0.")

    n = len(A)

    # Шаг 1. Приведение к форме Хессенберга
    A0, _ = hessenberg_reduction(A)
    Ak = copy_matrix(A0)

    eigenvalues = []
    total_iterations = 0

    # Размер активного верхнего левого блока,
    # в котором еще не выделены все собственные значения
    m = n

    while m > 0:
        # Если остался блок 1x1, его диагональный элемент и есть собственное значение
        if m == 1:
            eigenvalues.append(Ak[0][0])
            break

        # Если нижний поддиагональный элемент мал, то выделяем вещественное собственное значение
        if abs(Ak[m - 1][m - 2]) < eps:
            eigenvalues.append(Ak[m - 1][m - 1])
            m -= 1
            continue

        # Если остался блок 2x2, вычисляем его собственные значения напрямую
        if m == 2:
            a = Ak[0][0]
            b = Ak[0][1]
            c = Ak[1][0]
            d = Ak[1][1]
            lambda1, lambda2 = solve_quadratic_for_2x2_block(a, b, c, d)
            eigenvalues.append(lambda1)
            eigenvalues.append(lambda2)
            break

        # QR-итерации на активном блоке размера m x m
        converged = False
        for _ in range(max_iter):
            total_iterations += 1

            # Берем активный ведущий блок m x m
            active_block = [row[:m] for row in Ak[:m]]

            # QR-разложение активного блока
            Q, R = householder_qr(active_block)

            # A^(k+1) = R^(k) Q^(k)
            next_block = matmul(R, Q)

            # Возвращаем обновленный блок на место
            for i in range(m):
                for j in range(m):
                    Ak[i][j] = next_block[i][j]

            # Проверяем, не отделилось ли 1x1 снизу
            if abs(Ak[m - 1][m - 2]) < eps:
                converged = True
                break

            # Проверяем, не образовался ли 2x2 блок снизу
            # Это не строгая "детекция комплексности", но позволяет корректно завершать
            # процесс на последнем блоке 2x2.
            if abs(Ak[m - 2][m - 3]) < eps:
                converged = True
                break

        if not converged:
            raise RuntimeError(
                "QR-алгоритм не сошелся за заданное число итераций. "
                "Попробуйте увеличить max_iter или ослабить eps."
            )

    return eigenvalues, Ak, total_iterations, A0


# =========================================================
# ОСНОВНАЯ ПРОГРАММА
# =========================================================

def main():
    # Вариант 24
    A = [
        [-1.0, 8.0, 5.0],
        [8.0, -4.0, 4.0],
        [2.0, 9.0, -2.0]
    ]

    eps = float(input("Введите точность eps (например, 0.001): "))

    print_matrix(A, "Исходная матрица")

    # -----------------------------------------------------
    # 1. QR-разложение исходной матрицы
    # -----------------------------------------------------
    Q, R = householder_qr(A)
    print_matrix(Q, "Матрица Q")
    print_matrix(R, "Матрица R")
    print_matrix(matmul(Q, R), "Проверка Q * R")

    # Проверка ортогональности Q
    QtQ = matmul(transpose(Q), Q)
    print_matrix(QtQ, "Проверка ортогональности Q^T * Q")

    # -----------------------------------------------------
    # 2. Приведение к форме Хессенберга
    # -----------------------------------------------------
    A0, _ = hessenberg_reduction(A)
    print_matrix(A0, "Матрица A^(0) после приведения к форме Хессенберга")
    print(f"Проверка формы Хессенберга: {is_hessenberg(A0)}")
    print()

    # -----------------------------------------------------
    # 3. QR-алгоритм
    # -----------------------------------------------------
    eigenvalues, Ak, iterations, _ = qr_eigenvalues(A, eps=eps, max_iter=1000)

    print_matrix(Ak, "Матрица после QR-итераций")

    print("Собственные значения:")
    for i, lam in enumerate(eigenvalues, start=1):
        if abs(lam.imag) < 1e-12:
            print(f"lambda{i} = {lam.real:.6f}")
        else:
            print(f"lambda{i} = {lam}")

    print(f"\nКоличество QR-итераций: {iterations}")


if __name__ == "__main__":
    main()
