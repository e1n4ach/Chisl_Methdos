import numpy as np
import sys



def lu_decomposition_with_pivoting(A: np.ndarray):
    """
    LU-разложение с частичным выбором главного элемента.
    Возвращает P, L, U и число перестановок строк.
    Выполняется: P @ A = L @ U
    """
    A = A.astype(float).copy()   # работа с копией матрицы в float
    n = A.shape[0]

    P = np.eye(n)                # матрица перестановок
    L = np.zeros((n, n))         # нижняя треугольная матрица
    U = A.copy()                 # верхняя треугольная матрица
    swaps = 0                    # число перестановок строк

    for k in range(n):
        # Ищем главный элемент в текущем столбце
        pivot = np.argmax(np.abs(U[k:, k])) + k 

        # Если главный элемент почти ноль, разложение невозможно
        if abs(U[pivot, k]) < 1e-12:
            raise ValueError("Матрица вырождена: LU-разложение невозможно")

        # Переставляем строки, если главный элемент не на диагонали
        if pivot != k:
            U[[k, pivot], :] = U[[pivot, k], :]
            P[[k, pivot], :] = P[[pivot, k], :]

            # Уже найденные коэффициенты в L тоже нужно переставить
            if k > 0:
                L[[k, pivot], :k] = L[[pivot, k], :k]

            swaps += 1

        # В L на диагонали стоят единицы
        L[k, k] = 1.0

        # Зануляем элементы ниже диагонали
        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]          # коэффициент исключения
            U[i, k:] -= L[i, k] * U[k, k:]       # вычитание ведущей строки
            U[i, k] = 0.0                        # явно ставим ноль

    return P, L, U, swaps


def forward_substitution(L: np.ndarray, b: np.ndarray):
    """Решение L @ y = b методом прямой подстановки."""
    n = L.shape[0]
    y = np.zeros(n)

    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])

    return y


def backward_substitution(U: np.ndarray, y: np.ndarray):
    """Решение U @ x = y методом обратной подстановки."""
    n = U.shape[0]
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        if abs(U[i, i]) < 1e-12:
            raise ValueError("Нулевой элемент на диагонали U")
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]

    return x


def solve_lu(A: np.ndarray, b: np.ndarray):
    """
    Решение системы A @ x = b через LU-разложение:
    1) P @ A = L @ U
    2) L @ y = P @ b
    3) U @ x = y
    """
    P, L, U, swaps = lu_decomposition_with_pivoting(A)

    Pb = P @ b                           # переставляем правую часть
    y = forward_substitution(L, Pb)      # решаем L @ y = P @ b
    x = backward_substitution(U, y)      # решаем U @ x = y

    # Определитель через диагональ U и число перестановок
    detA = ((-1) ** swaps) * np.prod(np.diag(U))

    return x, P, L, U, detA


def inverse_from_lu(P: np.ndarray, L: np.ndarray, U: np.ndarray):
    """
    Нахождение обратной матрицы через решение систем
    для столбцов единичной матрицы.
    """
    n = L.shape[0]
    A_inv = np.zeros((n, n))

    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0                        # j-й столбец единичной матрицы

        y = forward_substitution(L, P @ e)
        col = backward_substitution(U, y)

        A_inv[:, j] = col                 # записываем столбец обратной матрицы

    return A_inv


if __name__ == "__main__":
    # Матрица и вектор для варианта 24
    A = np.array([
        [7, -5, 6, 7],
        [8, -1, -9, 1],
        [-3, 8, 8, 8],
        [2, -3, 6, -4]
    ], dtype=float)

    b = np.array([120, 31, 6, 25], dtype=float)

    # Решение системы
    x, P, L, U, detA = solve_lu(A, b)

    # Обратная матрица
    A_inv = inverse_from_lu(P, L, U)

    np.set_printoptions(precision=6, suppress=True)

    print("Исходная матрица A:")
    print(A)

    print("\nВектор b:")
    print(b)

    print("\nМатрица перестановок P:")
    print(P)

    print("\nМатрица L:")
    print(L)

    print("\nМатрица U:")
    print(U)

    print("\nПроверка P @ A:")
    print(P @ A)

    print("\nПроверка L @ U:")
    print(L @ U)

    print("\nРешение системы x:")
    print(x)

    print("\nПроверка A @ x:")
    print(A @ x)

    print("\nОпределитель det(A):")
    print(detA)

    print("\nОбратная матрица A^(-1):")
    print(A_inv)

    print("\nПроверка A @ A^(-1):")
    print(A @ A_inv)
