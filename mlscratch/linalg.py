import numpy as np

def shadow(b, a):
    """Возвращает вектор тени"""

    return a * (b @ a) / (a @ a)


def lstsq(a, y):
    """Функция МНК"""

    x = np.linalg.solve(a.transpose() @ a, a.transpose() @ y)
    return x
