import numpy as np

def shadow(b: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Возвращает вектор тени"""

    return a * (b @ a) / (a @ a)


def lstsq(a: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Функция МНК"""

    x = np.linalg.solve(a.transpose() @ a, a.transpose() @ y)
    return x
