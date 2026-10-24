import numpy as np


def mse(y: np.ndarray, y_pred: np.ndarray) -> float:
    """Среднее квадратов ошибок"""

    if y.shape != y_pred.shape: raise ValueError(f"разные формы: {y.shape} и {y_pred.shape}")

    return np.sum((y - y_pred)**2) / len(y)