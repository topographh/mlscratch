import numpy as np


def add_bias(X: np.ndarray) -> np.ndarray:
    """Добавляет 1 ко всем строкам слева"""

    xs = X.shape[0]

    ones = np.ones((xs, 1))

    return np.hstack([ones, X])


def standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """стандартизует каждый столбец (признак) таблицы X"""

    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    
    return ((X-mu) / sigma, mu, sigma)

