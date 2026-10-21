import numpy as np
from mlscratch.linalg import shadow, lstsq


def test_example_from_video():
    result = shadow(np.array([3.0, 1.0]), np.array([1.0, 1.0]))
    assert np.allclose(result, [2.0, 2.0])


def test_residual_is_perpendicular():
    b = np.array([5.0, 1.0])
    a = np.array([1.0, 1.0])
    r = b - shadow(b, a)
    assert np.isclose(r @ a, 0.0)


def test_lstsq_example():
    A = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    y = np.array([2.0, 0.0, 4.0])
    assert np.allclose(lstsq(A, y), [1.0, 1.0])


def test_lstsq_perpendicular():
    A = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    y = np.array([2.0, 0.0, 4.0])
    x = lstsq(A, y)
    r = y - A @ x
    assert np.allclose(A.transpose() @ r, 0)