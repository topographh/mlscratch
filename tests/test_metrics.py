import numpy as np

from mlscratch.metrics import mse


def test_mse():
    assert np.allclose(mse(np.array([3, 5, 7]), np.array([2, 5, 9])), 1.6666666666666667)