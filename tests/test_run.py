import numpy as np
from numcompute.stats import mean


def test_mean_smoke():
    assert mean(np.array([5, 10, 3])) == 6
