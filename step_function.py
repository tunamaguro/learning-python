import numpy as np


def step_function(x: np.ndarray):
    return np.array(x > 0, dtype=np.int32)
