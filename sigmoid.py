import numpy as np


def sigmoid(x: np.ndarray):
    return 1/(1+np.exp(-x))
