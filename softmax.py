import numpy as np


def softmax(a: np.ndarray):
    c: int = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a: int = np.sum(exp_a)
    y = exp_a/sum_exp_a
    return y
