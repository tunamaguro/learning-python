import numpy as np


def relu(x: int) -> int:
    return np.maximum(x, 0)


def sum_squared_error(y: np.ndarray, t: np.ndarray) -> float:
    return 0.5*np.sum((y-t)**2)


def cross_entropy_error(y: np.ndarray, t: np.ndarray) -> float:
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    if t.size == y.size:
        t = t.argmax(axis=1)

    delta = 1e-7
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t]+delta))/batch_size


def sigmoid(x: np.ndarray):
    return 1/(1+np.exp(-x))


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)


def softmax(a: np.ndarray):
    c: int = np.max(a, axis=-1, keepdims=True)
    exp_a = np.exp(a-c)
    sum_exp_a: int = np.sum(exp_a, axis=-1, keepdims=True)
    y = exp_a/sum_exp_a
    return y
