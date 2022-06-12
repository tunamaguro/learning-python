import numpy as np

from common.functions import cross_entropy_error, softmax


class MulLayer:
    def __init__(self) -> None:
        self.x = None
        self.y = None

    def forward(self, x: float, y: float):
        self.x = x
        self.y = y
        out = x*y
        return out

    def backward(self, dout: float):
        dx = dout*self.y
        dy = dout*self.x

        return dx, dy


class AddLayer:
    def __init__(self) -> None:
        pass

    def forward(self, x: float, y: float):
        out = x+y
        return out

    def backward(self, dout: float):
        dx = dout*1
        dy = dout*1
        return dx, dy


class Relu:
    def __init__(self) -> None:
        self.mask = None

    def forward(self, x: np.ndarray):
        self.mask = x <= 0
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx


class Sigmoid:
    def __init__(self) -> None:
        self.y = None
        pass

    def forward(self, x: np.ndarray):
        out = 1/(1+np.exp(-x))
        self.y = out

        return out

    def backward(self, dout: float):
        dx = dout*(1-self.y)*self.y

        return dx


class Affine:
    def __init__(self, W: np.ndarray, b: np.ndarray) -> None:
        self.W = W
        self.b = b
        self.x = None

        self.original_x_shape = None

        self.dW = None
        self.db = None

    def forward(self, x: np.ndarray):
        self.original_x_shape = x.shape
        self.x = x.reshape(x.shape[0], -1)
        out = np.dot(self.x, self.W)+self.b
        return out

    def backward(self, dout: np.ndarray):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        dx = dx.reshape(*self.original_x_shape)

        return dx


class SoftmaxWithLoss:
    def __init__(self) -> None:
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x: np.ndarray, t: np.ndarray):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y-self.t)/batch_size
        return dx
