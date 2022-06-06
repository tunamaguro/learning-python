# import numpy as np
# from common.functions import *
# from common.gradient import numerical_gradient


# class TwoLayerNet:
#     def __init__(self, input_size: int, hidden_size: int, output_size: int, weight_init_std=0.01) -> None:
#         self.params = {}
#         self.params['W1'] = weight_init_std * \
#             np.random.randn(input_size, hidden_size)
#         self.params['b1'] = np.zeros(hidden_size)
#         self.params['W2'] = weight_init_std * \
#             np.random.randn(hidden_size, output_size)
#         self.params['b2'] = np.zeros(output_size)

#     def predict(self, x: np.ndarray):
#         W1, W2 = self.params['W1'], self.params['W2']
#         b1, b2 = self.params['b1'], self.params['b2']
#         a1 = np.dot(x, W1)+b1
#         z1 = sigmoid(a1)
#         a2 = np.dot(z1, W2)+b2
#         y = softmax(a2)

#         return y

#     def loss(self, x: np.ndarray, t: np.ndarray):
#         y = self.predict(x)
#         return cross_entropy_error(y, t)

#     def accuracy(self, x: np.ndarray, t: np.ndarray):
#         c = self.predict(x)
#         y = np.argmax(c, axis=1)
#         t = np.argmax(t, axis=1)

#         accuracy = np.sum(y == t)/float(x.shape[0])
#         return accuracy

#     def numerical_gradient(self, x: np.ndarray, t: np.ndarray):
#         def loss_W(W): return self.loss(x, t)
#         grad = {}
#         W1, W2 = self.params['W1'], self.params['W2']
#         b1, b2 = self.params['b1'], self.params['b2']
#         grad['W1'] = numerical_gradient(loss_W, W1)
#         grad['b1'] = numerical_gradient(loss_W, b1)
#         grad['W2'] = numerical_gradient(loss_W, W2)
#         grad['b2'] = numerical_gradient(loss_W, b2)
#         return grad

#     def gradient(self, x, t):
#         W1, W2 = self.params['W1'], self.params['W2']
#         b1, b2 = self.params['b1'], self.params['b2']
#         grads = {}

#         batch_num = x.shape[0]

#         # forward
#         a1 = np.dot(x, W1) + b1
#         z1 = sigmoid(a1)
#         a2 = np.dot(z1, W2) + b2
#         y = softmax(a2)

#         # backward
#         dy = (y - t) / batch_num
#         grads['W2'] = np.dot(z1.T, dy)
#         grads['b2'] = np.sum(dy, axis=0)

#         dz1 = np.dot(dy, W2.T)
#         da1 = sigmoid_grad(a1) * dz1
#         grads['W1'] = np.dot(x.T, da1)
#         grads['b1'] = np.sum(da1, axis=0)

#         return grads
from common.functions import *
from common.gradient import numerical_gradient
import numpy as np


class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 重みの初期化
        self.params = {}
        self.params['W1'] = weight_init_std * \
            np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * \
            np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    # x:入力データ, t:教師データ
    def loss(self, x, t):
        y = self.predict(x)

        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # x:入力データ, t:教師データ
    def numerical_gradient(self, x, t):
        def loss_W(W): return self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}

        batch_num = x.shape[0]

        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)

        dz1 = np.dot(dy, W2.T)
        da1 = sigmoid_grad(a1) * dz1
        grads['W1'] = np.dot(x.T, da1)
        grads['b1'] = np.sum(da1, axis=0)

        return grads
