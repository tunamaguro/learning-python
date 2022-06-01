import numpy as np
from sigmoid import sigmoid


def identity_function(x: np.ndarray):
    return x


def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['B1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['B2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['B3'] = np.array([[0.1, 0.2]])
    return network


def foward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    B1, B2, B3 = network['B1'], network['B2'], network['B3']
    a1 = np.dot(x, W1)+B1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2)+B2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3)+B3
    y = identity_function(a3)
    return y


network = init_network()
x = np.array([1.0, 0.5])
y = foward(network, x)
print(y)
