import numpy as np
import matplotlib.pyplot as plt
from common.functions import relu, sigmoid


def tanh(x):
    return np.tanh(x)


input_data = np.random.randn(1000, 100)
node_num = 100
hidden_layer_size = 5

activations = {}

x = input_data

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1]

    w = np.random.randn(node_num, node_num)*np.sqrt(2/node_num)

    a: np.ndarray = np.dot(x, w)

    z = relu(a)

    activations[i] = z

for i, a in activations.items():
    plt.subplot(1, len(activations), i+1)
    plt.title(str(i+1)+"-layer")
    if i != 0:
        plt.yticks([], [])

    plt.hist(a.flatten())

plt.show()
