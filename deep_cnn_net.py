from collections import OrderedDict
from typing import TypedDict
import numpy as np
import pickle

from common.layers import Affine, Convolution, Dropout, Pooling, Relu, SoftmaxWithLoss


class Conv_Param(TypedDict):
    filter_num: int
    filter_size: int
    stride: int
    pad: int


class DeepCnnNet:
    def __init__(self, input_dim=(1, 28, 28),
                 conv_param1: Conv_Param = {
                     'filter_num': 16, 'filter_size': 3, 'stride': 1, 'pad': 1},
                 conv_param2: Conv_Param = {
                     'filter_num': 16, 'filter_size': 3, 'stride': 1, 'pad': 1},
                 conv_param3: Conv_Param = {
                     'filter_num': 32, 'filter_size': 3, 'stride': 1, 'pad': 1},
                 conv_param4: Conv_Param = {
                     'filter_num': 32, 'filter_size': 3, 'stride': 1, 'pad': 2},
                 conv_param5: Conv_Param = {
                     'filter_num': 64, 'filter_size': 3, 'stride': 1, 'pad': 1},
                 conv_param6: Conv_Param = {
                     'filter_num': 64, 'filter_size': 3, 'stride': 1, 'pad': 1},

                 hidden_size=100, output_size=10) -> None:
        input_size: int = input_dim[0]
        weight_init = np.array(
            [input_size*3*3, 16*3*3, 16*3*3, 32*3*3, 32*3*3, 64*3*3, 64*4*4, hidden_size])
        weight_init_he = np.sqrt(2/weight_init)
        self.params = {}
        for i, conv in enumerate([conv_param1, conv_param2, conv_param3, conv_param4, conv_param5, conv_param6]):
            WeightParamName = 'W'+str(i+1)
            biasParamName = 'b'+str(i+1)
            self.params[WeightParamName] = weight_init_he[i] * np.random.randn(
                conv['filter_num'], input_size, conv['filter_size'], conv['filter_size'])
            self.params[biasParamName] = np.zeros(conv['filter_num'])
            input_size = conv['filter_num']
        self.params['W7'] = weight_init_he[6] * \
            np.random.randn(64*4*4, hidden_size)
        self.params['b7'] = np.zeros(hidden_size)
        self.params['W8'] = weight_init_he[7] * \
            np.random.randn(hidden_size, output_size)
        self.params['b8'] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(
            self.params['W1'], self.params['b1'], conv_param1['stride'], conv_param1['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Conv2'] = Convolution(
            self.params['W2'], self.params['b2'], conv_param2['stride'], conv_param2['pad'])
        self.layers['Relu2'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2, pad=0)
        self.layers['Conv3'] = Convolution(
            self.params['W3'], self.params['b3'], conv_param3['stride'], conv_param3['pad'])
        self.layers['Relu3'] = Relu()
        self.layers['Conv4'] = Convolution(
            self.params['W4'], self.params['b4'], conv_param4['stride'], conv_param4['pad'])
        self.layers['Relu4'] = Relu()
        self.layers['Pool2'] = Pooling(pool_h=2, pool_w=2, stride=2, pad=0)
        self.layers['Conv5'] = Convolution(
            self.params['W5'], self.params['b5'], conv_param5['stride'], conv_param5['pad'])
        self.layers['Relu5'] = Relu()
        self.layers['Conv6'] = Convolution(
            self.params['W6'], self.params['b6'], conv_param6['stride'], conv_param6['pad'])
        self.layers['Relu6'] = Relu()
        self.layers['Pool3'] = Pooling(pool_h=2, pool_w=2, stride=2, pad=0)

        self.layers['Affine1'] = Affine(self.params['W7'], self.params['b7'])
        self.layers['Relu7'] = Relu()
        self.layers['Dropout1'] = Dropout(0.5)
        self.layers['Affine2'] = Affine(self.params['W8'], self.params['b8'])
        self.layers['Dropout2'] = Dropout(0.5)

        self.last_layer = SoftmaxWithLoss()
        pass

    def predict(self, x, train_flg=False):
        for layer in self.layers.values():
            if isinstance(layer, Dropout):
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)

        return x

    def loss(self, x, t, train_flg=False):
        y = self.predict(x, train_flg)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        acc = 0.0
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        for i in range(int(x.shape[0]/batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)
        accuracy = acc/x.shape[0]
        return accuracy

    def gradient(self, x, t):
        self.loss(x, t, train_flg=True)

        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()

        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        for i, layer_idx in enumerate(('Conv1', 'Conv2', 'Conv3', 'Conv4', 'Conv5', 'Conv6', 'Affine1', 'Affine2')):
            WeightGradName = 'W'+str(i+1)
            biasGradName = 'b'+str(i+1)
            grads[WeightGradName] = self.layers[layer_idx].dW
            grads[biasGradName] = self.layers[layer_idx].db

        return grads

    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, key in enumerate(['Conv1', 'Conv2', 'Conv3', 'Conv4', 'Conv5', 'Conv6', 'Affine1', 'Affine2']):
            self.layers[key].W = self.params['W' + str(i+1)]
            self.layers[key].b = self.params['b' + str(i+1)]
