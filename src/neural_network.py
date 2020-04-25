import numpy as np
import random
from typing import *

class ActivationFunction:
    def __init__(self, function, derivative):
        self.function = function
        self.derivative = derivative


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prim(x):
    y = sigmoid(x)
    return y * (1 - y)

class NeuralNetwork:
    def __init__(self, size: List[int], activation_function: ActivationFunction = ActivationFunction(sigmoid, sigmoid_prim)):
        self._layers = []
        self._sum = [0] * (len(size))
        self._activation = [0] * (len(size))
        self._size = size
        self._activation_function = activation_function
        for i in range(len(size) - 1):
            self._layers.append((np.random.rand(size[i + 1], size[i]) - 0.5) / 2)
        
    def forward_propagate(self, inp):
        """Calculate expected output from given input"""
        current = inp
        self._sum[0] = inp
        self._activation[0] = inp
        for i in range(1, len(self._size)):
            current = self._layers[i - 1] @ current
            self._sum[i] = current
            current = self._activation_function.function(current)
            self._activation[i] = current


    def size(self):
        """Returns the size of the NN"""
        return self._size

    def _backpropagate(self, err, training_rate):
        """performs backprogation given a error"""
        pcurr = err
        for i in range(len(self._activation) - 1, 0, -1):
            p_times_derivative = pcurr * self._activation_function.derivative(self._sum[i])
            pnext = self._layers[i - 1].transpose() @ p_times_derivative
            #print(p_times_derivative)
            #print(self._activation[i - 1])
            #print(p_times_derivative @ self._activation[i - 1].transpose())
            #print()
            self._layers[i - 1] = self._layers[i - 1] - training_rate * p_times_derivative @ self._activation[i - 1].transpose()
            pcurr = pnext

    def train(self, cases: List[Tuple[np.array, np.array]], training_rate, trials):
        """Perform backpropagation on the given cases"""
        for _j in range(trials):
            random.shuffle(cases)
            cost_sum = 0
            for case in cases:
                inp = case[0]
                out = case[1]
                self.forward_propagate(inp)
                error = self._activation[-1] - out
                cost_sum += sum(error * error)
                self._backpropagate(error, training_rate)
        print("Total error: ", cost_sum)
        print("Average error: ", cost_sum / len(cases))
