import numpy as np

import random
from typing import *

def print_training_progress(percentage_complete):
    print("The training progress is at {:.2f} %".format(percentage_complete*100))

class ActivationFunction:
    def __init__(self, function, derivative):
        self.function = function
        self.derivative = derivative


def _rectified_linear(x):
    return np.maximum(0.01*x, x)


def _rectified_linear_prim(x):
    return (x > 0) * 1.0 + (x <= 0) * 0.01


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))

def _sigmoid_prim(x):
    y = _sigmoid(x)
    return y * (1 - y)

SIGMOID = ActivationFunction(_sigmoid, _sigmoid_prim)
ReLU = ActivationFunction(_rectified_linear, _rectified_linear_prim)


class NeuralNetwork:
    def __init__(self, size: List[int], activation_function: ActivationFunction = SIGMOID):
        self._layers = []
        self._sum = [0] * (len(size))
        self._activation = [0] * (len(size))
        self._size = size
        self._activation_function = activation_function
        self._gradient = []
        for i in range(len(size) - 1):
            target_size = (size[i + 1] + (i != (len(size) - 2)), size[i] + 1)
            self._layers.append(np.random.standard_normal(target_size) / 1000)
            self._gradient.append(np.zeros_like(self._layers[-1]))
        


    def forward_propagate(self, inp: np.array):
        """Calculate expected output from given input"""
        if inp.size != self._size[0] + 1:
            inp = np.append(inp, 1)
        current = inp
        self._sum[0] = inp
        self._activation[0] = inp
        for i in range(1, len(self._size) - 1):
            current = self._layers[i - 1] @ current
            self._sum[i] = current
            current = self._activation_function.function(current)
            self._activation[i] = current
            self._activation[i][-1] = 1
        current = self._layers[-1] @ current
        self._sum[-1] = current
        self._activation[-1] = _sigmoid(current)


    def size(self):
        """Returns the size of the NN"""
        return self._size

    def _backpropagate(self, err: np.array):
        """performs backprogation given a error"""
        pcurr = err

        p_times_derivative = pcurr * _sigmoid_prim(self._sum[-1])
        pnext = self._layers[-1].transpose() @ p_times_derivative

        self._gradient[-1] = self._gradient[-1] + p_times_derivative @ self._activation[-2].transpose()

        pcurr = pnext
        for i in range(len(self._activation) - 2, 0, -1):
            pcurr[-1] = 0       #The derivative with respect to the last value is 0 because it is locked to 1
                                #This change will probably not affect much but we try to see if it makes any difference
                                # Tests seems to indicate that there is no advantage nor any disadvantage with it being set to 0 or not
            p_times_derivative = pcurr * self._activation_function.derivative(self._sum[i])
            pnext = self._layers[i - 1].transpose() @ p_times_derivative

            self._gradient[i - 1] = self._gradient[i - 1] + p_times_derivative @ self._activation[i - 1].transpose()

            pcurr = pnext

    def _gradient_reset(self):
        for i in range(len(self._layers)):
            self._gradient[i] = np.zeros_like(self._layers[i])

    def _update_network(self, training_rate: float, batches: int):
        multiplier = training_rate / batches
        for i in range(len(self._layers)):
            self._layers[i] = self._layers[i] - multiplier * self._gradient[i]
        self._gradient_reset()

    def train(self, cases: List[Tuple[np.array, np.array]], training_rate: float, epochs: int, batch_size: int, progress_update_function=print_training_progress):
        """Perform backpropagation on the given cases"""
        for index, case in enumerate(cases):
            cases[index] = (np.append(case[0], 1).reshape([case[0].size + 1, 1]), case[1])
        total_num = epochs * len(cases)
        update_progess_threshold = 0.05
        done = 0
        batches = 0
        for _j in range(epochs):
            random.shuffle(cases)
            cost_sum = 0
            for case in cases:
                batches += 1
                inp = case[0]
                out = case[1]
                self.forward_propagate(inp)
                error = self._activation[-1] - out
                cost_sum += sum(error * error)
                self._backpropagate(error)
                done += 1
                if done / total_num >= update_progess_threshold:
                    percentage_complete = done/total_num
                    progress_update_function(percentage_complete)
                    update_progess_threshold += 0.05
                if batches >= batch_size:
                    self._update_network(training_rate, batches)
                    batches = 0
        print("Total error: ", cost_sum)
        print("Average error: ", cost_sum / total_num)
        return (cost_sum / total_num)[0]

def test():
    cases = [([0, 0], [0]), ([0, 1], [1]), ([1, 0], [1]), ([1, 1], [0])]
    for i in range(len(cases)):
        cases[i] = (np.array(cases[i][0]).reshape(2, 1), np.array(cases[i][1]).reshape(1, 1))

    nn = NeuralNetwork([2, 784, 1])
    nn.train(cases, 0.1, 100000, 1)

if __name__ == "__main__":
    test()