import numpy

import random
import time

from dataset import *
import neural_network
import gui
import config

def main():
    print("Program started")

    dataset = DataSet()
    nn = neural_network.NeuralNetwork([784, 30, 10])

    if config.GUI_ENABLED:
        application = gui.MainApplication(dataset, nn)

    train_model_MNIST(nn, dataset, 0.01, 1, 1)
    test_model_MNIST(nn, dataset)

    print("Program exited")

def train_model_MNIST(model, dataset, learning_rate=0.01, epochs=1, batch_size = 1, status_function=neural_network.print_training_progress):
    """ Train a Neural Network Model on the MNIST Training Dataset. Returns the Average Error """
    print("Training Neural Network on MNIST Dataset")
    training_data = []
    for i in range(60000):
        expected = numpy.zeros((10, 1))
        expected[dataset.get_training_image_label(i)] = 1
        training_data.append( 
            (dataset.get_training_image_array(i).reshape([784, 1])
            , expected)
            )
    #avg_error = model.train(training_data, 0.01, 1, status_function)
    avg_error = model.train(training_data, learning_rate, epochs, batch_size, status_function)
    return avg_error

def test_model_MNIST(model : neural_network.NeuralNetwork, dataset) -> (float, list):
    """ 
    Test a Neural Network Model on the MNIST Test Dataset.
    Returns the success rate as a float and a list of indices for all images incorrectly recognized
    """
    print("Testing Neural Network on MNIST Test Dataset")
    total = 10000
    correct = 0
    incorrect_indices = []
    for i in range(10000):
        num = test_single_image_MINST(i, model, dataset)
        if num == dataset.get_test_image_label(i):
            correct += 1
        else:
            incorrect_indices.append(i)
    success_rate = correct / total
    print("Accuracy: {:.3f} %".format(success_rate * 100))
    return success_rate, incorrect_indices

def test_single_image_MINST(i, model: neural_network.NeuralNetwork , dataset : DataSet) -> int:
    """ Test a single image from MNIST on the Neural Network Model """
    model.forward_propagate(dataset.get_test_image_array(i).reshape([28*28, 1]))
    res = model._activation[-1]
    return numpy.nonzero(res == max(res))[0][0]