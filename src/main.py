import numpy, random, neural_network

from dataset import *
import visualiser

def main():
    print("Program started")
    dataset = DataSet()

    training_image = dataset.get_training_image_array(0)
    training_label = dataset.get_training_image_label(0)




    test_image = dataset.get_test_image_array(0)
    test_label = dataset.get_test_image_label(0)

    #visualiser.display_image(test_image, test_label)
    nn = neural_network.NeuralNetwork([784, 784, 10])
    train_model_MNIST(nn)
    test_model_MNIST(nn)

    print("Program exited")


def train_model_MNIST(model):
    dataset = DataSet()
    training_data = []
    for i in range(60000):
        expected = numpy.zeros((10, 1))
        expected[dataset.get_training_image_label(i)] = 1
        training_data.append( 
            (dataset.get_training_image_array(i).reshape([784, 1])
            , expected)
            )
    model.train(training_data, 0.01, 1)


def test_model_MNIST(model):
    total = 10000
    dataset = DataSet()
    correct = 0
    for i in range(10000):
        model.forward_propagate(dataset.get_test_image_array(i).reshape([28*28, 1]))
        res = model._activation[-1]
        res = res == max(res)
        if res[dataset.get_test_image_label(i)]:
            correct += 1
    print("Accuracy: ", round(correct / total * 100), " %")