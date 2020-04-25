import numpy, random, neural_network, time

from dataset import *
import visualiser
import gui

def main():
    print("Program started")
    dataset = DataSet()

    #training_image = dataset.get_training_image_array(0)
    #training_label = dataset.get_training_image_label(0)




    #test_image = dataset.get_test_image_array(0)
    #test_label = dataset.get_test_image_label(0)

    #visualiser.display_image(test_image, test_label)

    #application = gui.MainApplication()

    start = time.time()

    nn = neural_network.NeuralNetwork([784, 784, 10])

    train_model_MNIST(nn)
    
    end = time.time()
    print("Neural Network training took {}".format(end - start))

    print("Program exited")


def train_model_MNIST(model):
    dataset = DataSet()
    training_data = []
    for i in range(6000):
        expected = numpy.zeros((10, 1))
        expected[dataset.get_training_image_label(i)] = 1
        training_data.append( 
            (dataset.get_training_image_array(i).reshape([784, 1])
            , expected)
            )
    model.train(training_data, 0.01, 1)
    model.train(training_data, 0.01, 1)
    model.train(training_data, 0.01, 1)