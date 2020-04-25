import numpy

from dataset import *
import visualiser
import gui

def main():
    print("Program started")
    dataset = DataSet()

    training_image = dataset.get_training_image_array(0)
    training_label = dataset.get_training_image_label(0)


    test_image = dataset.get_test_image_array(0)
    test_label = dataset.get_test_image_label(0)

    #visualiser.display_image(test_image, test_label)

    #application = gui.MainApplication()


    print("Program exited")