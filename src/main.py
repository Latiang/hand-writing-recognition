import numpy
import PIL

from dataset import *

def main():
    print("Program started")
    dataset = DataSet()

    training_image = dataset.get_training_image_array(0)
    training_label = dataset.get_training_image_label(0)


    test_image = dataset.get_training_image_array(0)
    test_label = dataset.get_training_image_label(0)


    print("Program exited")