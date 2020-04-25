import numpy
import PIL

from dataset import *

def main():
    print("Program started")
    dataset = DataSet()

    test_image = dataset.get_training_image_array(2)
    test_label = dataset.get_training_image_label(2)

    print("Program exited")