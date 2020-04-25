import numpy
import PIL

from dataset import *

def main():
    print("Program started")
    dataset = DataSet()

    test = dataset.get_training_image_array(1)

    print("Program exited")