import numpy
from config import *

class DataSet():
    """ Represents the MNIST Dataset. Loads the dataset and simplifies data retrival """
    def __init__(self):
        """ Loads the dataset """
        self.training_data = self._load_numpy_array_from_file("train-images.idx3-ubyte")
        self.training_data_labels = self._load_numpy_array_from_file("train-labels.idx1-ubyte")
        self.test_data = self._load_numpy_array_from_file("t10k-images.idx3-ubyte")
        self.test_data_labels = self._load_numpy_array_from_file("t10k-labels.idx1-ubyte")

    def _load_numpy_array_from_file(self, path : str):
        """ Loads the dataset into a numpy array from the assets/ folder """
        return numpy.fromfile("assets/"+path, dtype='B')

    def get_training_image_array(self, image_count : int):
        """ Get a 28*28 numpy array of a training image from MNIST. Data starts after index 16 (16 bytes) """
        image_start = 16 + image_count * IMAGE_WIDTH*IMAGE_WIDTH
        image_end = image_start + IMAGE_WIDTH*IMAGE_WIDTH
        return self.training_data[image_start:image_end]

    def get_training_image_label(self, image_count: int) -> int:
        """ Get the integer label of a training image from MNIST. Data starts after index 16 (16 bytes) """
        index = 8 + image_count
        return int(self.training_data_labels[index])

    def get_test_image_array(self, image_count : int):
        """ Get a 28*28 numpy array of a test image from MNIST. Data starts after index 16 (16 bytes) """
        image_start = 16 + image_count * IMAGE_WIDTH*IMAGE_WIDTH
        image_end = image_start + IMAGE_WIDTH*IMAGE_WIDTH
        return self.test_data[image_start:image_end]

    def get_test_image_label(self, image_count: int) -> int:
        """ Get the integer label of a testing image from MNIST. Data starts after index 16 (16 bytes) """
        index = 8 + image_count
        return int(self.test_data_labels[index])