import numpy

IMAGE_WIDTH = 28

class DataSet():
    def __init__(self):
        self.training_data = self.load_numpy_array_from_file("train-images.idx3-ubyte")
        self.training_data_labels = self.load_numpy_array_from_file("train-labels.idx1-ubyte")
        self.test_data = self.load_numpy_array_from_file("t10k-images.idx3-ubyte")
        self.test_data_labels = self.load_numpy_array_from_file("t10k-labels.idx1-ubyte")

    def load_numpy_array_from_file(self, path):
        return numpy.fromfile("assets/"+path, dtype='B')

    def get_training_image_array(self, image_count : int):
        image_start = 8 + image_count * IMAGE_WIDTH*IMAGE_WIDTH
        image_end = image_start + IMAGE_WIDTH*IMAGE_WIDTH
        return self.training_data[image_start:image_end]

    def get_training_image_label(self, image_count):
        index = 8 + image_count
        return int(self.training_data_labels[index])

    def get_test_image_array(self, image_count : int):
        image_start = 8 + image_count * IMAGE_WIDTH*IMAGE_WIDTH
        image_end = image_start + IMAGE_WIDTH*IMAGE_WIDTH
        return self.test_data[image_start:image_end]

    def get_test_image_label(self, image_count):
        index = 8 + image_count
        return int(self.test_data_labels[index])