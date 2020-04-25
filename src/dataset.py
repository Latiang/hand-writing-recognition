import numpy

class DataSet():
    def __init__(self):
        self.training_data = self.load_numpy_array_from_file("train-images.idx3-ubyte")
        self.training_data_labels = self.load_numpy_array_from_file("train-labels.idx1-ubyte")
        self.test_data = self.load_numpy_array_from_file("t10k-images.idx3-ubyte")
        self.test_data_labels = self.load_numpy_array_from_file("t10k-labels.idx1-ubyte")

    def load_numpy_array_from_file(self, path):
        return numpy.fromfile("assets/"+path, dtype='B')