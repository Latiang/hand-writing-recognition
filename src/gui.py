""" GUI Implementation for the project using Qt5 Python bindings """

from PIL.ImageQt import ImageQt
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QHBoxLayout, QVBoxLayout, QPushButton, QProgressBar, QSpinBox, QDoubleSpinBox, QFrame, QGridLayout, QRadioButton
from PyQt5.QtGui import QIcon, QPixmap, QFont, QImage
from PyQt5 import QtCore

import numpy

import sys

import main
import config
import visualiser
import neural_network

class MainWindow(QWidget):
    def __init__(self, dataset, neural_network):
        """ Initiate the GUI Window and create all the relevant Widgets """
        super(MainWindow, self).__init__()

        self.current_dataset_image = 0
        self.average_error = 0
        self.image_indices = [0]
        self.image_full_indices = [0]
        self.image_incorrect_indices = []
        self.filterAllowAllImages()

        self.dataset = dataset
        self.neural_network = neural_network

        # Setup Window properties
        self.setWindowTitle('Hand-written Digit (MNIST) Recognition through Machine Learning')
        self.setFixedSize(825, 590)
        self.setStyleSheet("background-color: #181818; color: white")

        self.main_layout = QHBoxLayout()

        self.setupLeftPanel()

        # Line separator between Left and Right Panel
        separator_line = QFrame()
        separator_line.setFrameShape(QFrame.VLine)
        separator_line.setStyleSheet("QFrame { color : #535353; }")
        separator_line.setLineWidth(3)
        self.main_layout.addWidget(separator_line)

        self.setupRightPanel()

        self.setLayout(self.main_layout)
        
    def setupLeftPanel(self):
        """ Setup the Left Panel containing the current active Image for MNIST as well as some image controls """
        # Example layout box, vertical
        left_panel_layout = QVBoxLayout()

        # Image display
        self.image_label = QLabel(self)
        #image_label.resize(200, 200)
        left_panel_layout.addWidget(self.image_label)
        
        # Image controls
        image_controls_layout = QHBoxLayout()
        # Previous image
        previous_button = QPushButton('<----')
        previous_button.setFixedHeight(20)
        previous_button.clicked.connect(lambda x: self.previousImage())
        image_controls_layout.addWidget(previous_button)

        # Next image
        next_button = QPushButton('---->')
        next_button.setFixedHeight(20)
        next_button.clicked.connect(lambda x: self.nextImage())
        image_controls_layout.addWidget(next_button)

        left_panel_layout.addLayout(image_controls_layout)

        # Number label data
        number_labels_layout = QHBoxLayout()
        #number_labels_layout.setAlignment(Qt.AlignTop)

        number_font = QFont()
        number_font.setPointSize(40)
        number_font.setBold(True)
        # Dataset number label
        self.dataset_label = QLabel()
        self.dataset_label.setAlignment(Qt.AlignHCenter)
        self.dataset_label.setStyleSheet("QLabel { color : green; }")
        self.dataset_label.setFont(number_font)
        number_labels_layout.addWidget(self.dataset_label)

        # Predicted label
        self.predicted_label = QLabel()
        self.predicted_label.setAlignment(Qt.AlignHCenter)
        self.predicted_label.setStyleSheet("QLabel { color : RoyalBlue; }")
        self.predicted_label.setFont(number_font)
        number_labels_layout.addWidget(self.predicted_label)

        self.updateDatasetImage()

        left_panel_layout.addLayout(number_labels_layout)

        # Number label data text description
        number_text_layout = QHBoxLayout()

        dataset_text_label = QLabel("Actual")
        dataset_text_label.setAlignment(Qt.AlignHCenter)
        number_text_layout.addWidget(dataset_text_label)

        predicted_text_label = QLabel("Predicted")
        predicted_text_label.setAlignment(Qt.AlignHCenter)
        number_text_layout.addWidget(predicted_text_label)

        left_panel_layout.addLayout(number_text_layout)

        image_filter_buttons_layout = QHBoxLayout()
        image_filter_buttons_layout.setAlignment(Qt.AlignLeft)

        show_all_radiobutton = QRadioButton()
        show_all_radiobutton.setText("Show All")
        show_all_radiobutton.setStyleSheet("QRadioButton::indicator::unchecked {  border-radius:5px;   border-style: solid; border-width:1px;; border-color: gray;}")
        show_all_radiobutton.clicked.connect(lambda x: self.filterAllowAllImages())

        show_only_incorrect_radiobutton = QRadioButton()
        show_only_incorrect_radiobutton.setText("Show Only Incorrect")
        show_only_incorrect_radiobutton.setStyleSheet("QRadioButton::indicator::unchecked {  border-radius:5px;   border-style: solid; border-width:1px;; border-color: gray;}")
        show_only_incorrect_radiobutton.clicked.connect(lambda x: self.filterAllowOnlyIncorrectImages())

        image_filter_buttons_layout.addWidget(show_all_radiobutton)
        image_filter_buttons_layout.addWidget(show_only_incorrect_radiobutton)
        

        left_panel_layout.addLayout(image_filter_buttons_layout)

        self.main_layout.addLayout(left_panel_layout)
    
    def setupRightPanel(self):
        """ Setup the Right Panel containing buttons and settings for controlling the NN training and NN visualisation image """
        right_panel_layout = QVBoxLayout()

        # Neural Network Image Display
        self.nn_image_label = QLabel(self)
        self.updateNeuralNetworkImage()
        right_panel_layout.addWidget(self.nn_image_label)
        self.testCurrentImage()
        self.updateNeuralNetworkImage()

        # Train button
        network_related_layout = QHBoxLayout()
        network_related_layout.setAlignment(Qt.AlignBottom)

        training_settings_layout = QVBoxLayout()
        training_settings_layout.setAlignment(Qt.AlignBottom)

        training_info_layout = QVBoxLayout()
        training_info_layout.setAlignment(Qt.AlignBottom)


        self.training_output_layout = QHBoxLayout()
        self.training_output_layout.setAlignment(Qt.AlignHCenter)

        # Success rate
        self.success_rate_number = QLabel("Accuracy: 0 %")
        self.training_output_layout.addWidget(self.success_rate_number)

        # Error rate
        self.average_error_number = QLabel("Avg. Error: 0")
        self.training_output_layout.addWidget(self.average_error_number)

        training_info_layout.addLayout(self.training_output_layout)

        # Train Button
        train_button = QPushButton('Train')
        train_button.clicked.connect(lambda x: self.trainNeuralNetwork())
        training_info_layout.addWidget(train_button)

        # Progress bar
        self.progressBar = QProgressBar(self)
        self.progressBar.setGeometry(30, 40, 100, 25)
        training_info_layout.addWidget(self.progressBar)

        # Divider line between the train and input section of the right panel
        separator_line = QFrame()
        separator_line.setFrameShape(QFrame.VLine)
        separator_line.setStyleSheet("QFrame { color : #535353; }")
        separator_line.setLineWidth(2)

        # Hidden Neuron Layer input
        neurons_input_layout = QHBoxLayout()
        neurons_label = QLabel("Neurons")
        #training_settings2_layout.addWidget(neurons_label)
        self.neurons_input = QSpinBox()
        self.neurons_input.setMaximum(1000)
        self.neurons_input.setValue(self.neural_network._size[1])
        self.neurons_input.valueChanged.connect(lambda x: self.updateNeuralNetworkSize())
        neurons_input_layout.addWidget(neurons_label)
        neurons_input_layout.addWidget(self.neurons_input)
        training_settings_layout.addLayout(neurons_input_layout)

        # Epoch input
        epoch_input_layout = QHBoxLayout()
        epoch_label = QLabel("Epochs")
        self.epoch_input = QSpinBox()
        self.epoch_input.setValue(1)
        epoch_input_layout.addWidget(epoch_label)
        epoch_input_layout.addWidget(self.epoch_input)
        training_settings_layout.addLayout(epoch_input_layout)

        # Batches input
        batches_input_layout = QHBoxLayout()
        batches_label = QLabel("Batches")
        self.batches_input = QSpinBox()
        batches_input_layout.addWidget(batches_label)
        batches_input_layout.addWidget(self.batches_input)
        training_settings_layout.addLayout(batches_input_layout)

        # Learning Rate input
        learning_rate_input_layout = QHBoxLayout()
        learning_rate_label = QLabel("Learning Rate")
        self.learning_rate_input = QDoubleSpinBox()
        self.learning_rate_input.setDecimals(4)
        self.learning_rate_input.setValue(0.001)
        learning_rate_input_layout.addWidget(learning_rate_label)
        learning_rate_input_layout.addWidget(self.learning_rate_input)
        training_settings_layout.addLayout(learning_rate_input_layout)

        # Add all the sub layouts together
        network_related_layout.addLayout(training_info_layout)
        network_related_layout.addWidget(separator_line)
        network_related_layout.addLayout(training_settings_layout)
        right_panel_layout.addLayout(network_related_layout)

        self.main_layout.addLayout(right_panel_layout)

    def updateNeuralNetworkSize(self):
        """ Update the size of the hidden layer of the Neural Network, signal for neurons_input """
        neuron_count = self.neurons_input.value()
        print("Updating Neural Network size to {}".format(neuron_count))
        self.neural_network = neural_network.NeuralNetwork([784, neuron_count, 10])
        self.updateNeuralNetworkImage()
    
    def nextImage(self):
        """ Go to the next image from MNIST, signal for next_button """
        self.current_dataset_image = min(self.current_dataset_image + 1, len(self.image_indices))
        self.updateDatasetImage()
        self.testCurrentImage()
        print("Switched to image {}".format(self.current_dataset_image))

    def previousImage(self):
        """ Go to the previous image from MNIST, signal for previous_button """
        self.current_dataset_image = max(self.current_dataset_image - 1, 0)
        self.updateDatasetImage()
        self.testCurrentImage()
        print("Switched to image {}".format(self.current_dataset_image))

    def filterAllowAllImages(self):
        """ Function for allowing all images from MNIST to be picked for the Left Panel viewer, signal for show_all_radiobutton"""
        self.current_dataset_image = 0
        self.image_indices = range(config.TRAINING_SET_IMAGE_COUNT)
        #self.updateDatasetImage()

    def filterAllowOnlyIncorrectImages(self):
        """ Function for allowing only incorrectly recognized images from MNIST for the Left Panel viewer, signal for show_only_incorrect_radiobutton"""
        self.current_dataset_image = 0
        self.image_indices = self.image_incorrect_indices
        self.updateDatasetImage()

    def updateDatasetImage(self):
        """ Updates the MNIST dataset Image in the left panel based on the current_dataset_image counter """
        image_number_label = self.dataset.get_test_image_label(self.image_indices[self.current_dataset_image])
        image = visualiser.get_image(self.dataset.get_test_image_array(self.image_indices[self.current_dataset_image]))
        q_image = ImageQt(image)
        pixmap = QPixmap.fromImage(q_image)
        pixmap = pixmap.scaled(400, 400)
        self.image_label.setPixmap(pixmap)
        self.dataset_label.setText(str(image_number_label))
        self.predicted_label.setText(str(0))

    def updateNeuralNetworkImage(self):
        """ Updates the Neural Network Visualisation in the right panel """
        if config.DRAW_NEURALNET_ENABLED:
            nn_q_image = visualiser.create_neural_network_image(self.neural_network, 400)
            nn_pixmap = QPixmap.fromImage(nn_q_image)
            self.nn_image_label.setPixmap(nn_pixmap)

    def updateTrainingProgressCounter(self, progress):
        """ Updates the Progress Bar for Training in the right hand panel. It is sent as a function to the NN for automatic updating"""
        self.progressBar.setValue(progress*100)
        #self.updateNeuralNetworkImage()

    def trainNeuralNetwork(self):
        """ Train the Neural Network using the parameters specified in the right panel on the MNIST dataset, see main.py for more details """
        self.updateTrainingProgressCounter(0)
        self.average_error = main.train_model_MNIST(self.neural_network, self.dataset, float(self.learning_rate_input.value()), int(self.epoch_input.value()), self.updateTrainingProgressCounter)
        self.updateTrainingProgressCounter(1)
        self.testAllImages()
        self.updateNeuralNetworkImage()

    def testCurrentImage(self):
        """ Test the Neural Network on the currently selected Image from the MNIST Dataset """
        number = main.test_single_image_MINST(self.image_indices[self.current_dataset_image], self.neural_network, self.dataset)
        self.predicted_label.setText(str(number))
        self.updateNeuralNetworkImage()

    def testAllImages(self):
        """ Test the Neural Network on the MNIST test dataset and update relevant label fields, see main.py for more details """
        success_rate, self.image_incorrect_indices = main.test_model_MNIST(self.neural_network, self.dataset)
        success_rate_string = "Accuracy: {:.2f} %".format(success_rate * 100)
        self.average_error_number.setText("Avg. Error: {:.3f}".format(self.average_error))
        self.success_rate_number.setText(success_rate_string)

class MainApplication(QApplication):
    def __init__(self, dataset, neural_network):
        super().__init__([])

        window = MainWindow(dataset, neural_network)
        window.show()

        sys.exit(self.exec_())