from PIL import Image, ImageDraw, ImageFont
from PyQt5.QtGui import QImage

import math
import numpy
import random

import config
import neural_network

def get_image(array, size=28) -> Image:
    """ Return a numpy array as a grayscale PIL Image"""
    return Image.fromarray(array.reshape(size,size), 'L')

def create_neural_network_image(neural_network: neural_network.NeuralNetwork, size=400) -> QImage:
    """ Draw a visual representation a Neural Network using PIL Draw. Returns a QImage """
    new_size = size * config.DRAW_ANTIALIASING_MSAA

    image = Image.new('RGB', (new_size,new_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    x_margin = 20
    y_margin = 20

    x_spacing = (new_size-x_margin*2)/len(neural_network._size)
    x_value = x_margin + x_spacing/2
    neurons = []
    display_activation = False
    if len(neural_network._activation) == len(neural_network._size):
        if not isinstance(neural_network._activation[0], int):
            display_activation = True
    for layer, neuron_layer_count in enumerate(neural_network._size):
        neurons.append([])
        y_spacing = (new_size-y_margin*2)/neuron_layer_count
        y_value = y_margin + y_spacing/2 - max(y_spacing-3, 2)/2
        for i in range(neuron_layer_count):
            if display_activation and neural_network._activation[layer].dtype == numpy.dtype('float64'):
                activation = round(float(neural_network._activation[layer][i]), 1)
            else:
                activation = 0.0
            radius = max(y_spacing-5, 2)
            neuron = (x_value, y_value, radius, activation)
            neurons[layer].append(neuron)
            y_value += y_spacing
        x_value += x_spacing

    if config.DRAW_NEURALNET_CONNECTIONS:
        for layer_count, neuron_layer in enumerate(neurons[:-1]):
            random.shuffle(neuron_layer)
            connection_count = 0
            for neuron in neuron_layer:
                if connection_count > config.DRAW_NEURALNET_CONNECTIONS_MAX_LIMIT:
                        break
                for neuron2 in neurons[layer_count+1]:
                    connection_count += 1
                    if connection_count > config.DRAW_NEURALNET_CONNECTIONS_MAX_LIMIT:
                        break
                    draw.line((neuron[0]+neuron[2]/2, neuron[1]+neuron[2]/2, neuron2[0]+neuron2[2]/2, neuron2[1]+neuron2[2]/2), fill = 'white', width=1)

    font = ImageFont.truetype("assets/ARIALBD.TTF", 30)  

    for neuron_layer in neurons:
        for neuron in neuron_layer:
            draw.ellipse((neuron[0], neuron[1], neuron[0]+neuron[2], neuron[1]+neuron[2]), fill = 'gray', outline ='gray')

            new_radius = neuron[2]*0.75
            offset = (neuron[2] - new_radius) / 2
            color = (int(255*(1- neuron[3])), int((255*neuron[3])), 0)
            draw.ellipse((neuron[0] + offset, neuron[1] + offset, neuron[0]+neuron[2] - offset, neuron[1]+ neuron[2] - offset), fill = color, outline =color)
            if neuron[2] > 18:
                #if len(neuron[3]) == 1:
                    #draw.text((neuron[0]+neuron[2]/2 - 8, neuron[1]+neuron[2]/2 - 8), neuron[3], font=font)
                #elif len(neuron[3]) == 2:
                draw.text((neuron[0]+neuron[2]/2 - 20, neuron[1]+neuron[2]/2 - 20), str(neuron[3]), font=font)


    image = image.resize((size, size),resample=Image.ANTIALIAS)

    return QImage(image.tobytes("raw","RGB"), image.size[0], image.size[1], QImage.Format_RGB888)