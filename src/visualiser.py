from PIL import Image

def display_image(array, label):
    """ Displays a numpy array image"""
    print("Displaying image of digit {}".format(label))
    image = Image.fromarray(array.reshape(28,28), 'L')
    image.show()