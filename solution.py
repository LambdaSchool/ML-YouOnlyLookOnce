import numpy as np
import matplotlib.pyplot as plt

from keras.backend import tf as ktf
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions


def contains_banana(img):
    datagen = ImageDataGenerator()

    img = load_img('C:/users/solar/Desktop/' + img)
    img = ktf.image.resize_images(img, (224, 224))
    img_array = img_to_array(img)
    print(img_array.shape)
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.axis('off')
    plt.show()
    return img


def crop_image(img, quadrant):
    l = img
    return img


def find_banana(img):
    return "None"

contains_banana('example0.jpeg')
