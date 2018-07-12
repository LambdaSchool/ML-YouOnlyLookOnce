import numpy as np
import matplotlib.pyplot as plt

from keras.backend import tf as ktf
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions


def contains_banana(img):
    img = load_img('C:/Users/solar/Desktop/' + img, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)

    model = ResNet50(weights='imagenet')
    features = model.predict(img)
    results = decode_predictions(features, top=3)
    for r in results:
        return r[2] if r[1] == 'banana' else 0


def crop_image(img, quadrant):
    l = img
    return img


def find_banana(img):
    return "None"

print('Banana confidence = {}'.format(contains_banana('example0.jpeg')))
