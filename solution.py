import numpy as np

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from PIL import Image


def contains_banana(img):
    print("analyzing: ", img)
    original = image.load_img(img, target_size=(224, 224))
    numpy_image = image.img_to_array(original)
    input_image = preprocess_input(np.expand_dims(numpy_image, axis=0))
    resnet = ResNet50(weights='imagenet')
    preds = resnet.predict(input_image)
    labels = decode_predictions(preds, top=5)
    for each in labels[0]:
        if each[1] == 'banana':
            return each[2]
    return 0.0

# coordinate system: left, upper, right, and lower
def crop_image(img, quadrant):
    original = image.load_img(img, target_size=(224, 224))
    w, h = original.size
    if quadrant == 'TL':
        cropped_img = original.crop((0, 0, w*2/3, h*2/3))
    elif quadrant == 'TR':
        cropped_img = original.crop((w*1/3, 0, w*2/3, h*2/3))
    elif quadrant == 'BL':
        cropped_img = original.crop((0, w*1/3, w*2/3, h*2/3))
    elif quadrant == 'BR':
        cropped_img = original.crop((w*1/3, w*1/3, w*2/3, h*2/3))

    print(cropped_img.getbbox())
    return img

def find_banana(img):
	"""
	Change the contents of this function so it behaves correctly
	"""
	return "None"

# print(contains_banana('./sample_data/positive_examples/example0.jpeg'))
# print(contains_banana('./sample_data/positive_examples/example1.jpeg'))
# print(contains_banana('./sample_data/positive_examples/example2.jpeg'))
# print(contains_banana('./sample_data/negative_examples/example10.jpeg'))
# print(contains_banana('./sample_data/negative_examples/example11.jpeg'))
# print(contains_banana('./sample_data/negative_examples/example12.jpeg'))

crop_image('./sample_data/positive_examples/example0.jpeg', 'TL')
crop_image('./sample_data/positive_examples/example0.jpeg', 'TR')
crop_image('./sample_data/positive_examples/example0.jpeg', 'BL')
crop_image('./sample_data/positive_examples/example0.jpeg', 'BR')