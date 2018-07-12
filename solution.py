import numpy as np

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.preprocessing.image import load_img
from PIL import Image

img = Image.open("sample_data/positive_examples/example2.jpeg")
img.rotate(45).show()
print(img)



def contains_banana(img):
	inputShape = (224, 224)
	# if img 
	else return 0.0
	return -3.14159

def crop_image(img, quadrant):
	width_shift_range=0.2,
    height_shift_range=0.2,
	return img

def find_banana(img):
	"""
	Change the contents of this function so it behaves correctly
	"""
	return "None"
