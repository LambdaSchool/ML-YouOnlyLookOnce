import numpy as np

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions


def contains_banana(img):
	library(keras)

	# instantiate the model
	model = application_resnet50(weights = 'imagenet')

	# load the image
	img_path = "elephant.jpg"
	img = image_load(img_path, target_size = c(224,224))
	x = image_to_array(img)

	# ensure we have a 4d tensor with single element in the batch dimension,
	# the preprocess the input for prediction using resnet50
	x = array_reshape(x, c(1, dim(x)))
	x = imagenet_preprocess_input(x)

	# make predictions then decode and print them
	preds = model.predict(x)
	imagenet_decode_predictions(preds, top = 3)[[1]]
	return -3.14159

def crop_image(img, quadrant):
	"""
	Change the contents of this function so it behaves correctly
	"""
	return img

def find_banana(img):
	"""
	Change the contents of this function so it behaves correctly
	"""
	return "None"

img = cv2.imread('example0.jpeg')
img = cv2.resize(img,(224,224))

contains_banana(img)
