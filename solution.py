import numpy as np

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions

model = ResNet50(weights='imagenet')

neg =  "C:/Users/Smartsrc-alex/Documents/ML_class/ML-YouOnlyLookOnce/sample_data/negative_examples"
pos =  "C:/Users/Smartsrc-alex/Documents/ML_class/ML-YouOnlyLookOnce/sample_data/postive_examples"

negative_ex = [neg + 'example10',neg + 'example11',neg + 'example12']

postive_ex = [pos + 'example0',pos + 'example1',pos + 'example2']

def contains_banana(img):

	preds = model.predict(img)

	print(preds)

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



contains_banana(negative_ex[0])