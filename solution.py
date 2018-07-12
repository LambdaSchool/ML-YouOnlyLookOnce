import numpy as np
from pathlib import Path
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions

resnet = ResNet50()

def contains_banana(img):
	img = image.load_img(img, target_size=(224, 224))
	img = image.img_to_array(img)
	img = np.expand_dims(img, axis=0)
	img = preprocess_input(img)
	predictions = decode_predictions(resnet.predict(img), top=3)[0]
	probability = 0.0
	for _, class_name, class_probability in predictions:
		if class_name == 'banana':
			probability = class_probability
			break
	return probability

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

print(contains_banana(Path('/dev/notebooks/lambda-general/ML-YouOnlyLookOnce/sample_data/positive_examples/example0.jpeg')))