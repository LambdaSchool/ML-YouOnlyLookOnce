import numpy as np

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions


model = ResNet50(weights='imagenet')

img_path1 = '/positive_examples/example0.jpg'
img_path2 = '/positive_examples/example1.jpg'
img_path3 = '/positive_examples/example2.jpg'

img1 = image.load_img(img_path1, target_size=(224, 224))
img2 = image.load_img(img_path2, target_size=(224, 224))
img3 = image.load_img(img_path3, target_size=(224, 224))


x1 = image.img_to_array(img1)
x1 = np.expand_dims(x1, axis=0)
x1 = preprocess_input(x1)

preds = model.predict(x1)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=3)[0])
# Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]

def contains_banana(img):
	img_res = image.load_img(img, target_size=(224, 224))
	x = image.img_to_array(img_res)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)

	preds = model.predict(x)
	"""
	Change the contents of this function so it behaves correctly
	"""
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
