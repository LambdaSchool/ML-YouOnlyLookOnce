from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
#from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

model = ResNet50(weights='imagenet')

img_path = 'sample_data/positive_examples/example0.jpeg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)


def contains_banana(img):
	preds = model.predict(x)

	return ('Predicted:', decode_predictions(preds, top=3)[0])

'''('Predicted:',
 [('n03532672', 'hook', 0.12160819),
  ('n07753592', 'banana', 0.09228605),
  ('n03598930', 'jigsaw_puzzle', 0.05240591)])'''


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
