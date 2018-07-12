import numpy as np

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions

model = ResNet50(weights='imagenet')

neg =  "C:/Users/Smartsrc-alex/Documents/ML_class/ML-YouOnlyLookOnce/sample_data/negative_examples/"
pos =  "C:/Users/Smartsrc-alex/Documents/ML_class/ML-YouOnlyLookOnce/sample_data/positive_examples/"

negative_ex = [neg + 'example10.jpeg',neg + 'example11.jpeg',neg + 'example12.jpeg']

positive_ex = [pos + 'example0.jpeg',pos + 'example1.jpeg',pos + 'example2.jpeg']

def contains_banana(img):

	img = image.load_img(img, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)

	preds = model.predict(x)

	results = decode_predictions(preds, top=3)[0]
	results = [ res[1] for res in results]
	# print(results)

	if( 'banana' in results ): return 1
	else: return 0

def crop_image(img_path, quadrant):

	img = image.load_img(img_path, target_size=(2000,1500))


def find_banana(img):
	"""
	Change the contents of this function so it behaves correctly
	"""
	return "None"



print(contains_banana(positive_ex[0]))