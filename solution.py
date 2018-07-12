import numpy as np

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions


def contains_banana(img):
    model = ResNet50(include_top=True, 
                     weights='imagenet', 
                     input_tensor=None, 
                     input_shape=None, 
                     pooling=None, 
                     classes=1000)
    img = image.load_img(img, target_size=(224, 224))
    img = image.img_to_array(img)
    x = preprocess_input(np.expand_dims(img.copy(), axis=0))
    preds = model.predict(x)
    probs = decode_predictions(preds, top=3)
    if 'banana' in probs:
        return probs
    else:     
        return 0

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
