import numpy as np

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions

from skimage.transform import resize
from skimage.io import imread, imshow

model = ResNet50()

def contains_banana(img):
    rescaled = resize(img, (224, 224), mode='constant')
    
    processed = preprocess_input(rescaled, mode='tf')
    batch = np.expand_dims(processed, 0)
    
    predictions = decode_predictions(model.predict(batch))

    top_pred, top_score = predictions[0][0][1], predictions[0][0][2]
    
    if top_pred == 'banana':
        return top_score
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
