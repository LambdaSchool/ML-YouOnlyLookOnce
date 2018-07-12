import numpy as np

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions

def process_img_path(img_path):
  return image.load_img(img_path, target_size=(224, 224))

def img_contains_banana(img):
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  model = ResNet50(weights='imagenet')
  features = model.predict(x)
  print(features)
  results = decode_predictions(features, top=3)[0]
  print(results)
  for entry in results:
    if entry[1] == 'banana':
      return entry[2]
  return 0.0

if __name__ == '__main__':
  test_path = './example2.jpeg'
  test_img = process_img_path(test_path)
  test_result = img_contains_banana(test_img)

print('Banana confidence = {}'.format(test_result))

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
