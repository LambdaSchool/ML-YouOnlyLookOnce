import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL.ImageOps import invert

from keras.backend import tf as ktf
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

!wget -nc https://raw.githubusercontent.com/LambdaSchool/ML-YouOnlyLookOnce/master/sample_data/positive_examples/example0.jpeg
!wget -nc https://raw.githubusercontent.com/LambdaSchool/ML-YouOnlyLookOnce/master/sample_data/positive_examples/example1.jpeg
!wget -nc https://raw.githubusercontent.com/LambdaSchool/ML-YouOnlyLookOnce/master/sample_data/positive_examples/example2.jpeg

    
def contains_banana(img):
    img = img.resize((224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    model = ResNet50(weights='imagenet')
    features = model.predict(img)
    results = decode_predictions(features, top=3)[0]
    found = 0
    for r in results:
        if r[1] == 'banana':
          found = r[2]
    return np.max(found, 0)


def crop_image(img, quadrant):
    # make sure quardant input is valid
    if quadrant not in ['TL','TR','BL','BR']:
      print('Please chose from (\'TL\',\'TR\',\'BL\',\'BR\').')
      return None
    else:
      img = img_to_array(img)
      d = img.shape
      # height and width of the window are 2/3 of the image height and width
      h = int(d[0]*2/3)
      w = int(d[1]*2/3)
      # quadrant dictionary
      q = {"TL": img[:h,:w,:],
           "TR": img[:h,-w:,:],
           "BL": img[-h:,:w,:],
           "BR": img[-h:,-w:,:]}
      return array_to_img(q[quadrant])


def find_banana(img):
    img = img_to_array(img)
    ih, iw = img.shape[:2]
    quad = ['TL','TR','BL','BR']
    cropped = [crop_image(img, q) for q in quad]
    result = (0,0)
    
    for i in range(4):
      prob = contains_banana(cropped[i])
      
      if prob > result[0]:
        result = (prob, quad[i])
        h, w = img_to_array(cropped[i]).shape[:2]
        # define top left corners for each of the windows
        corner = [(0,0), (iw-w,0), (0, ih-h), (iw-w,ih-h)]
        rect = patches.Rectangle(corner[i], w, h, linewidth=1, edgecolor='r', facecolor='none')
        
        fig, ax = plt.subplots()
        ax.imshow(array_to_img(img))
        ax.add_patch(rect)
        plt.show()
    
    return result if result != (0,0) else None

# --------------------------------------------------------------------------------
for i in range(3):
  x = contains_banana('./example{}.jpeg'.format(i))
  print(i,': Banana confidence = {}'.format(x))

# --------------------------------------------------------------------------------    
img = load_img('./example1.jpeg', target_size=(224, 224))
cropped = crop_image(img, 'TR')

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize = (6,6))
ax1.imshow(crop_image(img, 'TL'))
ax2.imshow(crop_image(img, 'TR'))
ax3.imshow(crop_image(img, 'BL'))
ax4.imshow(crop_image(img, 'BR'))

for ax in (ax1, ax2, ax3, ax4):
    ax.axis("off")
fig.tight_layout()

# --------------------------------------------------------------------------------
img = load_img('./example0.jpeg', target_size=(224, 224))
find = find_banana(img)
print(find)
