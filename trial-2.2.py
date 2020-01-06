from keras.preprocessing.image import ImageDataGenerator 			 #Generate multiple images by performing operations like rotation,flip etc...
from tensorflow.keras.models import Sequential									 # Helps to model our nueral network 
from tensorflow.keras.layers import Conv2D, MaxPooling2D						 # Conv2D to extract features from image and MaxPooling2D to reduce the size of data
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense 		 # Activation to decide the activation function
															 		 # Dropout to drop some nuerons by learning the process to save time
															 	 	 # Flatten to convert image stored in 2D form to 1D format
															 		 # Dense to add hidden layers

from keras import backend as K  									 # To identify the channel of input image like rgb, height, width etc...
import numpy as np 
from keras.preprocessing import image 								 # To retrieve image from directory and preprocess it
import tensorflow as tf 
from tensorflow import keras


label = {0: 'Apple', 1: 'Banana',2: 'Onion', 3: 'Orange', 4: 'Potato',5: 'Tomato'}
fruits = [0,1,3]
vegetables = [2,4,5]

agromate = tf.keras.models.load_model('agromate.h5')
agromate.summary()
img_pred = image.load_img('data/train/banana/banana_1.jpg', target_size = (150,150))
img_pred = image.img_to_array(img_pred)
img_pred = np.expand_dims(img_pred, axis = 0)

result = agromate.predict(img_pred)
max_val = np.amax(result[0])
class_index = np.where(result[0] == max_val)

#print(max_val,class_index[0][0])
#print(result)

for i in fruits:
	if i == class_index[0][0]:
		isFruit = True
		break
	else:
		isFruit = False

if isFruit == True:
	img_class = 'Fruit:'
else: 
	img_class = 'Vegetable:'

prediction = label[class_index[0][0]]
print(img_class,prediction)