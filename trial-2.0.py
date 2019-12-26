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

def runModel(img_width,img_height,train_data_dir,valid_data_dir,n_train_samples,n_valid_samples,epochs,batch_size,outputSize):
	
	if K.image_data_format() == 'channels_first':
		input_shape = (3,img_width,img_height)
	else:
		input_shape = (img_width,img_height,3)

	train_data_gen = ImageDataGenerator( rescale = 1. / 255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
	test_data_gen = ImageDataGenerator( rescale = 1. / 255)

	train_generator = train_data_gen.flow_from_directory( 
		train_data_dir, 
		target_size = (img_width,img_height), 
		batch_size = batch_size, 
		class_mode = 'categorical')

	validation_generator = test_data_gen.flow_from_directory( 
		valid_data_dir, 
		target_size = (img_width,img_height), 
		batch_size = batch_size, 
		class_mode = 'categorical')


	model = Sequential()
	model.add(Conv2D(32, (3,3), input_shape = input_shape))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size = (2,2)))

	model.summary()

	model.add(Conv2D(32, (3,3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size = (2,2)))

	model.add(Conv2D(64, (3,3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size = (2,2)))

	model.add(Flatten())
	model.add(Dense(64))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(outputSize))
	model.add(Activation('softmax'))

	model.summary()

	model.compile(loss = 'categorical_crossentropy',
			  	optimizer = 'rmsprop',
			  	metrics = ['accuracy'])

	model.fit_generator(
			train_generator,
			steps_per_epoch = n_train_samples // batch_size,
			epochs = epochs,
			validation_data = validation_generator)

	model.save_weights('Saved-Weights.h5')

	img_pred = image.load_img('data/train/vegetables/potato/potato_4.jpeg', target_size = (150,150))
	img_pred = image.img_to_array(img_pred)
	img_pred = np.expand_dims(img_pred, axis = 0)

	result = model.predict(img_pred)

	return result
	

labelFruit = {0: 'Apple', 1: 'Banana',2: 'Orange'}
labelVeg = {0: 'Onion', 1: 'Potato',2: 'Tomato'}

img_width, img_height = 150, 150

train_data_dir = 'data/train'
valid_data_dir = 'data/validation'

n_train_samples = 20
n_valid_samples = 20
epochs = 10
batch_size = 2
outputSize = 2

isFruit = runModel(img_width,img_height,train_data_dir,valid_data_dir,n_train_samples,n_valid_samples,epochs,batch_size,outputSize)
print(isFruit)

max_val = np.amax(isFruit[0])
class_index = np.where(isFruit[0] == max_val)
#print(max_val,class_index[0][0])

if(class_index[0][0] == 0):
	train_data_dir = 'data/train/fruits'
	valid_data_dir = 'data/validation/fruits'
	outputSize = 3
	fruitResult = runModel(img_width,img_height,train_data_dir,valid_data_dir,n_train_samples,n_valid_samples,epochs,batch_size,outputSize)
	max_val = np.amax(fruitResult[0])
	class_index = np.where(fruitResult[0] == max_val)
	print(fruitResult)
	print(max_val, class_index[0])
	prediction = labelFruit[class_index[0][0]]
	print('Fruit:',prediction)
else:
	train_data_dir = 'data/train/vegetables'
	valid_data_dir = 'data/validation/vegetables'
	outputSize = 3
	vegResult = runModel(img_width,img_height,train_data_dir,valid_data_dir,n_train_samples,n_valid_samples,epochs,batch_size,outputSize)
	max_val = np.amax(vegResult[0])
	class_index = np.where(vegResult[0] == max_val)
	print(vegResult)
	print(max_val, class_index[0])
	prediction = labelVeg[class_index[0][0]]
	print('Vegetable:',prediction)


#max_val = np.amax(result[0])
#class_index = np.where(result[0] == max_val)
#print(result, max_val, class_index[0])
#print(result)
#prediction = labels[class_index[0][0]]

#if result[0][0] == 1:
	#prediction = 'banana'
#else:
	#prediction = 'apple'

	#print(prediction)