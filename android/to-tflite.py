import tensorflow as tf 
import numpy as np 
from tensorflow import keras
#from tensorflow.contrib import lite

keras_file = "agromate.h5"

converter = tf.lite.TFLiteConverter.from_keras_model_file(keras_file)
tflite_model = converter.convert()
open("agro.tflite","wb").write(tflite_model) 