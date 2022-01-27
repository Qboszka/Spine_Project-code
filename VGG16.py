from tensorflow.keras.applications.vgg16 import VGG16
import keras.layers as layers
import tensorflow as tf

base_model = VGG16(input_shape = (75, 122, 3), # Shape of our images
include_top = False, # Leave out the last fully connected layer
weights = 'imagenet')

for layer in base_model.layers:
    layer.trainable = False
