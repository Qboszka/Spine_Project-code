import tensorflow as tf
import keras
from keras import layers
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

model = Sequential()
model.add(layers.Conv2D (128, (3, 3), activation = 'relu', input_shape = (224, 224, 3)))
model.add(layers.MaxPool2D ((2, 2)))
model.add(layers.Conv2D (256, (3, 3), activation = 'relu'))
model.add(layers.BatchNormalization ())
model.add(layers.MaxPool2D ((2, 2)))
model.add(layers.Dropout (0.5))
model.add(layers.Conv2D (256, (3, 3), activation = 'relu'))
model.add(layers.BatchNormalization ())
model.add(layers.MaxPool2D ((2, 2)))
model.add(layers.Conv2D (256, (3, 3), activation = 'relu'))
model.add(layers.Dropout (0.5))
model.add(layers.Conv2D (256, (3, 3), activation = 'relu'))
model.add(layers.BatchNormalization ())
model.add(layers.MaxPool2D ((2, 2)))
model.add(layers.Flatten ())
model.add(layers.BatchNormalization ())
model.add(layers.Dense (128, activation = 'relu'))
model.add(layers.Dropout (0.5))
model.add(layers.Dense (64, activation = 'relu'))
model.add(layers.Dense (32, activation = 'relu'))
model.add(layers.Dense (1, activation = 'sigmoid'))

model.summary()