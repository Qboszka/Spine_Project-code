import tensorflow as tf
import keras
from keras.layers import Conv2D, Dropout, Flatten, MaxPool2D, Dense, BatchNormalization
from keras.models import Sequential


model = Sequential()
model.add(Conv2D(32,3,padding="same", activation="relu", input_shape=(75, 122, 3)))
model.add(MaxPool2D())

model.add(Conv2D(32, 3, padding="same", activation="relu"))
model.add(MaxPool2D())

model.add(Conv2D(64, 3, padding="same", activation="relu"))
model.add(MaxPool2D())
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(2, activation="softmax"))

model.summary()

