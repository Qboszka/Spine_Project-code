import tensorflow as tf
import keras
from keras.layers import Conv2D, Dropout, Flatten, MaxPool2D, Dense, BatchNormalization
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

model = Sequential()
model.add(Conv2D(32,3,padding="same", activation="relu", input_shape=(224,224,3)))
model.add(MaxPool2D())

model.add(Conv2D(32, 3, padding="same", activation="relu"))
model.add(MaxPool2D())

model.add(Conv2D(64, 3, padding="same", activation="relu"))
model.add(MaxPool2D())
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dense(2, activation="softmax"))

model.summary()

model_own = Sequential()
model_own.add(Conv2D(128, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model_own.add(MaxPool2D((2, 2)))
model_own.add(Conv2D(256, (3, 3), activation='relu'))
model_own.add(BatchNormalization())
model_own.add(MaxPool2D((2, 2)))
model_own.add(Dropout(0.5))
model_own.add(Conv2D(256, (3, 3), activation='relu'))
model_own.add(BatchNormalization())
model_own.add(MaxPool2D((2, 2)))
model_own.add(Conv2D(256, (3, 3), activation='relu'))
model_own.add(Dropout(0.5))
model_own.add(Conv2D(256, (3, 3), activation='relu'))
model_own.add(BatchNormalization())
model_own.add(MaxPool2D((2, 2)))
model_own.add(Flatten())
model_own.add(BatchNormalization())
model_own.add(Dense(128, activation='relu'))
model_own.add(Dropout(0.5))
model_own.add(Dense(64, activation='relu'))
model_own.add(Dense(32, activation='relu'))
model_own.add(Dense(2, activation='softmax'))