from load_data import get_data
from keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np

#params

#interpolation
img_size = 224

#unified
img_width = 122
img_height = 75

#Fetch data Windows
#train_data = get_data('C:\\Workspace_studies\\Project_main\\Input\\train')
#val_data = get_data('C:\\Workspace_studies\\Project_main\\Input\\val')
#test_data = get_data('C:\\Workspace_studies\\Project_main\\Input\\test')

#Fetch unified data Windows
train_data = get_data('C:\\Workspace_studies\\Project_main\\Input_unified\\train')
val_data = get_data('C:\\Workspace_studies\\Project_main\\Input_unified\\val')
test_data = get_data('C:\\Workspace_studies\\Project_main\\Input_unified\\test')

#Fetch data MacOS

#prepare train
x_train = []
y_train = []

#prepare val
x_val = []
y_val = []

#prepare test
x_test = []
y_test = []

for feature, label in train_data:
  x_train.append(feature)
  y_train.append(label)

for feature, label in val_data:
  x_val.append(feature)
  y_val.append(label)
  
for feature, label in test_data:
  x_test.append(feature)
  y_test.append(label)

# Normalize the data 
x_train = np.array(x_train) / 255
x_val = np.array(x_val) / 255
x_test = np.array(x_test) / 255

x_train.reshape(-1, img_width, img_height, 1)
y_train = np.array(y_train)

x_val.reshape(-1, img_width, img_height, 1)
y_val = np.array(y_val)

x_test.reshape(-1, img_width, img_height, 1)
y_test = np.array(y_test)

#DataGenerator with augmentation
datagen = ImageDataGenerator(
        featurewise_center = False,  # set input mean to 0 over the dataset
        samplewise_center = False,  # set each sample mean to 0
        featurewise_std_normalization = False,  # divide inputs by std of the dataset
        samplewise_std_normalization = False,  # divide each input by its std
        zca_whitening = False,  # apply ZCA whitening
        rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.2, # Randomly zoom image 
        width_shift_range = 0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range = 0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip = True,  # randomly flip images
        vertical_flip = True)  # randomly flip images

datagen.fit(x_train)



