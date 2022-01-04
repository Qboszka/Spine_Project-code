from load_data import get_data
from keras.preprocessing.image import ImageDataGenerator
import os

train_data_dir = ('C:\\Workspace_studies\\Project_main\\Input_unified\\train')
validation_data_dir = ('C:\\Workspace_studies\\Project_main\\Input_unified\\val')
test_data_dir = ('C:\\Workspace_studies\\Project_main\\Input_unified\\test')

#params
img_width = 122
img_height = 75
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip = True)
 
test_datagen = ImageDataGenerator(rescale=1. / 255)
 
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')
 
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=80,
    class_mode='binary')

