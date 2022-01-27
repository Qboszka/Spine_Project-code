from VGG16 import base_model
import keras.layers as layers
import tensorflow as tf
from keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop
from data_preprocessing_automatic import train_generator, validation_generator, test_generator
from data_preprocessing import x_train, y_train, x_val, y_val
import matplotlib.pyplot as plt   

base_model = base_model

#custom classifier 
x = layers.Flatten()(base_model.output)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(1, activation='sigmoid')(x)

model = Model(base_model.input, x)

opt = Adam(learning_rate = 0.000001)
model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])

#checkpoints
checkpoint_filepath = "C:\Workspace_studies\Project_main\Trainings\VGG16\model.{epoch:02d}-{val_accuracy:.2f}.hdf5"
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath = checkpoint_filepath,
    save_weights_only = False,
    monitor = 'val_accuracy',
    mode = 'max',
    save_best_only = True)

# 629 train images
# 79 val images
# 80 test images

#trainig params
batch_size = 32
steps_per_epoch = len(x_train) / batch_size
validation_steps = len(x_val) / batch_size

vgghist = model.fit(train_generator,
                    epochs = 500,
                    batch_size = batch_size,
                    steps_per_epoch = steps_per_epoch, 
                    validation_data = (x_val, y_val),
                    validation_steps = validation_steps,  
                    callbacks = [model_checkpoint_callback],
                    shuffle = True)

model.save("VGG16_Adam_Unified.h5")

acc = vgghist.history['accuracy']
val_acc = vgghist.history['val_accuracy']
loss = vgghist.history['loss']
val_loss = vgghist.history['val_loss']

epochs_range = range(500)

plt.figure(figsize = (15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label = 'Training Accuracy')
plt.plot(epochs_range, val_acc, label = 'Validation Accuracy')
plt.legend(loc = 'lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label = 'Training Loss')
plt.plot(epochs_range, val_loss, label = 'Validation Loss')
plt.legend(loc = 'upper right')
plt.title('Training and Validation Loss')
plt.show()