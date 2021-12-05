
from model import model
import tensorflow as tf
import keras
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from data_preprocessing import x_train, y_train, x_val, y_val

#compile
opt = Adam(learning_rate = 0.01)
model.compile(optimizer = opt, loss = tf.keras.losses.BinaryCrossentropy(from_logits = False) , metrics = ['accuracy'])

#checkpoints
checkpoint_filepath = "C:\Workspace_studies\Project_main\Trainings\OwnModel_1\model.{epoch:02d}-{val_accuracy:.2f}.hdf5"
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath = checkpoint_filepath,
    save_weights_only = False,
    monitor = 'val_accuracy',
    mode = 'max',
    save_best_only = True)

reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor = 0.2,
                              patience = 5, min_lr = 0.000001)

# 629 train images
# 79 val images
# 80 test images

#trainig params
batch_size = 32
steps_per_epoch = len(x_train) / batch_size
validation_steps = len(x_val) / batch_size

history = model.fit(x_train, y_train, 
                    epochs = 5,
                    batch_size = batch_size,
                    steps_per_epoch = steps_per_epoch, 
                    validation_data = (x_val, y_val),
                    validation_steps = validation_steps,  
                    callbacks = [model_checkpoint_callback, reduce_lr],
                    shuffle = True)

model.save("Own_epochs_500_lr_0.000001_opt_Adam_size_224.h5")

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(5)

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