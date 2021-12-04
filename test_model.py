import tensorflow as tf 
import numpy as np
from tensorflow import keras
from data_preprocessing import x_test, y_test, x_val, y_val
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, plot_confusion_matrix, roc_curve, roc_auc_score, auc

# load model
model_filepath = "C:\Workspace_studies\Project_main\Checkpoints\model.169-0.67.hdf5"
model = keras.models.load_model(model_filepath)

# evaluate
print("Evaluate")
result = model.evaluate(x_test, y_test , batch_size = 79)

# matrix
predict_x = model.predict(x_test)
predictions_x = np.argmax(predict_x, axis = 1)
print(classification_report(y_test, predictions_x, target_names = ['Famale (Class 0)','Male (Class 1)']))


