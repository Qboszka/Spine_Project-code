import tensorflow as tf 
import numpy as np
from tensorflow import keras
from data_preprocessing import x_test, y_test, x_val, y_val
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, plot_confusion_matrix, roc_curve, roc_auc_score, auc

# load model
model_filepath = "C:\Workspace_studies\Project_main\Own_epochs_500_lr_0.000001_opt_Adam_size_224.h5"   #"C:\Workspace_studies\Project_main\Trainings\OwnModel_1\model.03-0.58.hdf5"
model = keras.models.load_model(model_filepath)

# evaluate
print("Evaluation: ")
print("\n")

result = model.evaluate(x_test, y_test , batch_size = 79)
y_pred = (model.predict(x_test) > 0.5).astype("int32")  #model.predict(x_test)
#print(y_pred) # debug
#y_pred = np.argmax(y_pred , axis = 1)
#print(y_pred) # debug

# reports
print(classification_report(y_test, y_pred, target_names = ['Famale (Class 0)','Male (Class 1)']))

accuracy_score = accuracy_score(y_test, y_pred) * 100
print("Accuracy is {}%".format(accuracy_score))


