import tensorflow as tf 
from tensorflow import keras
from data_preprocessing import x_test, y_test
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, plot_confusion_matrix, roc_curve, roc_auc_score, auc

# prepare test_set
#test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
#test_dataset = test_dataset.batch(32)

# load model
model_filepath = "C:\Workspace_studies\Project_main\Checkpoints\model.169-0.67.hdf5"
model = keras.models.load_model(model_filepath)

# evaluate
print("Evaluate")
result = model.evaluate(x_test, y_test, batch_size = 80)
dict(zip(model.metrics_names, result))

# predict
#predictions = model.predict_classes(x_test)
#predictions = predictions.reshape(1,-1)[0]
#print(classification_report(y_test, predictions, target_names = ['Famale (Class 0)','Male (Class 1)']))