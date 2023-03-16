# All imports
import json
import tensorflow as tf
import tensorflow.keras as keras
from utils import predict
from cnn_V1 import prepare_datasets

# Create confusion matrix
# confusion_matrix = tf.math.confusion_matrix(labels, predictions)

# Print confusion matrix
# print(confusion_matrix)

# === ADAPTER AVEC LA SUITE === #
# ===
DATA_PATH = "data.json"
with open(DATA_PATH, "r") as fp:
    data_global = json.load(fp)
print("\nData successfully loaded!\n")

inputs_train, inputs_validation, inputs_test, targets_train, targets_validation, targets_test = prepare_datasets(
    0.1, 0.1)

reconstructed_model = keras.models.load_model('./cnn_model/cnn_v6')

# Generate example data
# labels = data_global['labels']
# print('labels shape: ', len(labels))

# print([mfcc for mfcc in data_global['mfcc']])
# predictions = tf.constant([predict(reconstructed_model, mfcc, label, data_global)[1] for mfcc, label in  zip(data_global['mfcc'],labels)])
# predictions = tf.constant([predict(reconstructed_model, mfcc, label, data_global)[1] for mfcc, label in zip(inputs_train, targets_train)])
predictions = tf.constant(
    [predict(reconstructed_model, mfcc, label, data_global)[0] for mfcc, label in zip(inputs_test, targets_test)])
labels = tf.constant([label for label in targets_test])
print('labels shape: ', labels.shape)
# print (predictions)
print('prediction shape: ', predictions.shape)
# print('labels: ', labels)
# print('predictions: ', predictions)
# print('predictions: ', predictions.numpy())
# Create confusion matrix
confusion_matrix = tf.math.confusion_matrix(labels[0], predictions[0], num_classes=10)

# Print confusion matrix
print(confusion_matrix)

# ===================================================================================================
# y_array = [y for i in range(10)]
# indexes_array = []

# predict(reconstructed_model, X, y, data_global)
# for i in inputs_validation:
#     # print(np.where(inputs_validation == i)[0][0])
#     # predict(reconstructed_model, i, targets_validation[np.where(inputs_validation == i)[0][0]], data_global)
#     prediction = reconstructed_model.predict(i[np.newaxis, ...])
#     indexes_array.append(np.argmax(prediction, axis=1)[0])
# indexes_array = []
# for i in inputs_validation:
#     prediction = reconstructed_model.predict(i[np.newaxis, ...])
#     # Find all indices of the highest probability values
#     max_indices = np.where(prediction == np.max(prediction))[1]
#     # Choose one index randomly
#     index = np.random.choice(max_indices)
#     indexes_array.append(index)
# # Convert inputs_validation and indexes_array to TensorFlow tensors
# inputs_validation_tensor = tf.constant(np.argmax(inputs_validation, axis=1))
# indexes_array_tensor = tf.constant(indexes_array)
# # Compute confusion matrix
# confusion_matrix = tf.math.confusion_matrix(
#     inputs_validation_tensor, indexes_array_tensor, num_classes=10)
# print(confusion_matrix)
# # tf.math.confusion_matrix(np.argmax(inputs_validation, axis=1), np.argmax(targets_validation), num_classes=10)
