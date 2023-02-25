# All imports
import os

import math

import tensorflow.keras as keras
import json
import IPython.display as ipd
import librosa
import librosa.display
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
import tensorflow_io as tfio
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from cnn_V1 import load_data, prepare_datasets
from utils import predict

# ===
DATA_PATH = "data.json"
with open(DATA_PATH, "r") as fp:
    data_global = json.load(fp)
print("\nData successfully loaded!\n")

inputs_train, inputs_validation, inputs_test, targets_train, targets_validation, targets_test = prepare_datasets(
    0.1, 0.2)

reconstructed_model = keras.models.load_model('./cnn_model/cnn_v3')

X = inputs_validation[99]
y = targets_validation[99]
y_array = [y for i in range(10)]
# indexes_array = []

# predict(reconstructed_model, X, y, data_global)
# for i in inputs_validation:
#     # print(np.where(inputs_validation == i)[0][0])
#     # predict(reconstructed_model, i, targets_validation[np.where(inputs_validation == i)[0][0]], data_global)
#     prediction = reconstructed_model.predict(i[np.newaxis, ...])
#     indexes_array.append(np.argmax(prediction, axis=1)[0])
indexes_array = []
for i in inputs_validation:
    prediction = reconstructed_model.predict(i[np.newaxis, ...])
    # Find all indices of the highest probability values
    max_indices = np.where(prediction == np.max(prediction))[1]
    # Choose one index randomly
    index = np.random.choice(max_indices)
    indexes_array.append(index)
# Convert inputs_validation and indexes_array to TensorFlow tensors
inputs_validation_tensor = tf.constant(np.argmax(inputs_validation, axis=1))
indexes_array_tensor = tf.constant(indexes_array)
# Compute confusion matrix
confusion_matrix = tf.math.confusion_matrix(
    inputs_validation_tensor, indexes_array_tensor, num_classes=10)
print(confusion_matrix)
# tf.math.confusion_matrix(np.argmax(inputs_validation, axis=1), np.argmax(targets_validation), num_classes=10)
