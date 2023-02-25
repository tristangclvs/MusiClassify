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
import tensorflow_io as tfio
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


# ======================================


def predict(model, X, y, data):
    X = X[np.newaxis, ...]
    prediction = model.predict(X)
    # print(prediction)
    # for i in range(len(prediction)):
    #     print("Genres: {}, Percentage: {}%".format(data["mapping"], prediction[i] * 100))

    # avoir l'index avec le plus d'occurences dans la prediction
    predicted_index = np.argmax(prediction, axis=1)
    music_genre = data["mapping"][predicted_index[0]]
    # print("Expected : {}, Predicted genre : {} || index : {}".format(y, music_genre, predicted_index))
    return predicted_index, prediction
