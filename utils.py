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

from music_cutter import trim_audio

# ======================================
# CONSTANTS
data_empty = {
    "mapping": [],
    "mfcc": [],
    "labels": []
}
sr = 22500
# ======================================

# def most_common(lst):
#     return max(set(lst), key=lst.count)

most_common = lambda lst: max(set(lst), key=lst.count)


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
    print("Expected : {}, Predicted genre : {} || index : {}".format(y, music_genre, predicted_index))
    return predicted_index, prediction


def file_mfcc(file, num_samples_per_segment, expected_num_mfcc_vectors_per_segment,
              dirpath=None,
              data=data_empty,
              hop_length=512, num_segments=5, n_mfcc=25, n_fft=2048,
              iterator=1,
              file_duration=30):
    """ Extracts mfcc from audio file and saves it into a json file along with genre labels. """

    # load audio file
    if dirpath is None:
        file_path = file
    else:
        file_path = os.path.join(dirpath, file)

    signal, sample_rate = librosa.load(file_path, sr=sr)

    if file_duration > 90:
        nb_segments = file_duration // 30
        for i in range(int(nb_segments)):
            # Set the start and end times for trimming (in seconds)
            start_time = 30 * i
            end_time = 30 * i + 30
            trimmed_audio = trim_audio(file_path, start_time, end_time)
            signal = trimmed_audio
            # signal = signal[len(signal) // 2: len(signal) // 2 + 30 * sample_rate]  # :(60 + 30 * sample_rate)
            print("Processing trimmed signal n°{}...".format(i + 1))
            # process all segments of audio file
            for segment in range(num_segments):
                start_sample = num_samples_per_segment * segment
                finish_sample = start_sample + num_samples_per_segment

                # store the mfcc for segment if it has the expected length
                mfcc = librosa.feature.mfcc(y=signal[start_sample:finish_sample],
                                            sr=sample_rate,
                                            n_fft=n_fft,
                                            n_mfcc=n_mfcc,
                                            hop_length=hop_length)

                mfcc = mfcc.T  # transpose the matrix

                if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                    data["mfcc"].append(mfcc.tolist())
                    data["labels"].append(iterator - 1)
                    print("{}, segment:{}".format(file_path.split('\\')[-1], segment))
