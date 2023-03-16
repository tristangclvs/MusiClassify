# All imports
import os

import math

import json
import IPython.display as ipd
import librosa
import librosa.display

from utils import file_mfcc, most_common, predict

# import numpy as np  # linear algebra
# import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
# import tensorflow as tf
# import tensorflow_io as tfio
# from matplotlib import pyplot as plt


# =============================================================================

# CONSTANTS
file_path = "data"
json_path = "data.json"
sr = 22500  # sample rate
duration = 30  # audio.duration.numpy() # duration in seconds
samples_per_track = sr * duration

data_empty = {
    "mapping": [],
    "mfcc": [],
    "labels": []
}


# =============================================================================


# Change the n_mfcc between 13 and 40
def save_mfcc(file_path, save_path, n_mfcc=25, n_fft=2048, hop_length=512, num_segments=5):
    # dictionary to store data
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": []
    }

    # num_samples_per_segment = samples_per_track // num_segments
    num_samples_per_segment = int(samples_per_track / num_segments)
    print("Number of samples per segment : ", num_samples_per_segment)
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)

    # loop through all the genres
    for i, (dirpath, dirnames, filenames) in enumerate(
            os.walk(file_path)):  # os.walk will go through all the folders in the file_path
        # ensure we're not at the root level
        if dirpath is not file_path:
            # save the semantic label
            # first loop that will store all the different genre names
            dirpath_components = dirpath.split("\\")  # data/blues => ["data", "blues"]
            print("dirpath_components 1 : ", dirpath_components)
            # dirpath_components = dirpath.split("/")[-1]  # data/blues => ["data", "blues"]
            print("dirpath_components 2 : ", dirpath_components)
            semantic_label = dirpath_components[-1]
            data["mapping"].append(semantic_label)
            print(f"\nProcessing {semantic_label}")

            # process file for a specific genre
            for f in filenames:
                file_mfcc(f, num_samples_per_segment, expected_num_mfcc_vectors_per_segment,
                          dirpath=dirpath,
                          data=data,
                          hop_length=hop_length, num_segments=num_segments, n_mfcc=n_mfcc, n_fft=n_fft,
                          iterator=i)

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
    print("Saved json file")


if __name__ == "__main__":
    # print("Uncomment function to run it")
    save_mfcc(file_path, json_path, num_segments=6)
