# ======== Imports ========
import audioread
import os
import math
import json

import sys

# import from parent directory
from mfcc_extraction import file_mfcc
from audio_converters import audio_to_wav
from create_model import predict, load_data
from utils import most_common

import numpy as np
from tensorflow import keras

# ===== End imports =======

# getting the name of the directory
# where the file is present.
current = os.path.dirname(os.path.realpath(__file__))

# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)

# adding the parent directory to
# the sys.path.
sys.path.append(parent)

# ======================================

inputs_folder = "inputs"
outputs_folder = "outputs"
output_file = "test.wav"
expected_genre = "Not given"

# Change the working directory to work in the project folder (needed for Pycharm)
print(os.getcwd())
os.chdir("../")
print(os.getcwd())
inputs_folder_path = os.path.join(os.getcwd(), inputs_folder)
outputs_folder_path = os.path.join(os.getcwd(), outputs_folder)

version_number = ""
while version_number not in ["6", "7", "8", "9", "10", "11", "12", "13", "14", "15"]:
    version_number = input("Choose a version number (6 - 15): ")
CNN_MODEL = "cnn_v" + version_number

print("Chosen CNN model: {0}".format(CNN_MODEL))

# Convert all files in inputs folder to wav ======================================
if input('Convert all files to wav ? (y/n) ') == 'y':
    print("\nConverting files... (this may take a while)\n")
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(inputs_folder_path)):
        for file in filenames:
            if "{0}.wav".format(file.split('.')[0]) not in os.listdir(outputs_folder_path):
                abs_file_path = os.path.join(inputs_folder_path, file)
                audio_to_wav(abs_file_path)
    print("\nAll files converted !\n")

# ================================================================================
print("Disclaimer: All files come from the Free Music Archive (https://freemusicarchive.org/).")
print("Chose a file from the provided list (genres are provided by the artists): ")
print("1. Sean Ferree - Shine - Rock")
print("2. Lobo Loco - Hey I felt in Love - Blues/Country ")
print("3. HoliznaPATREON - Getting Clean - Jazz/Pop ")

chosen_file_nb = input(">> ")
while chosen_file_nb not in ["1", "2", "3"]:
    chosen_file_nb = input(">> ")

if chosen_file_nb == "1":
    chosen_file = "Sean Ferree - Shine.mp3"
    expected_genre = "Rock"
elif chosen_file_nb == "2":
    chosen_file = "Lobo Loco - Hey I felt in Love (ID 1945).mp3"
    expected_genre = "Blues/Country"
elif chosen_file_nb == "3":
    chosen_file = "HoliznaPATREON - Getting Clean.mp3"
    expected_genre = "Jazz/Pop"

file_path_music = os.path.join("test_real_music", chosen_file)  # file to predict
json_path = "data.json"
json2 = "../../data.json"
sr = 22500  # sample rate
audio_file = audioread.audio_open(file_path_music)
print("Duration: ", audio_file.duration)
duration = 30  # duration in seconds
samples_per_track = sr * duration
num_segments = 6  # 7 with cnn_v3, 6 with cnn_v6
hop_length = 512

num_samples_per_segment = int(samples_per_track / num_segments)
expected_num_mfcc_vectors_per_segment = math.ceil(
    num_samples_per_segment / hop_length)

reconstructed_model = keras.models.load_model(os.path.join(os.getcwd(), "cnn_model", CNN_MODEL))

data = {
    "mapping": [
        "blues",
        "classical",
        "country",
        "disco",
        "hiphop",
        "jazz",
        "metal",
        "pop",
        "reggae",
        "rock"
    ],
    "mfcc": [],
    "labels": []
}

# === Trim and make mfcc ===

file_mfcc(file_path_music, num_samples_per_segment=num_samples_per_segment,
          expected_num_mfcc_vectors_per_segment=expected_num_mfcc_vectors_per_segment,
          dirpath=None,
          data=data,
          hop_length=512, num_segments=num_segments, n_mfcc=25, n_fft=2048,
          iterator=1,
          file_duration=audio_file.duration)

with open(json_path, "w") as fp:
    json.dump(data, fp, indent=4)

print("Saved json file")
print('MFCC saved')

inputs, targets = load_data(data)

# === Predict ===
stock = []
predictions = []
final_predictions = []
final_predictions2 = []
for i in range(len(inputs)):
    X = inputs[i]
    predicted_index, prediction = predict(reconstructed_model, X, expected_genre, data)
    predictions.append(100 * np.array(prediction[0]))
    stock.append(predicted_index[0])

final_predictions.append(np.mean(predictions, axis=0))
print(" ============ ", end="\n\n")

print("Predicted music genre is `{}`.".format(data["mapping"][most_common(stock)]), end="\n\n")

print(" ============ ", end="\n\n")
