# All imports
import os

import math

import json

from utils import file_mfcc, most_common, predict

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
    # Pour lancer l'extraction des mfcc, si le fichier json n'existe pas déjà
    if not os.path.exists(json_path):
        print("Json file does not exist")
        print("Process of files and creation of Json will begin", end='\n\n')
        save_mfcc(file_path, json_path, n_mfcc=40, n_fft=2048, hop_length=512, num_segments=5)  # n_mfcc=25
    else:  # sinon on le charge
        print("Json file already exists")
        with open(json_path, "r") as fp:
            print("Loading file... Please wait...")
            data_global = json.load(fp)
            print("Json file loaded")
