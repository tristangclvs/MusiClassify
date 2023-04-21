import subprocess

## First of all, install all required packages
# Define the command to use
cmd = ['pip', 'install', '-r', 'requirements.txt']

# Use subprocess to execute the command
print("Installing required packages...")
process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
stdout, stderr = process.communicate()
print("Packages installed !")

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import seaborn as sns
import numpy as np
import tensorflow as tf
from confusion_matrix import plot_conf_mat

# =================================================================================================


allowed_answers = ["y", "n", "escape"]
# Main file to run the programs
print("===============================================================================")
print("                        Welcome to MusiClassify                                ")
print("You will go through the steps to run all commands and programs of this project.")
print("You can run the programs or exit at any time by typing 'Escape'.")
print("===============================================================================", end='\n\n')

command = input("Do you want to start ? (y/n) ").lower().split()[0]
while command not in allowed_answers:
    command = input("Please enter a valid answer (y/n) ").lower().split()[0]

if command == "y":
    print()
    command = input("Do you want to extract mfccs ? (y/n) ").lower().split()[0]
    while command not in allowed_answers:
        command = input("Please enter a valid answer (y/n) ").lower().split()[0]
    if command == "y":
        with open("mfcc_extraction.py") as f:
            extraction = compile(f.read(), "mfcc_extraction.py", "exec")
        exec(extraction, {"__name__": "__main__"})
        print()
        print("MFCCs extracted !")
        print("Let's create a model and train it !")
        print()
        BATCH_SIZE = 64
        epochs_nb = ""
        while not epochs_nb.isdigit():
            epochs_nb = input("How many epochs do you want to train the model (high numbers require good machine) ? ")
        epochs_nb = int(epochs_nb)
        variables = {"BATCH_SIZE": BATCH_SIZE, "epochs_nb": epochs_nb}

        with open("create_model.py") as f:
            model_creation_and_training = compile(f.read(), "create_model.py", "exec")
        exec(model_creation_and_training, variables)
    elif command == "escape":
        exit()


else:
    print("Exiting program...")
    exit()
