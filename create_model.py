# All imports
import os
import json
import math
import random

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import seaborn as sns
import numpy as np
import tensorflow as tf

import tensorflow.keras as keras
import IPython.display as ipd
import librosa
import librosa.display
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow_io as tfio
import matplotlib
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from utils import most_common, predict

from confusion_matrix import plot_conf_mat


# ======================================
def test_range(model, inputs_test, targets_test, nb_tests):
    correct_count = 0
    for i in range(nb_tests):
        # On va chercher les éléments à tester dans un jeu de données n'ayant pas été vu par le modèle lors de l'entraînement
        rand = random.randrange(len(inputs_test))
        X = inputs_test[rand]
        y = targets_test[rand]
        predicted_index, prediction = predict(model, X, y, data)
        if predicted_index == y:
            print("Prediction correcte")
            correct_count += 1
        else:
            print("Mauvaise prediction")
    print()
    print(f"Echantillon de {nb_tests} essais")
    print(f"Bonnes predictions : {(correct_count / nb_tests) * 100:.2f}%")


def plot_loss_acc(history):
    """Plot training and (optionally) validation loss and accuracy"""
    # Setup plots
    # % matplotlib inline
    plt.rcParams["figure.figsize"] = 10, 8
    # % config InlineBackend.figure_format = 'retina'
    sns.set()

    loss = history.history["loss"]
    epochs = range(1, len(loss) + 1)

    plt.figure(figsize=(10, 10))

    plt.subplot(2, 1, 1)
    plt.plot(epochs, loss, ".--", label="Training loss")
    final_loss = loss[-1]
    title = "Training loss: {:.4f}".format(final_loss)
    plt.ylabel("Loss")
    if "val_loss" in history.history:
        val_loss = history.history["val_loss"]
        plt.plot(epochs, val_loss, "o-", label="Validation loss")
        final_val_loss = val_loss[-1]
        title += ", Validation loss: {:.4f}".format(final_val_loss)
    plt.title(title)
    plt.legend()

    acc = history.history["accuracy"]

    plt.subplot(2, 1, 2)
    plt.plot(epochs, acc, ".--", label="Training acc")
    final_acc = acc[-1]
    title = "Training accuracy: {:.2f}%".format(final_acc * 100)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    if "val_accuracy" in history.history:
        val_acc = history.history["val_accuracy"]
        plt.plot(epochs, val_acc, "o-", label="Validation acc")
        final_val_acc = val_acc[-1]
        title += ", Validation accuracy: {:.2f}%".format(final_val_acc * 100)
    plt.title(title)
    plt.legend()
    plt.show()


# Load data from json file
def load_data(data):
    """
    Loads the data from the specified path and converts it into numpy arrays
    :param data: the data file
    :return: The inputs and targets are being returned.
    """
    # Convert lists to numpy arrays
    inputs = np.array(data["mfcc"])
    print('inputs: ', inputs.shape)
    targets = np.array(data["labels"])
    print('targets: ', targets.shape)

    return inputs, targets


def prepare_datasets(inputs, targets, test_size, validation_size):
    # Check if split_data folder exists and if the files are already created
    if os.path.exists("split_data"):
        print("Split data already exists")
        inputs_train = np.load("split_data/inputs_train.npy")
        inputs_validation = np.load("split_data/inputs_validation.npy")
        inputs_test = np.load("split_data/inputs_test.npy")
        targets_train = np.load("split_data/targets_train.npy")
        targets_validation = np.load("split_data/targets_validation.npy")
        targets_test = np.load("split_data/targets_test.npy")
    else:
        print("Split data doesn't exist, need to create it")
        # split in train and test
        inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs, targets,
                                                                                  test_size=test_size,
                                                                                  random_state=0)
        # split in validation and test
        inputs_train, inputs_validation, targets_train, targets_validation = train_test_split(inputs_train,
                                                                                              targets_train,
                                                                                              test_size=validation_size,
                                                                                              random_state=1)
        # convert inputs to 3D arrays
        inputs_train = inputs_train[..., np.newaxis]  # 4D array -> (num_samples, 130, 13, 1)
        inputs_test = inputs_test[..., np.newaxis]
        inputs_validation = inputs_validation[..., np.newaxis]

        # ==============================================================================================================
        inputs_targets_array = [inputs_train, inputs_validation, inputs_test, targets_train, targets_validation,
                                targets_test]
        names_array = ["inputs_train", "inputs_validation", "inputs_test", "targets_train", "targets_validation",
                       "targets_test"]
        for array, name in zip(inputs_targets_array, names_array):
            # save the arrays to files
            np.save(f'split_data/{name}.npy', array)
        # ==============================================================================================================

    return inputs_train, inputs_validation, inputs_test, targets_train, targets_validation, targets_test


def build_model(input_shape, number_of_genres):
    # Define the CNN model
    model = tf.keras.Sequential()

    # Add the first convolutional layer
    model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu',
                                     input_shape=input_shape))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))

    # Add the second convolutional layer
    model.add(tf.keras.layers.Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))

    # Add the third convolutional layer
    model.add(tf.keras.layers.Conv2D(512, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Conv2D(256, kernel_size=(2, 2), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    # Flatten the output from the convolutional layers
    model.add(tf.keras.layers.Flatten())

    # Add a fully connected layer for classification
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(256, activation='relu'))

    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))

    model.add(tf.keras.layers.Dense(number_of_genres, activation='softmax'))
    return model


DATA_PATH = "data.json"
with open(DATA_PATH, "r") as fp:
    data = json.load(fp)

random.seed()

# create train, validation and test sets
inputs, targets = load_data(data)

print("\nData successfully loaded!\n")

inputs_train, inputs_validation, inputs_test, targets_train, targets_validation, targets_test = prepare_datasets(
    inputs, targets, 0.1, 0.2)

# build the CNN net
input_shape = (inputs_train.shape[1], inputs_train.shape[2], inputs_train.shape[3])

# print(input_shape)
number_of_genres = 10
model = build_model(input_shape, number_of_genres)
model.summary()

# compile the network
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=keras.optimizers.Adam(learning_rate=0.0001),
              metrics=['accuracy'])

# create the EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',  # monitor the validation loss
    patience=10,  # stop after 3 epochs of no improvement
    restore_best_weights=True  # restore the weights from the best epoch
)

# train the CNN
print("Training the model... This may take a while...")
history = model.fit(inputs_train, targets_train,
                    validation_data=(inputs_validation, targets_validation),
                    batch_size=BATCH_SIZE,
                    epochs=epochs_nb,
                    callbacks=[early_stopping])

# plot accuracy and error over the epochs
plot_loss_acc(history)

# evaluate the CNN on a sample
test_range(model, inputs_test=inputs_test, targets_test=targets_test, nb_tests=20)

# plot the confusion matrix
plot_conf_mat(model, inputs_test, targets_test, colormap=plt.cm.Greens)
