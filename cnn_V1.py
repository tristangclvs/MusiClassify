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

from utils import predict

# ======================================


DATA_PATH = "data.json"
with open(DATA_PATH, "r") as fp:
    data_global = json.load(fp)


def plot_history(history):
    fig, axs = plt.subplots(2)

    # create accuracy subplot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create accuracy subplot
    axs[1].plot(history.history["loss"], label="train loss")
    axs[1].plot(history.history["val_loss"], label="test loss")
    axs[1].set_ylabel("Loss")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="lower right")
    axs[1].set_title("Loss eval")

    plt.show()


# Load data from json file
def load_data(data_path):
    """
    Loads the data from the specified path and converts it into numpy arrays

    :param data_path: the path to the data file
    :return: The inputs and targets are being returned.
    """
    with open(data_path, "r") as fp:
        data = json.load(fp)

    # convert lists to numpy arrays
    inputs = np.array(data["mfcc"])
    print('inputs: ', inputs.shape)
    targets = np.array(data["labels"])
    print('targets: ', targets.shape)

    return inputs, targets


def prepare_datasets(inputs, targets, test_size, validation_size):
    # split in train and test
    inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs, targets,
                                                                              test_size=test_size,
                                                                              random_state=0)

    # split in validation and test
    inputs_train, inputs_validation, targets_train, targets_validation = train_test_split(inputs_train, targets_train,
                                                                                          test_size=validation_size,
                                                                                          random_state=1)

    # convert inputs to 3D arrays
    inputs_train = inputs_train[..., np.newaxis]  # 4D array -> (num_samples, 130, 13, 1)
    inputs_test = inputs_test[..., np.newaxis]
    inputs_validation = inputs_validation[..., np.newaxis]

    return inputs_train, inputs_validation, inputs_test, targets_train, targets_validation, targets_test


def build_model(input_shape, number_of_genres):
    # Define the CNN model
    model = tf.keras.Sequential()

    # Add the first convolutional layer
    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu',
                                     input_shape=input_shape))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(tf.keras.layers.Dropout(0.25))

    # Add the second convolutional layer
    model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(tf.keras.layers.Dropout(0.25))

    # Add the third convolutional layer
    model.add(tf.keras.layers.Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(tf.keras.layers.Dropout(0.25))

    # Flatten the output from the convolutional layers
    model.add(tf.keras.layers.Flatten())

    # Add a fully connected layer for classification
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.25))

    # model.add(tf.keras.layers.Dense(128, activation='relu'))
    # model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.Dense(number_of_genres, activation='softmax'))
    return model


if __name__ == "__main__":
    # create train, validation and test sets
    inputs, targets = load_data(DATA_PATH)

    print("\nData successfully loaded!\n")

    inputs_train, inputs_validation, inputs_test, targets_train, targets_validation, targets_test = prepare_datasets(
        inputs, targets, 0.1, 0.2)
    inputs_targets_array = [inputs_train, inputs_validation, inputs_test, targets_train, targets_validation,
                            targets_test]
    names_array = ["inputs_train", "inputs_validation", "inputs_test", "targets_train", "targets_validation",
                   "targets_test"]
    for array, name in zip(inputs_targets_array, names_array):
        # save the arrays to files
        np.save(f'split_data/{name}.npy', array)

    # load the array from the file
    arr = np.load('split_data/inputs_train.npy')

    # # build the CNN net
    # input_shape = (inputs_train.shape[1], inputs_train.shape[2], inputs_train.shape[3])
    # print(input_shape)
    # number_of_genres = 10
    # model = build_model(input_shape, number_of_genres)
    # model.summary()
    #
    # # compile the network
    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    #
    # # model.compile(optimizer=optimizer,
    # #               loss='sparse_categorical_crossentropy',
    # #               metrics=['accuracy'])
    # model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #               optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    #               metrics=['accuracy'])
    #
    # # train the CNN
    # history = model.fit(inputs_train, targets_train,
    #                     validation_data=(inputs_validation, targets_validation),
    #                     batch_size=32,
    #                     epochs=50)  # batch_size=32
    #
    # # plot accuracy and error over the epochs
    # plot_history(history)
    #
    # # evaluate the CNN on the test set
    # test_loss, test_acc = model.evaluate(inputs_test, targets_test, verbose=2)
    # print('Test accuracy:', test_acc)
    # print('Test loss:', test_loss)
    #
    # # make predictions on a sample
    # X = inputs_test[99]
    # y = targets_test[99]
    # predict(model, X, y, data_global)
