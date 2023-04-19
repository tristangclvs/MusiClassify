from cnn_V1 import load_data, DATA_PATH
import os
import numpy as np
from sklearn.model_selection import train_test_split


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

# def split_data_file(inputs, targets, test_size=0.1, validation_size=0.2):
#     # test_size = 0.1
#     # validation_size = 0.2
#     inputs_train, inputs_validation, inputs_test, targets_train, targets_validation, targets_test = prepare_datasets(
#         inputs, targets,
#         test_size, validation_size)
#     inputs_targets_array = [inputs_train, inputs_validation, inputs_test, targets_train, targets_validation,
#                             targets_test]
#     names_array = ["inputs_train", "inputs_validation", "inputs_test", "targets_train", "targets_validation",
#                    "targets_test"]
#     for array, name in zip(inputs_targets_array, names_array):
#         # save the arrays to files
#         np.save(f'split_data/{name}.npy', array)
#
#     return inputs_train, inputs_validation, inputs_test, targets_train, targets_validation, targets_test
