from cnn_V1 import prepare_datasets, load_data, DATA_PATH
import numpy as np
from sklearn.model_selection import train_test_split

def prepare_datasets(test_size, validation_size):
    # load data
    inputs, targets = load_data(DATA_PATH)
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


def split_data_file(inputs, targets, test_size=0.1, validation_size=0.2):
    # test_size = 0.1
    # validation_size = 0.2
    inputs_train, inputs_validation, inputs_test, targets_train, targets_validation, targets_test = prepare_datasets(inputs, targets,
                                                                                                                     test_size, validation_size)
    inputs_targets_array = [inputs_train, inputs_validation, inputs_test, targets_train, targets_validation,
                            targets_test]
    names_array = ["inputs_train", "inputs_validation", "inputs_test", "targets_train", "targets_validation",
                   "targets_test"]
    for array, name in zip(inputs_targets_array, names_array):
        # save the arrays to files
        np.save(f'split_data/{name}.npy', array)
        
    return inputs_train, inputs_validation, inputs_test, targets_train, targets_validation, targets_test
