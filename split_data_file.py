from cnn_V1 import prepare_datasets, load_data, DATA_PATH
import json

print("Loading data...")
inputs, targets = load_data(DATA_PATH)
print("\nData successfully loaded!\n")

inputs_train, inputs_validation, inputs_test, targets_train, targets_validation, targets_test = prepare_datasets(
    0.1, 0.2)

print("\nData successfully split!\n")
print("Train set: ", inputs_train.shape, targets_train.shape)
print("Validation set: ", inputs_validation.shape, targets_validation.shape)
print("Test set: ", inputs_test.shape, targets_test.shape)

data_empty = {
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

data_train = {
    "mfcc": [inputs_train.tolist()],
    "labels": [targets_train.tolist()]
}
data_validation = {
    "mfcc": [inputs_validation.tolist()],
    "labels": [targets_validation.tolist()]
}
data_test = {
    "mfcc": [inputs_test.tolist()],
    "labels": [targets_test.tolist()]
}

with open(f"data_train.json", "w") as fp:
    json.dump(data_train, fp, indent=4)
print("Saved train json file")

with open(f"data_validation.json", "w") as fp:
    json.dump(data_validation, fp, indent=4)
print("Saved validation json file")

with open(f"data_test.json", "w") as fp:
    json.dump(data_test, fp, indent=4)
print("Saved test json file")
