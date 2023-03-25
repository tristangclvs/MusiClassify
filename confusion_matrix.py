from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# Plot the confusion matrix for a model and a dataset
def plot_conf_mat(model, inputs_test, targets_test, colormap=plt.cm.Greens):
    y_probs = model.predict(inputs_test)
    y_preds = np.argmax(y_probs, axis=1)

    with sns.axes_style("white"):  # Temporary hide Seaborn grid lines
        conf_matrix = confusion_matrix(targets_test, y_preds)
        print(sum(conf_matrix.diagonal()))
        print(conf_matrix.sum(axis=1))
        print(conf_matrix.sum(axis=0))
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100  # Compute percentages
        conf_matrix = np.nan_to_num(conf_matrix, nan=0)  # Replace NaN with zero

        display = ConfusionMatrixDisplay(conf_matrix.astype(int),
                                         display_labels=["blues", "classical", "country", "disco", "hiphop", "jazz",
                                                         "metal", "pop", "reggae", "rock"])

        display.plot(cmap=colormap, values_format="d")
        plt.title(f"Confusion Matrix of model v{model_version}")
        plt.xticks(rotation=45)
        plt.show()


# model_version = ""
# while model_version not in ["9", "10", "11"]:
#     model_version = input("Select model version (9, 10, 11):  ")

cmaps = [plt.cm.Greens, plt.cm.Blues, plt.cm.Reds]
i = 0
for model_version in ["9", "10", "11"]:
    model = tf.keras.models.load_model(f"cnn_model/cnn_v{model_version}")

    inputs_test = np.load("split_data/inputs_test.npy")
    targets_test = np.load("split_data/targets_test.npy")

    plot_conf_mat(model, inputs_test, targets_test, colormap=cmaps[i])
    i += 1
