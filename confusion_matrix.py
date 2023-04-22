import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import seaborn as sns
import numpy as np
import tensorflow as tf


# Plot the confusion matrix for a model and a dataset
def plot_conf_mat(model, inputs_test, targets_test, colormap=plt.cm.Greens, model_version=None):
    """ Plot the confusion matrix for a model and a dataset
     :param model: The model to evaluate
     :param inputs_test: The inputs of the test dataset
     :param targets_test: The targets of the test dataset
     :param colormap: The colormap to use for the confusion matrix """
    plt.rcParams["figure.figsize"] = 10, 8
    sns.set()

    y_probs = model.predict(inputs_test)
    y_preds = np.argmax(y_probs, axis=1)

    with sns.axes_style("white"):  # Temporary hide Seaborn grid lines
        conf_matrix = confusion_matrix(targets_test, y_preds)
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100  # Compute percentages
        conf_matrix = np.nan_to_num(conf_matrix, nan=0)  # Replace NaN with zero

        display = ConfusionMatrixDisplay(conf_matrix.astype(int),
                                         display_labels=["blues", "classical", "country", "disco", "hiphop", "jazz",
                                                         "metal", "pop", "reggae", "rock"])

        display.plot(cmap=colormap, values_format="d")
        if model_version is None:
            plt.title("Confusion Matrix of model")
        else:
            plt.title(f"Confusion Matrix of model v{model_version}")
        plt.title("Confusion Matrix")
        plt.xticks(rotation=45)
        plt.show()


if __name__ == "__main__":
    model_version = "15"
    model = tf.keras.models.load_model(f"cnn_model/cnn_v{model_version}")
    inputs_test = np.load("split_data/inputs_test.npy")
    targets_test = np.load("split_data/targets_test.npy")
    plot_conf_mat(model, inputs_test, targets_test, colormap=plt.cm.Greens)
