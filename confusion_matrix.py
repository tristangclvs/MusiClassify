from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# Plot the confusion matrix for a model and a dataset
def plot_conf_mat(model, inputs_test, targets_test):
    y_probs = model.predict(inputs_test)
    y_preds = np.argmax(y_probs, axis=1)

    with sns.axes_style("white"):  # Temporary hide Seaborn grid lines
        conf_matrix = confusion_matrix(targets_test, y_preds)
        display = ConfusionMatrixDisplay(conf_matrix,
                                         display_labels=["blues", "classical", "country", "disco", "hiphop", "jazz",
                                                         "metal", "pop", "reggae", "rock"])

        display.plot()
        plt.xticks(rotation=45)
        plt.show()


model = tf.keras.models.load_model("cnn_model/cnn_v10")

inputs_test = np.load("split_data/inputs_test.npy")
targets_test = np.load("split_data/targets_test.npy")

plot_conf_mat(model, inputs_test, targets_test)
