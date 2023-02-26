import os
from tensorflow import keras
import shap

reconstructed_model = keras.models.load_model(os.path.join(os.getcwd(), "cnn_model", "cnn_v3"))
explainer = shap.Explainer(reconstructed_model)
shap_values = explainer()
shap.summary_plot(shap_values, inputs)
