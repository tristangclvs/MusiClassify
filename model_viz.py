import tensorflow.keras as keras
import tensorflow as tf

model = keras.models.load_model('./cnn_model/cnn_v6')

tf.keras.utils.plot_model(model, show_shapes=True)
