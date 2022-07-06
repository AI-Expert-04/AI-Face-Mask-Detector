# 03. keras_create_model.py
import tensorflow as tf

model = tf.keras.applications.VGG16()
print(model.summary())
