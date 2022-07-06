# 07. keras_load_model.py
import tensorflow as tf
import matplotlib.pyplot as plt
import os

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    '../data/',
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=(224, 224),
    batch_size=16
)

valid_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    '../data/',
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=(224, 224),
    batch_size=16
)

resize_and_crop = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomCrop(height=224, width=224),
    tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
])

rc_train_dataset = train_dataset.map(lambda x, y: (resize_and_crop(x), y))
rc_valid_dataset = valid_dataset.map(lambda x, y: (resize_and_crop(x), y))

# 모델 생성
model = tf.keras.models.load_model('../models/mymodel')

model.layers[0].trainable = False
model.layers[2].trainable = True

print(model.summary())

epochs = 2
history = model.fit(
    rc_train_dataset,
    epochs=epochs,
    validation_data=rc_valid_dataset
)
print(history)
