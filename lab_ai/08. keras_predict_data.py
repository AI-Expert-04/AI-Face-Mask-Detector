# 08. keras_predict_data.py
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

model = tf.keras.models.load_model('../models/mymodel')

print(model.summary())

plt.figure(0)
plt.title('Valid Dataset Predict')
for images, labels in valid_dataset.take(1):
    rc_images = resize_and_crop(images)
    predict = model.predict(rc_images)
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(images[i].numpy().astype('uint8'))
        if predict[i][0] > 0.5:
            label_index = 1
        else:
            label_index = 0

        if labels[i] == label_index:
            result = 'OK'
        else:
            result = 'Wrong'

        plt.title(valid_dataset.class_names[label_index] + '\n' + result)
        plt.axis('off')

plt.show()
