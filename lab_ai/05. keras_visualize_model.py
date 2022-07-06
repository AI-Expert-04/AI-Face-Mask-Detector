# 05. keras_visualize_model.py
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
model = tf.keras.applications.MobileNet(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

model.trainable = False

model = tf.keras.Sequential([
    model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1)
])

print(model.summary())

# 모델 학습
if not os.path.exists('../logs'):
    os.mkdir('../logs')

tensorboard = tf.keras.callbacks.TensorBoard(log_dir='../logs')

learning_rate = 0.0001

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.RMSprop(lr=learning_rate),
    metrics=['accuracy']
)

epochs = 2
history = model.fit(
    rc_train_dataset,
    epochs=epochs,
    validation_data=rc_valid_dataset,
    callbacks=[tensorboard]
)
print(history)

# Anaconda Prompt 에서 face-mask-detector-env 환경 활성화 시킨 후
# Face Mask Detector 프로젝트 최상위 폴더로 이동한 상태에서
# tensorboard --logdir=./logs
