# ai.py
import tensorflow as tf
import numpy as np
import os

resize_and_crop = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomCrop(height=224, width=224),
    tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
])


# 학습 데이터 로드
def load_data():
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        'data/',
        validation_split=0.2,
        subset='training',
        seed=123,
        image_size=(224, 224),
        batch_size=16
    )

    valid_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        'data/',
        validation_split=0.2,
        subset='validation',
        seed=123,
        image_size=(224, 224),
        batch_size=16
    )

    rc_train_dataset = train_dataset.map(lambda x, y: (resize_and_crop(x), y))
    rc_valid_dataset = valid_dataset.map(lambda x, y: (resize_and_crop(x), y))

    return rc_train_dataset, rc_valid_dataset


# 모델 생성
def create_model():
    if os.path.exists('models/mymodel'):
        model = tf.keras.models.load_model('models/mymodel')

        model.layers[0].trainable = False
        model.layers[2].trainable = True
    else:
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

        learning_rate = 0.001
        model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.RMSprop(lr=learning_rate),
            metrics=['accuracy']
        )
        
        train_dataset, valid_dataset = load_data()
        train_model(model, 2, train_dataset, valid_dataset, True)
    return model


# 모델 학습
def train_model(model, epochs, train_dataset, valid_dataset, save_model):
    history = model.fit(train_dataset, epochs=epochs, validation_data=valid_dataset)
    if save_model:
        model.save('models/mymodel')
    return history


# 학습된 모델로 예측
def predict(model, image):
    rc_image = resize_and_crop(np.array([image]))
    result = model.predict(rc_image)
    if result[0] > 0.5:
        return 1
    else:
        return 0


if __name__ == '__main__':
    train_dataset, valid_dataset = load_data()
    model = create_model()
    train_model(model, 2, train_dataset, valid_dataset, True)
