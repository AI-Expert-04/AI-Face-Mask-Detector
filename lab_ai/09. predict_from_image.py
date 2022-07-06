# 09. predict_from_image.py
import face_recognition
from PIL import Image, ImageDraw
import tensorflow as tf
import numpy as np

resize_and_crop = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomCrop(height=224, width=224),
    tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
])

model = tf.keras.models.load_model('../models/mymodel')

face_image_path = '../data/without_mask/0.jpg'

face_image_np = face_recognition.load_image_file(face_image_path)
face_locations = face_recognition.face_locations(face_image_np)

face_image = Image.fromarray(face_image_np)
draw = ImageDraw.Draw(face_image)

for face_location in face_locations:
    top = face_location[0]
    right = face_location[1]
    bottom = face_location[2]
    left = face_location[3]
    face_crop = face_image.crop((left - 10, top - 10, right + 10, bottom + 10))
    face_crop = face_crop.resize((224, 224))
    face_crop_np = np.array(face_crop)
    rc_face_crop = resize_and_crop(np.array([face_crop_np]))
    predict = model.predict(rc_face_crop)
    if predict[0][0] > 0.5:
        label = 'without_mask'
    else:
        label = 'with_mask'
        
    draw.rectangle(((left, top), (right, bottom)), outline=(255, 255, 0), width=4)
    draw.text((left, top - 10), label)

face_image.show()
