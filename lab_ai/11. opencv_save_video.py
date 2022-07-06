# 11. opencv_save_video.py
import tensorflow as tf
import numpy as np
import cv2
import os

resize_and_crop = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomCrop(height=224, width=224),
    tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
])

face_mask_recognition_model = cv2.dnn.readNet(
    '../models/face_mask_recognition.prototxt',
    '../models/face_mask_recognition.caffemodel'
)

mask_detector_model = tf.keras.models.load_model('../models/mymodel')

cap = cv2.VideoCapture('../data/04.mp4')
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

if not os.path.exists('../outputs'):
    os.mkdir('../outputs')

out = None

while cap.isOpened():
    ret, image = cap.read()
    if not ret:
        break

    height, width = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, scalefactor=1., size=(300, 300), mean=(104., 177., 123.))
    face_mask_recognition_model.setInput(blob)
    face_locations = face_mask_recognition_model.forward()

    result_image = image.copy()

    for i in range(face_locations.shape[2]):
        confidence = face_locations[0, 0, i, 2]
        if confidence < 0.5:
            continue

        left = int(face_locations[0, 0, i, 3] * width)
        top = int(face_locations[0, 0, i, 4] * height)
        right = int(face_locations[0, 0, i, 5] * width)
        bottom = int(face_locations[0, 0, i, 6] * height)

        face_image = image[top:bottom, left:right]
        face_image = cv2.resize(face_image, dsize=(224, 224))
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

        rc_face_image = resize_and_crop(np.array([face_image]))

        predict = mask_detector_model.predict(rc_face_image)
        if predict[0][0] > 0.5:
            color = (0, 0, 255)
            label = 'without_mask'
        else:
            color = (0, 255, 0)
            label = 'with_mask'

        cv2.rectangle(
            result_image,
            pt1=(left, top),
            pt2=(right, bottom),
            thickness=2,
            color=color,
            lineType=cv2.LINE_AA
        )

        cv2.putText(
            result_image,
            text=label,
            org=(left, top - 10),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.8,
            color=color,
            thickness=2,
            lineType=cv2.LINE_AA
        )

    if out is None:
        out = cv2.VideoWriter(
            '../outputs/output.mp4',
            fourcc,
            cap.get(cv2.CAP_PROP_FPS),
            (image.shape[1], image.shape[0])
        )
    else:
        out.write(result_image)

    cv2.imshow('result', result_image)
    if cv2.waitKey(1) == ord('q'):
        break

out.release()
cap.release()
