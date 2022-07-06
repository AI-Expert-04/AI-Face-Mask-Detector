from PIL import Image, ImageDraw
# 04. keras_train_model.py
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

resize_and_crop = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomCrop(height=224, width=224),
    tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
])

model = tf.keras.models.load_model('models/mymodel')

face_image_path = 'data/without_mask/3.jpg'
face_image = Image.open(face_image_path)
face_image = face_image.resize((224, 224))

face_image_np = np.array(face_image)
rc_face_image_np = resize_and_crop(np.array([face_image_np]))

predict = model.predict(rc_face_image_np)
print(predict)

plt.figure(1)
plt.title('valid_dataset_predict')
plt.imshow(face_image_np.astype('uint8'))
if predict[0][0] > 0.5:
    label = 'with_mask'
else:
    label = 'without_mask'

plt.title(label)
plt.axis('off')

plt.show()

