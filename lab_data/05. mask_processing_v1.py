# 얼굴 영역 정보를 이용한 마스크 합성

import face_recognition
from PIL import Image, ImageDraw

face_image_path = '../data/without_mask/1.jpg'
mask_image_path = '../data/mask.png'

face_image_np = face_recognition.load_image_file(face_image_path)
face_locations = face_recognition.face_locations(face_image_np)

face_image = Image.fromarray(face_image_np)
draw = ImageDraw.Draw(face_image)

for face_location in face_locations:
    top = face_location[0]
    right = face_location[1]
    bottom = face_location[2]
    left = face_location[3]
    draw.rectangle(((left, top), (right, bottom)), outline=(255, 255, 0), width=2)

    mask_image = Image.open(mask_image_path)
    mask_width = right - left
    mask_height = bottom - top
    mask_image = mask_image.resize((mask_width, mask_height))

    face_image.paste(mask_image, (left, int(top * 1.3)), mask_image)

    # face_image.show()

face_image.show()
