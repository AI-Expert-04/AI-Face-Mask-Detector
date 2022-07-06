# 데이터 파트
from urllib.request import Request, urlopen
import json
import os

import face_recognition
from PIL import Image, ImageDraw
import numpy as np


# 다운로드 기능(without_mask, with_mask, mask)
def download_image(kind):
    if kind == 'without_mask':
        api_url = 'https://api.github.com/repos/prajnasb/observations/contents/experiements/data/without_mask?ref=master'
        hds = {'User-Agent': 'Mozilla/5.0'}

        request = Request(api_url, headers=hds)
        response = urlopen(request)
        directory_bytes = response.read()
        directory_str = directory_bytes.decode('utf-8')

        contents = json.loads(directory_str)

        for i in range(len(contents)):
            content = contents[i]

            request = Request(content['download_url'])
            response = urlopen(request)
            image_data = response.read()

            if not os.path.exists('data'):
                os.mkdir('data')
            if not os.path.exists('data/without_mask'):
                os.mkdir('data/without_mask')

            image_file = open('data/without_mask/' + content['name'], 'wb')
            image_file.write(image_data)
            image_file.close()
            print('without_mask 이미지 다운로드 중(' + str(i + 1) + '/' + str(len(contents)) + '): ' + content['name'])
        print('without_mask 이미지 다운로드 완료')
    elif kind == 'with_mask':
        api_url = 'https://api.github.com/repos/prajnasb/observations/contents/experiements/data/with_mask?ref=master'
        hds = {'User-Agent': 'Mozilla/5.0'}

        request = Request(api_url, headers=hds)
        response = urlopen(request)
        directory_bytes = response.read()
        directory_str = directory_bytes.decode('utf-8')

        contents = json.loads(directory_str)

        for i in range(len(contents)):
            content = contents[i]

            request = Request(content['download_url'])
            response = urlopen(request)
            image_data = response.read()

            if not os.path.exists('data'):
                os.mkdir('data')
            if not os.path.exists('data/with_mask'):
                os.mkdir('data/with_mask')

            image_file = open('data/with_mask/' + content['name'], 'wb')
            image_file.write(image_data)
            image_file.close()
            print('with_mask 이미지 다운로드 중(' + str(i + 1) + '/' + str(len(contents)) + '): ' + content['name'])
        print('with_mask 이미지 다운로드 완료')
    elif kind == 'mask':
        mask_image_download_url = 'https://github.com/prajnasb/observations/raw/master/mask_classifier/Data_Generator/images/blue-mask.png'

        request = Request(mask_image_download_url)
        response = urlopen(request)
        image_data = response.read()

        if not os.path.exists('data'):
            os.mkdir('data')

        image_file = open('data/mask.png', 'wb')
        image_file.write(image_data)
        image_file.close()
        print('mask 이미지 다운로드 완료')


# 점과 점 사이의 거리
def distance_point_to_point(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


# 점과 직선 사이의 거리
def distance_point_to_line(point, line_point1, line_point2):
    if line_point1[0] == line_point2[0]:
        return np.abs(point[0] - line_point1[0])
    a = -(line_point1[1] - line_point2[1]) / (line_point1[0] - line_point2[0])
    b = 1
    c = -a * line_point1[0] - b * line_point1[1]
    return np.abs(a * point[0] + b * point[1] + c) / np.sqrt(a ** 2 + b ** 2)


def rotate_point(origin, point, radian):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + np.cos(radian) * (px - ox) - np.sin(radian) * (py - oy)
    qy = oy + np.sin(radian) * (px - ox) + np.cos(radian) * (py - oy)
    return qx, qy


# 마스크 합성
def mask_processing(face_image_file_name):
    # 이미지 경로 생성
    face_image_path = 'data/without_mask/' + face_image_file_name
    mask_image_path = 'data/mask.png'

    # 얼굴 영역 추출, 얼굴 랜드마크 추출
    face_image_np = face_recognition.load_image_file(face_image_path)
    face_locations = face_recognition.face_locations(face_image_np)
    face_landmarks = face_recognition.face_landmarks(face_image_np, face_locations)

    # 결과 이미지 생성
    face_image = Image.fromarray(face_image_np)
    mask_image = Image.open(mask_image_path)

    face_count = 0

    # 마스크 합성
    for face_landmark in face_landmarks:
        if ('nose_bridge' not in face_landmark) or ('chin' not in face_landmark):
            continue

        # 마스크 너비 보정값 (1.2)
        mask_width_ratio = 1.2

        # 마스크 높이 계산 (nose_bridge 2번째 점, chin 9번째 점 사이의 길이)
        mask_height = int(distance_point_to_point(face_landmark['nose_bridge'][1], face_landmark['chin'][8]))

        # 마스크 좌/우 분할
        mask_left_image = mask_image.crop((0, 0, mask_image.width // 2, mask_image.height))
        mask_right_image = mask_image.crop((mask_image.width // 2, 0, mask_image.width, mask_image.height))

        # 왼쪽 마스크 너비 계산
        mask_left_width = int(distance_point_to_line(face_landmark['chin'][0], face_landmark['nose_bridge'][0], face_landmark['chin'][8]) * mask_width_ratio)

        # 왼쪽 마스크 크기 조절
        mask_left_image = mask_left_image.resize((mask_left_width, mask_height))

        # 오른쪽 마스크 너비 계산
        mask_right_width = int(distance_point_to_line(face_landmark['chin'][16], face_landmark['nose_bridge'][0], face_landmark['chin'][8]) * mask_width_ratio)

        # 오른쪽 마스크 크기 조절
        mask_right_image = mask_right_image.resize((mask_right_width, mask_height))

        # 좌/우 마스크 연결
        mask_image = Image.new('RGBA', (mask_left_width + mask_right_width, mask_height))
        mask_image.paste(mask_left_image, (0, 0), mask_left_image)
        mask_image.paste(mask_right_image, (mask_left_width, 0), mask_right_image)

        # 얼굴 회전 각도 계산
        dx = face_landmark['chin'][8][0] - face_landmark['nose_bridge'][0][0]
        dy = face_landmark['chin'][8][1] - face_landmark['nose_bridge'][0][1]

        face_radian = np.arctan2(dy, dx)
        face_degree = np.rad2deg(face_radian)

        # 마스크 회전
        mask_degree = (90 - face_degree + 360) % 360
        mask_image = mask_image.rotate(mask_degree, expand=True)

        # 마스크 위치 계산
        mask_radian = np.deg2rad(-mask_degree)

        center_x = (face_landmark['nose_bridge'][1][0] + face_landmark['chin'][8][0]) // 2
        center_y = (face_landmark['nose_bridge'][1][1] + face_landmark['chin'][8][1]) // 2

        p1 = rotate_point((center_x, center_y), (center_x - mask_left_width, center_y - mask_height // 2), mask_radian)
        p2 = rotate_point((center_x, center_y), (center_x - mask_left_width, center_y + mask_height // 2), mask_radian)
        p3 = rotate_point((center_x, center_y), (center_x + mask_right_width, center_y - mask_height // 2), mask_radian)
        p4 = rotate_point((center_x, center_y), (center_x + mask_right_width, center_y + mask_height // 2), mask_radian)

        box_x = int(min(p1[0], p2[0], p3[0], p4[0]))
        box_y = int(min(p1[1], p2[1], p3[1], p4[1]))

        # 마스크 합성 (붙여넣기)
        face_image.paste(mask_image, (box_x, box_y), mask_image)

        face_count += 1

    # 결과 이미지 반환
    return face_image, face_count


# 데이터 생성
def generate_data():
    face_image_base_path = 'data/without_mask/'

    save_path = 'data/with_mask/'

    face_image_file_names = os.listdir(face_image_base_path)

    for i in range(len(face_image_file_names)):
        face_image_file_name = face_image_file_names[i]

        face_image, face_count = mask_processing(face_image_file_name)

        if face_count == 0:
            os.remove(face_image_base_path + face_image_file_name)
            print('얼굴 인식 실패(' + str(i + 1) + '/' + str(len(face_image_file_names)) + '): ' + face_image_file_name)
        else:
            if not os.path.exists(save_path):
                os.mkdir(save_path)

            face_image.save(save_path + face_image_file_name)
            print('마스크 합성 중(' + str(i + 1) + '/' + str(len(face_image_file_names)) + '): ' + face_image_file_name)
    print('마스크 합성 완료')


if __name__ == '__main__':
    # download_image('without_mask')
    generate_data()
