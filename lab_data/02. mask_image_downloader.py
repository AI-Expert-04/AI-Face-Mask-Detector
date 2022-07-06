# 마스크 이미지 다운로드
from urllib.request import Request, urlopen
import json
import os


def image_download(url, filepath):
    request = Request(url)
    response = urlopen(request)

    image_data = response.read()

    file = open(filepath, 'wb')
    file.write(image_data)
    file.close()
    print(url + '로 부터 ' + filepath + '에 다운로드 완료')
    # url로 부터 filepath에 다운로드 완료

mask_url = 'https://github.com/prajnasb/observations/raw/master/mask_classifier/Data_Generator/images/blue-mask.png'
image_download(mask_url, '../data/mask.png')
