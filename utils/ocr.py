import os
from aip import AipOcr
from PIL import Image

"""
ref: 待补全……
"""

APP_ID = '16742130'
API_KEY = '8Xz7ffedKiGjqzuezq6xb3kh'
SECRECT_KEY = 'xhtHCFaNTj2kIlfqE1QGPV360IiX6PIb'

def baiduOCR(img_path):
    """ 将单张图片发送到百度进行识别 """

    client = AipOcr(APP_ID, API_KEY, SECRECT_KEY)
    word_list = []
    with open(img_path, 'rb') as f:
        message = client.basicGeneral(f.read())
        for text in message.get('words_result', []):
            word_list.append(text.get("words", ''))
    return ",".join(word_list)

if __name__ == "__main__":
    img_path = "data/test_data/crop_imgs/000143.jpg"
    print(baiduOCR(img_path))
