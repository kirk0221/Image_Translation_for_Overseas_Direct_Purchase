import paddleocr
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
from PIL import Image, ImageDraw, ImageFont
from googletrans import Translator
from lama.bin import predict as lama
import cv2
from collections import Counter
import numpy as np

# PATH

IMG_PATH = './dataset/' #하위 폴더의 dataset에서 이미지 데이터 받기
CHECK_PATH = "./check/" # inpainting을 위한 마스크와 손상된 이미지
TARGET_PATH = "./inpainting_image/" # inpainting 이미지
SAVE_PATH = './result/' #결과 이미지 result폴더에 넣기

FONT = 'HMKMRHD.TTF'

# function

def check_two_equal(a, b, c, d): # 입력된 4개의 숫자 중 2개의 숫자 확인 오차 2까지 허용
    return sum(abs(x - y) <= 2 for x, y in [(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)]) >= 2
    
def compare_pixels(pixel1, pixel2):
    diff = sum(abs(int(p1) - int(p2)) for p1, p2 in zip(pixel1, pixel2)) # 픽셀간의 차이 계산
    return diff / 3 >= 50 # 모든 차이 값의 평균이 50 이상 차이나면 True 반환


# OCR

image_file_name = os.listdir(IMG_PATH) #이미지 파일 이름 리스트 받아오기

ocr = paddleocr.PaddleOCR() # OCR 생성
translator = Translator() # 번역기 생성
words = [] # 번역 단어 저장
locations = [] # 위치정보 저장

translate_except_list = ['SAMSUNG', 'AKG', 'JLAB', 'CUCKOO', 'F', 'S', 'M', 'L'] # 번역하지 않을 단어 리스트

for i in range(len(image_file_name)):
    img_name = IMG_PATH + image_file_name[i] # 이미지 이름 전체 경로 가져오기

    image_mask_name = image_file_name[i].split('.')[0]

    ocr_result = ocr.ocr(img_name) # 이미지 ocr
    
    image = Image.open(img_name) # 이미지 열기

    width, height = image.size # 이미지 크기 가져오기

    mask_img = Image.new('RGB', (width, height), color='black') # 마스크를 위한 검은색 이미지 생성

    word = [] # 번역위해 리스트 생성
    location = [] # 위치 정보 저장 위해 리스트 생성

    for j in range(len(ocr_result[0])):
        if (ocr_result[0][j][1][1] > 0.8) and (ocr_result[0][j][1][0] not in translate_except_list): # 정확도 80%보다 크고 번역하지 않을 단어 리스트 안에 없을 경우 동작
            x1, y1 = ocr_result[0][j][0][0]  # 좌상단
            x2, y2 = ocr_result[0][j][0][1]  # 우상단
            x3, y3 = ocr_result[0][j][0][2]  # 우하단
            x4, y4 = ocr_result[0][j][0][3]  # 좌하단

            if check_two_equal(x1,x2,x3,x4) or check_two_equal(y1,y2,y3,y4):

                # 마진 5씩 주기
                x_max = max(x1,x2,x3,x4) + 5 
                x_min = min(x1,x2,x3,x4) - 5
                y_max = max(y1,y2,y3,y4) + 5
                y_min = min(y1,y2,y3,y4) - 5

                # 위치정보 저장
                locate = []
                locate.append(x_max)
                locate.append(x_min)
                locate.append(y_max)
                locate.append(y_min)
                location.append(locate)

                draw = ImageDraw.Draw(image)
                draw_mask = ImageDraw.Draw(mask_img)

                draw.polygon([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)], fill='white') # 흰색으로 원본 이미지 빈칸 만들기
                
                draw_mask.polygon([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)], fill='white') # 흰색으로 마스크 이미지 빈칸 만들기

                image.save(CHECK_PATH + image_file_name[i])
                mask_img.save((CHECK_PATH + image_mask_name + '_mask.png'))

                word.append(ocr_result[0][j][1][0]) # ocr추출 단어 저장
    words.append(word) # 외국어 추출 단어 저장
    locations.append(location) # 위치정보 저장

# Translation

for i in range(len(image_file_name)):
    for j in range(len(words[i])):
        words[i][j] = translator.translate(words[i][j], dest='ko').text # 한국어로 번역하여 대신 넣어줌

# Image Inpainting

lama.main() # lama 이용해 image inpainting

# Image Processing

target_file_name = os.listdir(TARGET_PATH)
for i in range(len(target_file_name)):
    img_name_path = TARGET_PATH + target_file_name[i] # inpainting한 이미지 전체 경로 가져오기
    original_img_path = IMG_PATH + image_file_name[i] # 원본 이미지 전체 경로 가져오기
    target_name = image_file_name[i].split('_')[0] # _mask 제거한 이미지 이름 가져오기
    target_image = Image.open(img_name_path) # inpainting한 이미지 가져오기
    draw = ImageDraw.Draw(target_image) # target 이미지에 그리기위해 draw변수에 넣어줌

    inpainting_img = cv2.imread(img_name_path)
    original_img = cv2.imread(original_img_path)

    for j in range(len(words[i])):
        x_max, x_min, y_max, y_min = locations[i][j] # 이미지 위치 받아오기
        word = words[i][j] # 이미지 단어 받아오기

        # 이미지 폰트 크기 조절

        img_width = int(x_max - x_min)
        img_height = int(y_max - y_min)
        
        font_size = int(y_max-y_min-10)     # 높이를 기준으로 폰트크기 초기화
        
        img_box = Image.new('RGB', (img_width, img_height))     # 바운딩 박스 크기와 동일한 이미지 생성
        draw_box = ImageDraw.Draw(img_box)                      
        
        font = ImageFont.truetype(FONT, font_size)
        
        text_width, text_height = draw_box.textsize(word, font=font)    # 랜더링한 텍스트의 너비, 높이 
     
        # 랜더링할 텍스트의 크기가 기존 바운딩 박스 크기보다 클 때 크기 조정
        while text_width > img_width or text_height > img_height:
            
            font_size -= 1
            font = ImageFont.truetype(FONT, font_size)
            text_width, text_height = draw_box.textsize(word, font=font)

        # 색상 추출

        color_lst = []
        for k in range(int(x_max - x_min - 10)):
            for l in range(int(y_max - y_min - 10)):
                inpainting_pixel_color = inpainting_img[int(y_min+5 + l), int(x_min+5 + k)]
                original_pixel_color = original_img[int(y_min+5 + l), int(x_min+5 + k)]
                if compare_pixels(original_pixel_color, inpainting_pixel_color) : # 오차값으로 비교해서 그보다 더 큰 경우에만
                    color_lst.append(str(original_pixel_color)) # 원본 이미지 픽셀의 글자부분으로 판단, 리스트에 넣어줌
            color_lst = list(filter(None, color_lst)) # 중간에 None 값들 제거
        counts = Counter(color_lst) # 리스트에서 최빈값을 구하고
        if len(counts.most_common()) == 0: # 만약 최빈값이 안나오면 기본값은 검은색임
            color = [0,0,0]
        else:
            original_count = Counter(original_pixel_color)
            count = counts.most_common()[i][0] # 최빈값을 리스트에 넣어줌
            count = count[1:-1]
            color = list(count.split(" "))
            color = list(filter(None, color)) # 중간에 None 값들 제거

        draw.text((x_min+5, y_min+15), word, font=font, anchor='lm', fill=(int(color[2]),int(color[1]),int(color[0]))) # 이미지에 글씨 그리기
    target_image.save(SAVE_PATH + target_name)
