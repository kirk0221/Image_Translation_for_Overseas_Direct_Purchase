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
    
def compare_pixels(pixel1, pixel2, a):
    diff = sum(abs(int(p1) - int(p2)) for p1, p2 in zip(pixel1, pixel2)) # 픽셀간의 차이 계산
    return diff / 3 >= a # 모든 차이 값의 평균이 a 이상 차이나면 True 반환

def contrast_ratio(B1, B2):
    # 각각의 색상 값을 0~1 범위로 정규화
    normalized_B1 = [c/255 for c in B1]
    normalized_B2 = [c/255 for c in B2]

    # Relative luminance 계산
    luminance_B1 = (normalized_B1[2] * 0.0722 + normalized_B1[1] * 0.7152 + normalized_B1[0] * 0.2126)
    luminance_B2 = (normalized_B2[2] * 0.0722 + normalized_B2[1] * 0.7152 + normalized_B2[0] * 0.2126)
    # Contrast ratio 계산
    contrast = (max(luminance_B1, luminance_B2) + 0.05) / (min(luminance_B1, luminance_B2) + 0.05)
    return contrast

def list_most_common(list_counts, a):
        list_count = list_counts.most_common()[a][0] # 최빈값을 리스트에 넣어주기
        list_count = list_count[1:-1] # 대괄호 제거
        list_color = list(list_count.split(" ")) # 띄어쓰기를 기준으로 나눠주기
        list_color = list(filter(None,list_color)) # 중간에 None 값들 제거
        list_color = [int(list_color[0]),int(list_color[1]),int(list_color[2])] # 정수로 변환
        return list_color

# OCR

image_file_name = os.listdir(IMG_PATH) #이미지 파일 이름 리스트 받아오기
ocr = paddleocr.PaddleOCR() # OCR 생성
translator = Translator() # 번역기 생성
name_location_word_font_color_list = [] # 이미지 정보 리스트

translate_except_list = ['SAMSUNG', 'AKG', 'JLAB', 'IPX7', 'CUCKOO', 'F', 'S', 'M', 'L'] # 번역하지 않을 단어 리스트

file_length = len(image_file_name)

for i in range(file_length):
    name_location_word_font_color_list.append([image_file_name[i]])
    img_name = IMG_PATH + image_file_name[i] # 이미지 이름 전체 경로 가져오기
    
    image_mask_name = image_file_name[i].split('.')[0]

    ocr_result = ocr.ocr(img_name) # 이미지 ocr
    
    image = Image.open(img_name) # 이미지 열기

    width, height = image.size # 이미지 크기 가져오기

    mask_img = Image.new('RGB', (width, height), color='black') # 마스크를 위한 검은색 이미지 생성

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

                draw = ImageDraw.Draw(image)
                draw_mask = ImageDraw.Draw(mask_img)

                draw.polygon([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)], fill='white') # 흰색으로 원본 이미지 빈칸 만들기
                
                draw_mask.polygon([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)], fill='white') # 흰색으로 마스크 이미지 빈칸 만들기

                image.save(CHECK_PATH + image_file_name[i])
                mask_img.save((CHECK_PATH + image_mask_name + '_mask.png'))

                name_location_word_font_color_list[i].append([locate, ocr_result[0][j][1][0]]) # 위치 및 ocr추출 단어 저장

# Translation

for i in range(file_length):
    for j in range(1,len(name_location_word_font_color_list[i])):
        name_location_word_font_color_list[i][j][1] = translator.translate(name_location_word_font_color_list[i][j][1], dest='ko').text # 한국어로 번역하여 대신 넣어줌

# Image Inpainting

lama.main() # lama 이용해 image inpainting

# Image Processing

target_file_name = os.listdir(TARGET_PATH)
for i in range(file_length):
    img_name_path = TARGET_PATH + target_file_name[i] # inpainting한 이미지 전체 경로 가져오기
    original_img_path = IMG_PATH + image_file_name[i] # 원본 이미지 전체 경로 가져오기
    target_name = image_file_name[i].split('_')[0] # _mask 제거한 이미지 이름 가져오기
    target_image = Image.open(img_name_path) # inpainting한 이미지 가져오기

    inpainting_img = cv2.imread(img_name_path) # inpainting 된 이미지를 cv2 이미지로 읽기
    original_img = cv2.imread(original_img_path) # original 이미지를 cv2 이미지로 읽긴

    for j in range(1,len(name_location_word_font_color_list[i])):
        x_max, x_min, y_max, y_min = name_location_word_font_color_list[i][j][0] # 이미지 위치 받아오기
        word = name_location_word_font_color_list[i][j][1] # 이미지 단어 받아오기

        # 이미지 폰트 크기 조절
        img_width = int(x_max - x_min)
        img_height = int(y_max - y_min)
        
        font_size = int(y_max-y_min-10) # 높이를 기준으로 폰트크기 초기화
        
        img_box = Image.new('RGB', (img_width, img_height)) # 바운딩 박스 크기와 동일한 이미지 생성
        draw_box = ImageDraw.Draw(img_box)                      
        
        font = ImageFont.truetype(FONT, font_size)
        
        text_width, text_height = draw_box.textsize(word, font=font) # 랜더링한 텍스트의 너비, 높이 
     
        # 랜더링할 텍스트의 크기가 기존 바운딩 박스 크기보다 클 때 크기 조정
        while text_width > img_width or text_height > img_height:
            
            font_size -= 1
            font = ImageFont.truetype(FONT, font_size)
            text_width, text_height = draw_box.textsize(word, font=font)

        # 색상 추출
        color_lst = []
        background_color_lst = []
        for k in range(int(x_max - x_min - 10)): # 픽셀 x축 이동
            for l in range(int(y_max - y_min - 10)): # 픽셀 y축 이동
                background_pixel_color = inpainting_img[int(y_min+5 + l), int(x_min+5 + k)]
                background_color_lst.append(str(background_pixel_color))
                original_pixel_color = original_img[int(y_min+5 + l), int(x_min+5 + k)]
                if compare_pixels(original_pixel_color, background_pixel_color,80) : # 오차값으로 비교해서 그보다 더 큰 경우에만
                    color_lst.append(str(original_pixel_color)) # 원본 이미지 픽셀의 글자부분으로 판단, 리스트에 넣어줌
            color_lst = list(filter(None, color_lst)) # 중간에 None 값들 제거
        counts = Counter(color_lst) # 리스트에서 최빈값을 구하고
        background_color_lst = list(filter(None, background_color_lst)) # 중간에 None 값들 제거
        background_counts = Counter(background_color_lst) # 배경 색 최빈값 구하기
        background_color = list_most_common(background_counts, 0)
        if len(counts.most_common()) == 0: # 만약 최빈값이 안나오면 기본값은 검은색임
            color = [0,0,0]
        else:
            if not compare_pixels(background_color, [255,255,255], 10):
                color = [0,0,0]
            else:
                color = list_most_common(counts, 0)
        name_location_word_font_color_list[i][j].append(font) # 폰트 크기 정보 저장
        name_location_word_font_color_list[i][j].append(tuple(color[2],color[1],color[0])) # 색상 정보 저장

# Draw Image

for i in range(file_length):
    img_name_path = TARGET_PATH + target_file_name[i] # inpainting한 이미지 전체 경로 가져오기
    original_img_path = IMG_PATH + image_file_name[i] # 원본 이미지 전체 경로 가져오기
    target_name = image_file_name[i].split('_')[0] # _mask 제거한 이미지 이름 가져오기
    target_image = Image.open(img_name_path) # inpainting한 이미지 가져오기
    draw = ImageDraw.Draw(target_image) # target 이미지에 그리기위해 draw변수에 넣어줌

    inpainting_img = cv2.imread(img_name_path)
    original_img = cv2.imread(original_img_path)
    for j in range(1,len(name_location_word_font_color_list[i])):
        x_max, x_min, y_max, y_min = name_location_word_font_color_list[i][j][0] # 이미지 위치 받아오기
        draw.text(xy=(x_min+5, y_min+15), text=name_location_word_font_color_list[i][j][1], font=name_location_word_font_color_list[i][j][2], anchor='lm', fill=(name_location_word_font_color_list[i][j][3])) # 이미지에 글씨 그리기
    target_image.save(SAVE_PATH + name_location_word_font_color_list[i][0])