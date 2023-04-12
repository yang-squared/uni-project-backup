import cv2
import sys
import numpy as np
import os

for n in os.listdir('./data_skincancer/'):
    path='./data_skincancer/'+n
    src = cv2.imread(path, cv2.IMREAD_GRAYSCALE)   #이미지 파일 불러오기 추후 반복문 사용하여 폴더내에 모든 이미지 불러오기할 에정(완료)
    color_out=src.copy()
    color_out=255-color_out         #이미지 색반전
    if src is None:
        print('Image load failed!')
        sys.exit()
    _, src_bin = cv2.threshold(color_out, 0, 255, cv2.THRESH_OTSU)          #흑백처리
    cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(src_bin)
    dst = cv2.cvtColor(color_out, cv2.COLOR_GRAY2BGR)                       #이진화
    for i in range(1, cnt): 
        (x, y, w, h, area) = stats[i]
        if(w>=50 and h>=50):                              #라벨크기조정

            cv2.rectangle(dst, (x, y, w, h), (0, 255, 0))   #라벨링
    cv2.imwrite(f'./labeling_skincancer/{n}.jpg',src)      #지정된 경로에 레이블링된 이미지 저장 추후 반복문 사용하여 모든 이미지 저장할 예정(완료)

#이미지폴더 지정후 불러오기(완료)
#레이블링된 이미지들 폴더 지정후 저장하기(완료)
#위에 해결후 레이블링된 이미지들 학습