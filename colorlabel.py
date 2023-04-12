import cv2
import sys
import numpy as np
import os

for n in os.listdir('./data_skincancer/'):
    path='./data_skincancer/'+ n
    img_color = cv2.imread(path, cv2.IMREAD_COLOR)

    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    _, th = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)          #임계처리

    cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(th)

    img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)                       

    for i in range(1, cnt): 
        (x, y, w, h, area) = stats[i]
        if(w>=200 and h>= 200):                              #라벨크기조정

            cv2.rectangle(img_color, (x, y, w, h), (0, 255, 0))   #라벨링

    label = img_color
    cv2.imwrite(f'./labeling_skincancer/{n}.jpg',img_color)

#목적: 원본사진에 라벨이 추가된 사진
#(07-03): 원본사진을 흑백처리->임계처리->라벨링= 흑백처리된 사진에 라벨링됨
#(07-04): 컬러 라벨완료