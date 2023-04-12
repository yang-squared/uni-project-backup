import cv2
import sys
import numpy as np
import os

src = cv2.imread('actinic keratosis_original_c04832e1-6e77-4981-9f72-e30ea97b9a32.jpg', cv2.IMREAD_GRAYSCALE)   #이미지 파일 불러오기(흑백)
color_out=src.copy()
color_out=255-color_out         #이미지 색반전
if src is None:
    print('Image load failed!')
    sys.exit()

_, src_bin = cv2.threshold(color_out, 0, 255, cv2.THRESH_OTSU)          #이진화

cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(src_bin)

dst = cv2.cvtColor(color_out, cv2.COLOR_GRAY2BGR)                      #라벨링 준비 

for i in range(1, cnt): 
    (x, y, w, h, area) = stats[i]
    if(w>=50 and h >= 50):                              #라벨크기조정

        cv2.rectangle(dst, (x, y, w, h), (0, 255, 0))   #라벨링

cv2.imshow('color_out',color_out)
cv2.imshow('src_bin', src_bin)
cv2.imshow('dst', dst)
cv2.imshow('src',src)
cv2.waitKey()
cv2.destroyAllWindows()