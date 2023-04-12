import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle

def load_data_image(data_dir, width=300, height=300):
    x_data = []
    y_data = []
    dirs = os.listdir(data_dir)
    dirs = [dir for dir in dirs if not dir.startswith ('.')] #.DS_Store 제외
    for dir in dirs:
        #print(dir) #left face
        files = os.listdir(os.path.join(data_dir, dir))
        files = [file for file in files if not file.startswith ('.')] #.DS_Store 제외
        #files = files[:100] #데이터를 100개로 제한
        for file in files:
            #print(file) #63.jpg
            image = Image.open(os.path.join(data_dir, dir, file))
            image = image.resize((width, height))
            numpy_image = np.array(image) #이미지 타입을 넘파이 타입으로 변환
            x_data.append(numpy_image)
            if dir == 'skincancer':
                y_data.append(0)
            elif dir == 'non-skincancer':
                y_data.append(1)
    x_data = np.array(x_data)
    y_data = np.array(y_data)

    return x_data, y_data

##########데이터 로드

x_data, y_data = load_data_image('C:\\Users\\양승재\\opencv')

labels = ['skin cancer', 'non-skincancer']

##########데이터 분석

##########데이터 전처리

x_data = x_data.reshape(-1, 300 * 300 * 3)
x_data = (x_data - 127.5) / 127.5 #표준 정규화 ([-1, 1])

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=777, stratify=y_data)

##########모델 생성

model = SVC(kernel='rbf', C=1.0, gamma='auto', confidence=True)

##########모델 학습

model.fit(x_train, y_train)

##########모델 검증

print(model.score(x_train, y_train)) #

print(model.score(x_test, y_test)) #0.5

##########모델 예측

image = Image.open('actinic keratosis_original_ISIC_0025780.jpg_1a17fc6e-4ce6-4bf7-9ba2-15ec90ce44c0.jpg')
image = image.resize((300, 300))
numpy_image = np.array(image) 
numpy_image = numpy_image.reshape(300 * 300 * 3)
x_test = [numpy_image]
x_test = np.array(x_test)
x_test = (x_test - 127.5) / 127.5 #표준 정규화 ([-1, 1])

y_predict = model.predict(x_test)
label = labels[y_predict[0]]
y_predict = model.predict_proba(x_test)
confidence = y_predict[0][y_predict[0].argmax()]

print(label, confidence) 