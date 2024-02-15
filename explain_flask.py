# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 11:46:20 2024

@author: kai
"""


from flask import Flask, render_template, Response
'''
Flask 웹 프레임워크를 사용하기 위해 필요한 모듈들을 임포트합니다.
Flask: Flask 애플리케이션을 생성하기 위한 클래스입니다.
render_template: HTML 템플릿을 렌더링하기 위한 함수입니다.
Response: HTTP 응답을 생성하기 위한 클래스입니다.
'''

import cv2
'''
OpenCV(Open Source Computer Vision Library)는 이미지 및 비디오 처리를 위한 라이브러리입니다.
cv2는 OpenCV의 Python 바인딩을 나타냅니다.
'''
import numpy as np
'''
NumPy는 과학 및 수학 연산을 위한 파이썬 라이브러리입니다.
np는 관례적으로 사용되는 NumPy의 별칭입니다.
'''

from os import listdir
from os.path import isfile, join
'''
os 모듈은 운영 체제와 상호 작용하기 위한 함수를 제공합니다.
listdir: 지정된 디렉토리의 파일 및 디렉토리 목록을 반환합니다.
isfile: 주어진 경로가 파일인지 확인하는 함수.
join: 디렉토리 및 파일 이름의 결합을 안전하게 처리하는 함수.
'''

from scipy.spatial import distance
'''
scipy는 과학 및 공학 연산을 위한 라이브러리입니다.
distance 모듈은 공간 거리 및 유사성 메트릭을 계산하는 함수를 제공합니다.
'''




from imutils import face_utils
'''
imutils는 OpenCV 기능을 편리하게 사용할 수 있도록 도와주는 유틸리티 함수들을 제공하는 모듈입니다.
face_utils 모듈은 얼굴 관련 유틸리티 함수를 제공합니다.
'''

import dlib
'''
dlib은 머신 러닝 및 이미지 처리를 위한 라이브러리입니다.
주로 얼굴 감지 및 얼굴 특징점 예측을 위한 기능을 제공합니다.
'''




from pygame import mixer

'''
pygame은 게임 개발을 위한 크로스 플랫폼 파이썬 라이브러리입니다.
mixer 모듈은 음악 및 소리 재생을 관리하는 기능을 제공합니다.
'''



app = Flask(__name__)
'''
Flask 애플리케이션을 생성합니다. __name__은 현재 모듈의 이름을 나타내며, Flask에게 현재 모듈이 어플리케이션의 시작점이라는 것을 알려줍니다.
'''
cap = cv2.VideoCapture(0)
'''
OpenCV의 cv2.VideoCapture()를 사용하여 웹캠으로부터 비디오를 스트리밍합니다. 0은 기본 카메라를 나타냅니다.
'''
data_path = 'static/faces/'
'''
얼굴 이미지 데이터가 저장된 디렉토리의 경로를 data_path 변수에 저장합니다.
'static/faces/'는 상대 경로로, 현재 스크립트가 위치한 디렉토리에서 static/faces/ 경로를 나타냅니다.
'''
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]
'''
listdir(data_path)는 data_path 디렉토리에 있는 모든 파일 및 서브디렉토리의 리스트를 반환합니다.
isfile(join(data_path, f))는 각각의 파일에 대해 해당 경로가 파일인지 확인합니다.
리스트 컴프리헨션을 사용하여 data_path 디렉토리에 있는 파일만을 모은 onlyfiles 리스트를 생성합니다.
'''


Training_Data, Labels = [], []
'''
Training_Data: 얼굴 이미지 데이터를 저장하는 리스트
Labels: 해당 얼굴 이미지의 레이블(식별자)을 저장하는 리스트
'''



for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(i)
    
'''
for i, files in enumerate(onlyfiles):

enumerate(onlyfiles): onlyfiles 리스트의 각 항목과 해당 인덱스를 순회하는 반복 가능 객체를 생성합니다.
i: 현재 파일의 인덱스를 나타냅니다.
files: 현재 파일의 이름을 나타냅니다.
image_path = data_path + onlyfiles[i]

현재 파일의 경로를 data_path와 파일 이름을 합쳐서 생성합니다.
images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

cv2.imread: OpenCV를 사용하여 이미지를 읽어옵니다.
cv2.IMREAD_GRAYSCALE: 이미지를 그레이스케일로 읽어옵니다.
Training_Data.append(np.asarray(images, dtype=np.uint8))

np.asarray(images, dtype=np.uint8): 이미지를 NumPy 배열로 변환하고 데이터 타입을 부호 없는 8비트 정수로 지정합니다.
Training_Data.append(...): 변환된 이미지를 Training_Data 리스트에 추가합니다.
Labels.append(i)

현재 이미지에 대한 레이블(클래스 식별자)을 Labels 리스트에 추가합니다. 이 예제에서는 간단하게 이미지의 인덱스를 사용하여 레이블을 할당합니다.
'''






Labels = np.asarray(Labels, dtype=np.int32)

model = cv2.face.LBPHFaceRecognizer_create()
model.train(np.asarray(Training_Data), np.asarray(Labels))

'''
Labels = np.asarray(Labels, dtype=np.int32)

np.asarray(Labels, dtype=np.int32): Labels 리스트를 NumPy 배열로 변환하고 데이터 타입을 32비트 정수로 지정합니다.
이는 훈련 데이터의 레이블을 NumPy 배열로 변환하는 과정입니다.

model = cv2.face.LBPHFaceRecognizer_create()

cv2.face.LBPHFaceRecognizer_create(): LBPH 얼굴 인식기 객체를 생성합니다.
model.train(np.asarray(Training_Data), np.asarray(Labels))

model.train(...): 생성한 모델에 대해 훈련을 수행합니다.
np.asarray(Training_Data): 훈련 데이터의 이미지를 NumPy 배열로 변환하여 모델에 전달합니다.
np.asarray(Labels): 훈련 데이터의 레이블을 NumPy 배열로 변환하여 모델에 전달합니다.
'''




print("Model Training Complete!!!!!")






face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

mixer.init()
mixer.music.load("music.wav")
'''
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cv2.CascadeClassifier('haarcascade_frontalface_default.xml'): Haar Cascade 얼굴 분류기 객체를 생성합니다. 이 분류기는 이미지에서 얼굴을 감지하는 데 사용됩니다. 분류기는 'haarcascade_frontalface_default.xml' 파일을 사용하여 초기화됩니다.
mixer.init()

mixer.init(): pygame의 mixer 모듈을 초기화합니다. 음악을 재생하기 전에 필요한 초기화 단계입니다.
mixer.music.load("music.wav")

mixer.music.load("music.wav"): pygame의 mixer 모듈을 사용하여 'music.wav'라는 음악 파일을 로드합니다. 이후에 mixer.music.play() 등을 사용하여 음악을 재생할 수 있습니다.
'''




def face_detector(img, size=0.5):
    #face_detector 함수를 정의합니다. 이 함수는 이미지 (img)를 입력으로 받으며, 선택적으로 기본값이 0.5인 size 매개변수를 가지고 있습니다.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #OpenCV의 cv2.cvtColor 함수를 사용하여 입력 이미지 (img)를 BGR 색 공간에서 그레이스케일로 변환합니다.
    
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    #face_classifier의 detectMultiScale 메서드를 사용하여 그레이스케일 이미지에서 얼굴을 검출합니다.
    ##gray는 그레이스케일 이미지이며, 1.3과 5는 스케일 계수 및 최소 이웃값입니다.
    if faces is ():
        return img, []
    #얼굴이 검출되지 않았다면 (faces가 빈 튜플인 경우) 원본 이미지와 빈 리스트를 반환합니다.
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi = img[y:y + h, x:x + w]
        roi = cv2.resize(roi, (200, 200))

    return img, roi
'''
for (x, y, w, h) in faces::

faces에는 얼굴들의 좌표와 크기가 저장되어 있습니다.
(x, y)는 얼굴의 좌상단 좌표이고, (w, h)는 얼굴의 폭과 높이를 나타냅니다.
이 반복문은 모든 얼굴에 대해 아래의 작업을 수행합니다.
cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2):

cv2.rectangle 함수를 사용하여 이미지 (img)에 현재 얼굴 주위에 사각형을 그립니다.
(x, y)는 사각형의 좌상단 좌표이고, (x + w, y + h)는 우하단 좌표입니다.
(0, 255, 255)는 BGR 색상으로 노란색을 의미하며, 2는 사각형의 두께를 나타냅니다.
roi = img[y:y + h, x:x + w]:

현재 얼굴의 좌표를 이용하여 이미지 (img)에서 해당 얼굴의 관심 영역(ROI)를 추출합니다.
img[y:y + h, x:x + w]는 이미지에서 해당 영역을 슬라이싱하는 방법입니다.
roi = cv2.resize(roi, (200, 200)):

추출된 관심 영역 (ROI)을 cv2.resize 함수를 사용하여 크기를 (200, 200)으로 조정합니다.
return img, roi:

반복문을 통해 모든 얼굴에 대해 위의 작업을 수행한 후, 최종적으로 수정된 이미지 (img)와 마지막 얼굴의 크기가 조정된 ROI (roi)를 반환합니다.
'''






def generate_frames():
    '''
    generate_frames 함수를 정의하고, 무한 루프를 시작합니다. 이 함수는 계속해서 프레임을 생성하는 역할을 합니다.
    '''
    while True:
        success, frame = cap.read()
        '''
        cap.read()를 통해 웹캠에서 프레임을 읽어옵니다. success는 프레임을 성공적으로 읽어왔는지 여부를 나타내며, frame에 현재 프레임이 저장됩니다.
        '''
        if not success:
            break
        
        #만약 success가 False인 경우, 프레임을 읽어오는 데 실패했다는 것이므로 루프를 종료합니다.
        
        else:
            img, face = face_detector(frame)
            
            
            #프레임을 성공적으로 읽어왔다면, face_detector 함수를 사용하여 얼굴을 검출하고, 검출된 얼굴 이미지와 함께 수정된 원본 이미지를 받아옵니다.
            
            try:
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                result = model.predict(face)

                if result[1] < 500:
                    confidence = int(100 * (1 - (result[1]) / 300))
                    display_string = str(confidence) + '%'
                cv2.putText(img, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (250, 120, 255), 2)
                
                
                

                if confidence > 75:
                    cv2.putText(img, "Unlocked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(img, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

            except:
                cv2.putText(img, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
                pass

            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


"""
if result[1] < 500::

얼굴 인식 모델이 반환한 결과의 두 번째 요소(result[1])가 500 미만인 경우에만 아래의 코드를 실행합니다. 이는 모델이 얼굴을 신뢰할 정도로 확실한 경우에만 해당합니다.
confidence = int(100 * (1 - (result[1]) / 300)):

얼굴 신뢰도를 계산합니다. 이 부분은 간단한 선형 변환을 통해 계산되며, 계산된 값은 0에서 100 사이의 정수로 제한됩니다.
result[1]이 작을수록 얼굴이 모델에 의해 높은 신뢰도로 인식되었음을 나타냅니다.
display_string = str(confidence) + '%':

계산된 얼굴 신뢰도를 문자열로 변환하고, 뒤에 '%' 기호를 추가하여 화면에 표시될 텍스트를 생성합니다.
cv2.putText(img, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (250, 120, 255), 2):

생성된 텍스트를 이미지에 추가합니다.
cv2.putText 함수를 사용하여 텍스트를 이미지에 추가하고 있습니다.
img는 원본 이미지입니다.
display_string은 화면에 표시될 텍스트입니다.
(100, 120)은 텍스트의 시작 위치입니다.
cv2.FONT_HERSHEY_COMPLEX은 폰트의 형태를 나타냅니다.
1은 폰트 크기를 나타냅니다.
(250, 120, 255)는 텍스트의 색상을 나타냅니다. 이 경우, 보라색입니다.
2는 텍스트의 두께를 나타냅니다.
"""

'''
얼굴의 신뢰도가 75%보다 크면 "Unlocked", 그렇지 않으면 "Locked"라는 문자열을 화면에 추가합니다.
'''


'''
예외 처리 부분에서는 얼굴을 찾지 못한 경우를 다루고 있습니다. cv2.putText 함수를 사용하여 "Face Not Found"라는 문자열을 이미지에 추가하고 있습니다. 이 경우 텍스트의 위치는 (250, 450)이며, 폰트 크기는 1, 색상은 빨간색으로 설정되어 있습니다.
'''



'''
ret, buffer = cv2.imencode('.jpg', img):

cv2.imencode 함수를 사용하여 이미지(img)를 JPEG 형식으로 인코딩합니다.
ret은 인코딩의 성공 여부를 나타내는 부울 값입니다.
buffer에는 인코딩된 이미지가 바이트 형태로 저장됩니다.
frame = buffer.tobytes():

buffer를 바이트 형태로 변환하여 frame에 저장합니다.

yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n'):

yield 문을 사용하여 인코딩된 이미지를 웹 스트리밍 프레임으로 반환합니다.
반환되는 데이터는 마임 타입이 포함된 프레임입니다.
b'--frame\r\n'은 프레임의 시작을 나타냅니다.
b'Content-Type: image/jpeg\r\n\r\n'은 마임 타입이 JPEG 이미지임을 나타냅니다.
이어서 실제 이미지 데이터가 옵니다.
마지막으로 b'\r\n\r\n'은 프레임의 끝을 나타냅니다.
'''




@app.route('/')
def main():
    return render_template('main.html')

@app.route('/first_page')
def first_page():
    return render_template('first.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/second_page')
def second_page():
    return render_template('second.html')

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear
'''
위에것은 사진 참고 
'''

thresh = 0.15
frame_check = 40
'''
thresh는 눈을 감은 것으로 판단할 EAR의 임계값을 나타냅니다.
frame_check는 눈의 상태를 확인하기 위한 연속된 프레임 수를 나타냅니다. 즉, 이 값만큼 연속된 프레임 동안 눈이 감겨 있다면 해당 상태로 간주합니다.
'''
'''
눈의 종횡비를 이용하여 눈의 상태를 감지하는데 사용될 것으로 예상됩니다. 눈의 EAR이 임계값(thresh)보다 작고, 연속된 프레임 수가 frame_check 이상이면 눈이 감겨 있다고 판단할 수 있습니다.

'''
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
'''
detect = dlib.get_frontal_face_detector():
dlib 라이브러리를 사용하여 얼굴을 감지하기 위한 얼굴 검출기를 생성합니다.


predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat"):
dlib를 사용하여 얼굴의 랜드마크를 예측하기 위한 모델을 설정합니다. 여기서 "shape_predictor_68_face_landmarks.dat"는 미리 학습된 모델 파일의 경로를 나타냅니다.
'''

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
'''
face_utils.FACIAL_LANDMARKS_68_IDXS: 68개의 얼굴 랜드마크에 대한 인덱스 정보를 담고 있는 딕셔너리입니다. 이 딕셔너리는 왼쪽 눈, 오른쪽 눈, 코, 입 등의 랜드마크에 대한 시작과 끝 인덱스를 제공합니다.

"left_eye" 및 "right_eye": 왼쪽 눈과 오른쪽 눈에 해당하는 랜드마크의 키입니다.

(lStart, lEnd) 및 (rStart, rEnd): 왼쪽 눈과 오른쪽 눈에 대한 랜드마크의 시작과 끝 인덱스를 나타내는 튜플입니다. 이들 변수에는 각각 왼쪽 눈과 오른쪽 눈의 랜드마크 인덱스 범위가 할당됩니다.
'''

def generate_frames_with_eye_detection():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            subjects = detect(gray, 0)
            for subject in subjects:
                shape = predict(gray, subject)
                shape = face_utils.shape_to_np(shape)
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
#                cv2.putText(frame, f'EAR: {ear}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
'''
프레임 읽기 및 전처리:

while True:: 무한 루프를 시작하여 계속해서 웹캠에서 프레임을 읽습니다.
success, frame = cap.read(): 웹캠에서 프레임을 읽어옵니다. success가 True이면 프레임을 성공적으로 읽은 것이고, frame에 현재 프레임이 저장됩니다.
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY): 읽어온 프레임을 흑백으로 변환합니다. 얼굴 감지를 위해서는 흑백 이미지가 사용됩니다.
얼굴 감지 및 랜드마크 예측:

subjects = detect(gray, 0): 흑백 이미지에서 얼굴을 감지하고, 감지된 얼굴 정보를 subjects에 저장합니다.
for subject in subjects:: 감지된 각 얼굴에 대해 반복합니다.
shape = predict(gray, subject): 각 얼굴에 대해 랜드마크(눈, 코, 입 등의 특징점)를 예측합니다.
shape = face_utils.shape_to_np(shape): 얼굴 랜드마크를 NumPy 배열로 변환하여 다루기 쉽게 합니다.
눈 좌표 추출 및 EAR 계산:

leftEye = shape[lStart:lEnd], rightEye = shape[rStart:rEnd]: 각 얼굴에서 왼쪽 눈과 오른쪽 눈의 좌표를 추출합니다.
leftEAR = eye_aspect_ratio(leftEye), rightEAR = eye_aspect_ratio(rightEye): 각 눈의 EAR을 계산합니다.
평균 EAR 계산 및 시각적 표시:

ear = (leftEAR + rightEAR) / 2.0: 두 눈의 EAR의 평균을 계산합니다.
leftEyeHull = cv2.convexHull(leftEye), rightEyeHull = cv2.convexHull(rightEye): 눈 주변에 볼록 다각형(Convex Hull)을 생성합니다.
cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1): 왼쪽 눈 주변에 생성된 볼록 다각형을 프레임에 그립니다.
cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1): 오른쪽 눈 주변에 생성된 볼록 다각형을 프레임에 그립니다.
'''
                if ear < thresh:
                    cv2.putText(frame, "Wake up!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    mixer.music.play()

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
'''
눈의 상태 확인 및 경고 표시:

if ear < thresh:: 눈의 종횡비(EAR)가 설정한 임계값(thresh)보다 낮은지 확인합니다.
cv2.putText(frame, "Wake up!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2): 눈의 상태가 졸음에 해당하는 경우 프레임에 "Wake up!" 메시지를 표시합니다. 이 메시지는 프레임의 좌표 (10, 60)에 빨간색으로 표시되며, 폰트 크기는 0.7, 폰트 스타일은 cv2.FONT_HERSHEY_SIMPLEX, 두께는 2로 설정됩니다.
음악 재생:

mixer.music.play(): 눈이 감지되어 "Wake up!" 메시지가 표시된 경우에는 mixer 모듈을 사용하여 설정된 음악을 재생합니다. (mixer는 일반적으로 Pygame 라이브러리에서 제공하는 것으로 가정합니다. mixer.music.play()는 음악을 재생하는 함수로, 이전에 Pygame과 함께 초기화되어 있어야 합니다.)
프레임 인코딩 및 반환:

ret, buffer = cv2.imencode('.jpg', frame): 프레임을 JPEG 형식으로 인코딩하여 성공 여부와 인코딩된 바이트를 반환합니다. ret은 성공 여부를 나타내는 불리언 값이고, buffer에는 인코딩된 이미지가 저장됩니다.
frame = buffer.tobytes(): 인코딩된 바이트를 다시 frame 변수에 할당합니다.
yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n'): 프레임을 반환하는데, 이는 웹 페이지에 스트리밍할 때 사용됩니다. 반환된 데이터는 MIME 형식으로 이루어져 있으며, 이미지의 바이트 데이터가 포함된 HTTP 멀티파트 응답입니다. 이 부분은 주로 웹 기반의 스트리밍 서비스에서 사용됩니다.






'''
@app.route('/video_feed_with_eye_detection')
def video_feed_with_eye_detection():
    return Response(generate_frames_with_eye_detection(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

'''
/video_feed_with_eye_detection' 엔드포인트 정의:

@app.route('/video_feed_with_eye_detection'): '/video_feed_with_eye_detection' 경로에 대한 엔드포인트를 정의합니다.
def video_feed_with_eye_detection():: '/video_feed_with_eye_detection' 엔드포인트의 핸들러 함수를 정의합니다.
스트리밍 응답 생성:

generate_frames_with_eye_detection(): 눈 감지와 관련된 프레임을 생성하는 함수인 generate_frames_with_eye_detection를 호출하여 스트리밍하는 응답을 생성합니다.
Response(generate_frames_with_eye_detection(), mimetype='multipart/x-mixed-replace; boundary=frame'): Response 객체를 생성하여 스트리밍 응답을 구성합니다. mimetype은 멀티파트 스트리밍을 나타내며, boundary는 각 프레임을 구분하기 위한 구분자입니다. 이 응답 객체가 클라이언트로 전송되면, 클라이언트는 이 구분자를 기준으로 프레임을 분리하고 갱신합니다.
Flask 앱 실행:

if __name__ == '__main__':: 스크립트가 직접 실행될 때만 아래의 코드 블록이 실행되도록 합니다.
app.run(host='0.0.0.0', port=8080): Flask 앱을 실행하고, 외부에서 접근 가능한 0.0.0.0 주소와 8080 포트를 사용합니다. 이는 개발 서버를 실행하는 코드로, 실제 운영 환경에서는 보안상의 이유로 사용되지 않습니다.
'''
