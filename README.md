
### 프로젝트 명 : 차량 얼굴인식 및 졸음운전 방지 서비스 (Facial recognition & drawsy detection Project)
---

- **프로젝트 소개**

  - **참여자**
    - 민지은, 송창훈, 양승원 총 3명

  - **수행일정**
  ![image](https://github.com/JEMinn/open_CV/assets/160000163/2421b2a6-4c83-40d9-aa23-671fda58c0ec)

  - **Flow Chart**
  <img src="https://github.com/JEMinn/open_CV/assets/160000163/53620040-6e85-4dbc-9a70-0f9560c980cd"  width="430" height="500"/>

  - **목적**
    
    - 학습한 딥러닝을 토대로 자동차의 불법적인 사용이나 졸음 운전으로 인한 교통사고로 인한 사망 및 부상을 방지하기 위해 해당 프로젝트 기획함
    - 또한 등록된 운전자 식별 및 그에 따른 기록은 운전자의 안전 및 차량 사용 통계를 수집할 수 있는 기회 제공할 수 있는 기능 추가

  - **기능**
    - LBPH 모델로 얼굴 인식을 통한 등록자 구분
    - LANDMARK, YOLOV8 두가지 모델을 이용하여 졸음 감지 구현


- **모델 소개**
  1. LBPH 모델 (Local Binary Pattern Histogram)
     - 얼굴 인식 및 검출을 위한 특징 추출 방법
     - LBPH 이미지의 각 픽셀에 대해 해당 픽셀의 이웃 픽셀과 비교하여 이진 패턴을 생성하여 LBP 히스토그램을 계산하고 이를 결합하는 방식
     - LBPH 주요 단계
       - 이미지를 그레이스케일로 변환
       - 이미지의 각 픽셀에 대해 중심 픽셀과 이웃 픽셀 간의 대소 비교를 수행하여 이진 패턴을 생성
       - 이진 패턴을 기반으로 지역 이진 패턴(Local Binary Patterns, LBP)을 생성. 각 픽셀은 이웃 픽셀과 비교하여 0 또는 1의 값을 가짐
         <img src="https://github.com/JEMinn/open_CV/assets/160000163/1ca45996-2026-45e0-88aa-225958d27855"  width="400" height="100"/>
       - 이미지를 격자로 나누고 각 격자에 대해 LBP 히스토그램을 계산
         <img src="https://github.com/JEMinn/open_CV/assets/160000163/a0220f57-2d56-4b23-aa4c-d011daf97046"  width="550" height="130"/>
       - 각 격자의 LBP 히스토그램을 하나의 특징 벡터로 결합
       - 결합된 특징 벡터를 사용하여 얼굴을 인식하거나 분류됨

  2. LANDMARK
    - 'dlib'에서 제공되는 사전 훈련된 모델로, 얼굴(눈, 코, 입, 귀 등)에 대한 68개의 랜드마크를 예측하는 데 사용
    - 얼굴 보정, 감정 인식, 표정 분석, 안경 착용 여부 감지 등의 작업에 활용
    - 해당 프로젝트는 졸음운전 방지를 기반으로 기획되어 눈에 대한 랜드마크 이용하였음
    - 눈의 종횡비로 눈의 열린 정도(가로 및 세로)를 측정하여 아래의 사진과 같이 졸음 탐지
      <img src="https://github.com/JEMinn/open_CV/assets/160000163/52f0453a-3599-437b-84bd-ce29c2370ba9"  width="400" height="200"/>
    - 눈의 종횡비 계산식
      <img src="https://github.com/JEMinn/open_CV/assets/160000163/70aae36a-597d-4484-8312-ff0d3b53668c"  width="180" height="100"/>

  3. YOLOv8(You Only Look Once)
    - 객체 검출을 위한 딥러닝 모델로, 단일 네트워크 구조를 사용하여 이미지에서 객체의 위치와 클래스를 실시간으로 예측
    - 객체 검출 및 추적, 자율 주행 차량, 보안 시스템 등 다양한 응용 분야에서 사용
      <img src="https://github.com/JEMinn/open_CV/assets/160000163/3c160c2b-14a3-4370-8222-00fcec300957"  width="400" height="200"/>
    - 아래와 같이 모델이 예측한 빨간색 바운딩 박스(predicted bounding box)와 사람이 사물의 위치에 라벨링한 초록색 정답 바운딩 박스(ground-truth bounding box)가 최대한 가까워지도록 학습
    - IoU 임계값이 0.5 이상인 경우, 모델이 예측한 박스와 실제 객체의 박스가 충분히 겹친 것으로 간주되어 해당 예측은 일반적으로 정확하게 객체를 감지한 것으로 판단
      <img src="https://github.com/JEMinn/open_CV/assets/160000163/d983d055-dc43-4b92-9432-948771fc5059"  width="300" height="150"/>
      <img src="https://github.com/JEMinn/open_CV/assets/160000163/a2606001-0fa7-4e24-a3ce-ad9b1c096f2c"  width="300" height="150"/>

- **내용 소개**



<img src="https://github.com/JEMinn/open_CV/assets/160000163/78457f6c-6f03-4c7f-9be2-77c7bed02c94"  width="400" height="400"/>










