
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




<img src="https://github.com/JEMinn/open_CV/assets/160000163/78457f6c-6f03-4c7f-9be2-77c7bed02c94"  width="400" height="400"/>


<img src="https://github.com/JEMinn/open_CV/assets/160000163/52f0453a-3599-437b-84bd-ce29c2370ba9"  width="400" height="400"/>

<img src="https://github.com/JEMinn/open_CV/assets/160000163/3c160c2b-14a3-4370-8222-00fcec300957"  width="500" height="300"/>



<img src="https://github.com/JEMinn/open_CV/assets/160000163/d983d055-dc43-4b92-9432-948771fc5059"  width="300" height="150"/>

<img src="https://github.com/JEMinn/open_CV/assets/160000163/a2606001-0fa7-4e24-a3ce-ad9b1c096f2c"  width="300" height="150"/>




