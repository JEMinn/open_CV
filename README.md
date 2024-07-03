### 프로젝트 명 : 차량 얼굴인식 및 졸음운전 방지 서비스 (Facial recognition & drawsy detection Project)
---

- **프로젝트 소개**

  - **목적**
    - 졸음운전과 차량 도난 방지를 목표로, 딥러닝을 활용하여 자동차의 불법적인 사용과 졸음 운전으로 인한 사고를 예방하고, 등록된 운전자의 안전 및 차량 사용 통계를 수집하는 기능 추가

  - **기능**
    - LBPH 모델로 얼굴 인식을 통한 등록자 구분
    - LANDMARK, YOLOV8 모델을 이용한 졸음 감지
    - HTML&CSS를 이용한 FLASK 웹페이지 제작 

  - **수행일정**
  ![image](https://github.com/JEMinn/open_CV/assets/160000163/2421b2a6-4c83-40d9-aa23-671fda58c0ec)

  - **Flow Chart**
  <img src="https://github.com/JEMinn/open_CV/assets/160000163/53620040-6e85-4dbc-9a70-0f9560c980cd"  width="430" height="500"/>

---

- **모델 소개**
  1. **LBPH 모델 (Local Binary Pattern Histogram)**
     - 얼굴 인식 및 검출을 위한 특징 추출 방법
     - 그레이스케일 변환, 이진 패턴 생성, LBP 히스토그램 계산 후 결합
     - 이미지 격자 나눔 후 LBP 히스토그램 계산
         
         <img src="https://github.com/JEMinn/open_CV/assets/160000163/1ca45996-2026-45e0-88aa-225958d27855"  width="400" height="100"/>

         <img src="https://github.com/JEMinn/open_CV/assets/160000163/a0220f57-2d56-4b23-aa4c-d011daf97046"  width="550" height="130"/>

      
         　
  2. **LANDMARK**
    - 'dlib'의 사전 훈련된 모델로 얼굴의 68개 랜드마크 예측
    - 졸음운전 방지를 위해 눈 종횡비 이용<br>
      <img src="https://github.com/JEMinn/open_CV/assets/160000163/52f0453a-3599-437b-84bd-ce29c2370ba9"  width="400" height="200"/>
    - 눈의 종횡비 계산식
      
      <img src="https://github.com/JEMinn/open_CV/assets/160000163/70aae36a-597d-4484-8312-ff0d3b53668c"  width="250" height="80"/>
　
  3. **YOLOv8(You Only Look Once)**
      
    - 객체 검출을 위한 딥러닝 모델로, 단일 네트워크 구조를 사용하여 이미지에서 객체의 위치와 클래스를 실시간으로 예측
  　
    - 객체 검출 및 추적, 자율 주행 차량, 보안 시스템 등 다양한 응용 분야에서 사용
  　
      <img src="https://github.com/JEMinn/open_CV/assets/160000163/3c160c2b-14a3-4370-8222-00fcec300957"  width="400" height="200"/>
    - 아래와 같이 모델이 예측한 빨간색 바운딩 박스(predicted bounding box)와 사람이 사물의 위치에 라벨링한 초록색 정답 바운딩 박스(ground-truth bounding box)가 최대한 가까워지도록 학습
    - IoU 임계값이 0.5 이상인 경우, 모델이 예측한 박스와 실제 객체의 박스가 충분히 겹친 것으로 간주되어 해당 예측은 일반적으로 정확하게 객체를 감지한 것으로 판단
      　
      <img src="https://github.com/JEMinn/open_CV/assets/160000163/d983d055-dc43-4b92-9432-948771fc5059"  width="300" height="150"/>
      　
      <img src="https://github.com/JEMinn/open_CV/assets/160000163/a2606001-0fa7-4e24-a3ce-ad9b1c096f2c"  width="300" height="150"/>

---

- **내용 소개**
  1. **데이터 수집**
    <img src="https://github.com/JEMinn/open_CV/assets/160000163/51f1b888-c25e-4dcc-8646-b2e825917047"  width="350" height="200"/>
    
    - LBPH와 YOLOv8 모델은 사진을 직접 촬영하여 수집
    - LANDMARK 모델의 경우 눈의 횡종비로 측정하기에 따로 수집이 필요하지 않음

　
  2. **데이터 전처리**

 　
    <img src="https://github.com/JEMinn/open_CV/assets/160000163/e6cde029-13ab-4102-a15c-d91717c9f0cf"  width="350" height="200"/>

  3. **모델 학습 및 평가**
     
    - LBPH
    
      <img src="https://github.com/JEMinn/open_CV/assets/160000163/78457f6c-6f03-4c7f-9be2-77c7bed02c94"  width="300" height="300"/>
      - 차량 얼굴인식을 통해 등록된 운전자 또는 외부인 식별 및 DB 수집 가능

    - YOLOv8
      <img src="https://github.com/JEMinn/open_CV/assets/160000163/93b36f34-7f5c-489e-bacb-f0458cbc426b"  width="350" height="200"/>
      <img src="https://github.com/JEMinn/open_CV/assets/160000163/c21db574-0dbe-41da-83b0-fe38dee483eb"  width="700" height="100"/>
      - Box(Precision) : 바운딩 박스 정밀도 0.975 / Recall : 재현율 0.952 / mAP50 : 평균 정밀도 50%는 0.995 / mAP50-95 : 50~95%는 0.853
      - 전반적으로 높은 정밀도와 Recall을 보여 "awake" 및 "drowsy" 클래스에 대한 성능이 좋은 것으로 보여짐
      - 단, 일부 객체의 바운딩 박스는 50% 이상의 IoU에서 정밀도가 하락함
      - 이 결과를 토대로 모델 조정 또는 추가적인 훈련 수행 후 해당 성능 향상 가능
     
  4. **웹플라스크 구현**

　
    <img src="https://github.com/JEMinn/open_CV/assets/160000163/783f2728-3c16-4515-be6b-d3c4bea37f1c"  width="500" height="300"/>

　
    - LBPH 모델을 사용한 안면인식 후 등록자에 한하여 차량 잠금해제가 가능하며, LANDMARK 및 YOLOv8 두가지 버전의 모델로 졸음운전 시 사운드 실행을 구현함
    - (졸업운전 관련 모델 서치하였을 때 LANDMARK와 YOLOv8 버전 모두 구현하고 싶어 두가지로 진행하였음)





