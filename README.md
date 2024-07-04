### 📌 프로젝트 명 : 차량 얼굴인식 및 졸음운전 방지 서비스 (Facial recognition & drawsy detection Project)
---

 ## ✨ 프로젝트 소개<br>

  ✨ **목적**
  - 졸음운전과 차량 도난 방지를 목표로, 딥러닝을 활용하여 자동차의 불법적인 사용과 졸음 운전으로 인한 사고를 예방하고, 등록된 운전자의 안전 및 차량 사용 통계를 수집하는 기능 추가<br><br>

  🌟 **기능**
  - LBPH 모델로 얼굴 인식을 통한 등록자 구분
  - LANDMARK, YOLOV8 모델을 이용한 졸음 감지
  - HTML&CSS를 이용한 FLASK 웹페이지 제작 <br><br>

  📅 **수행일정**
![image](https://github.com/JEMinn/open_CV/assets/160000163/2421b2a6-4c83-40d9-aa23-671fda58c0ec)

  🛠️ **Flow Chart**<br>
  <img src="https://github.com/JEMinn/open_CV/assets/160000163/53620040-6e85-4dbc-9a70-0f9560c980cd"  width="430" height="500"/><br><br>

---

📖 **주요 모델**
  1. **LBPH 모델 (Local Binary Pattern Histogram)**
     - 얼굴 인식을 통해 등록자를 구분<br><br>
  2. **LANDMARK**
     - 눈 종횡비를 이용하여 졸음 상태를 감지<br><br>
  3. **YOLOv8(You Only Look Once) 모델**
     - 실시간 객체 인식을 통해 졸음 상태를 감지



---

📖 **내용 소개**<br>
  1. **데이터 수집**<br>
    <img src="https://github.com/JEMinn/open_CV/assets/160000163/51f1b888-c25e-4dcc-8646-b2e825917047"  width="350" height="200"/>
  - LBPH와 YOLOv8 모델은 사진을 직접 촬영하여 수집
  - LANDMARK 모델의 경우 눈의 횡종비로 측정하기에 따로 수집이 불필요<br><br>

　
  2. **데이터 전처리**

 　
    <img src="https://github.com/JEMinn/open_CV/assets/160000163/e6cde029-13ab-4102-a15c-d91717c9f0cf"  width="350" height="200"/>

  3. **모델 학습 및 평가**<br>
  - LBPH<br>
      <img src="https://github.com/JEMinn/open_CV/assets/160000163/78457f6c-6f03-4c7f-9be2-77c7bed02c94"  width="300" height="300"/>
      - 차량 얼굴인식을 통해 등록된 운전자 또는 외부인 식별 및 DB 수집 가능
    
  - YOLOv8<br>
      <img src="https://github.com/JEMinn/open_CV/assets/160000163/93b36f34-7f5c-489e-bacb-f0458cbc426b"  width="350" height="200"/>
      <img src="https://github.com/JEMinn/open_CV/assets/160000163/c21db574-0dbe-41da-83b0-fe38dee483eb"  width="700" height="100"/>
      - 높은 정밀도와 Recall로 "awake" 및 "drowsy" 클래스 성능 우수
      - 단, 일부 객체의 바운딩 박스는 50% 이상의 IoU에서 정밀도가 하락함
      - 이 결과를 토대로 모델 조정 또는 추가적인 훈련 수행 후 해당 성능 향상 가능<br><br>
     
  4. **웹플라스크 구현**

　
    <img src="https://github.com/JEMinn/open_CV/assets/160000163/783f2728-3c16-4515-be6b-d3c4bea37f1c"  width="500" height="300"/>

 - LBPH 모델을 사용한 안면인식 후 등록자만 차량 잠금 해제
 - LANDMARK 및 YOLOv8 모델로 졸음운전 시 사운드 실행





