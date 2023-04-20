# Personalized-stock-portfolio-model

## About model

* LSTM 모델 사용

#### 활용 데이터
1. **average_model** – 주가 + 네이버 종목토론방에서의 각 제목 텍스트들을 **따로 KR-Finbert**를 이용해 감성분석한 후, 날짜별로 평균 낸 데이터
2. **combined_model** – 주가 + 네이버 종목토론방에서 전체 텍스트를 **날짜별로 합친 후** 감성분석을 진행
3. **no sentiment** – 주가

|                | MSE | MAE | RMSE |
|:--------------:|:---:|:---:|:----:|
| **average_model**  | **754903.7** | **686.77954**  |   **868.8519**   |
| combined_model |   806165.1  |  711.77496   |   897.867   |
| no sentiment   |  816364.75   |   720.88043  |   903.52905   |
