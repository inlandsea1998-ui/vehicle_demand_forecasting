# 🚗 차량 수요 예측 & 재고 최적화 (Vehicle Demand Forecasting & Inventory Optimization)

## 📌 프로젝트 개요

자동차 시장에서 월별 판매량의 정확한 예측은 생산 계획, 재고 관리, 딜러 배분에 직결됩니다.
본 프로젝트는 LSTM 딥러닝 모델을 활용해 월별 차량 수요를 예측하고, 예측 결과를 기반으로 적정 재고 수량을 최적화하는 파이프라인을 구현합니다.

> **핵심 목표**: 재고 부족(기회 손실)과 과잉 재고(보관 비용) 사이의 최적 균형점 도출

---

## 🗂️ 프로젝트 구조

```
vehicle-demand-forecasting/
│
├── vehicle_demand_forecasting.ipynb   # 메인 노트북
├── README.md
└── output/
    ├── eda_sales.png                  # EDA 시각화 결과
    └── lstm_results.png               # 모델 예측 결과 시각화
```

---

## 🔧 Tech Stack

| 분류 | 라이브러리 |
|------|-----------|
| 딥러닝 | `PyTorch` |
| 데이터 처리 | `Pandas`, `NumPy` |
| 전처리 / 평가 | `Scikit-learn` |
| 최적화 | `SciPy` |
| 시각화 | `Matplotlib`, `Seaborn` |

---

## 📊 파이프라인 단계

### Step 0. 라이브러리 설치 및 임포트
PyTorch, Pandas, Scikit-learn 등 필요 패키지를 설치하고, 재현성을 위해 랜덤 시드를 고정합니다.

### Step 1. 데이터 수집 & 전처리

**옵션 A (기본 — 바로 실행 가능)**: 현실적인 차량 판매 시뮬레이션 데이터 생성
- 2018년 ~ 2023년 (72개월) 월별 판매 데이터
- 트렌드 + 계절성 + 코로나 영향(2020) + 노이즈를 결합한 현실적인 데이터 구성

**옵션 B**: Kaggle [Car Sales Report](https://www.kaggle.com/datasets/gauthamp10/car-sales-report) 실제 데이터 사용

피처 엔지니어링:
- 월(month), 분기(quarter)
- 1개월 / 3개월 전 판매량 (lag feature)
- 3개월 이동평균 (rolling mean)
- Min-Max 정규화 후 12개월 시퀀스 생성 → Train/Test = 80:20 분리

### Step 2. LSTM 모델 설계 & 학습

```
입력: 12개월 시퀀스 → LSTM → Dropout → Fully Connected → 다음 달 판매량 예측
```

- **Hidden Size**: 64
- **Layers**: 2
- **Dropout**: 0.2
- **Optimizer**: Adam (lr=0.001)
- **Loss**: MSELoss
- **Epochs**: 100

### Step 3. 모델 평가 & 결과 시각화

| 지표 | 설명 |
|------|------|
| RMSE | 예측 오차의 크기 (대 단위) |
| MAE | 평균 절대 오차 |
| MAPE | 퍼센트 기준 오차율 |

예측 결과와 학습 손실 곡선을 시각화하여 저장합니다.

### Step 4. 하이퍼파라미터 최적화 실험

4가지 조합을 비교 실험합니다:

| hidden_size | num_layers | learning_rate |
|-------------|------------|---------------|
| 32 | 1 | 0.001 |
| 64 | 2 | 0.001 (기본 모델) |
| 128 | 2 | 0.001 |
| 64 | 3 | 0.0005 |

RMSE 기준으로 최적 모델을 자동 선정합니다.

### Step 5. 재고 최적화 (SciPy)

예측된 수요를 기반으로 `scipy.optimize.minimize`로 총 재고 비용을 최소화합니다.

```
총 비용 = 보관 비용 (과잉 재고) + 기회 손실 비용 (재고 부족) + 고정 주문 비용
```

| 파라미터 | 값 |
|---------|-----|
| 보관 비용 | 30만원 / 대 / 월 |
| 기회 손실 비용 | 200만원 / 대 |
| 고정 주문 비용 | 500만원 |

최종 권장 주문량 = 최적 기본 주문량 + 안전 재고 (예측 오차 표준편차 × 1.5)

### Step 6. 최종 요약 & 결론

모델 성능, 최적 주문량, 비즈니스 인사이트를 종합 출력합니다.

---

## ▶️ 실행 방법

1. **Google Colab에서 열기** (권장)
   - 노트북 파일을 Colab에 업로드 후 **런타임 → 모두 실행** (`Ctrl+F9`)

2. **로컬 환경 실행**
   ```bash
   pip install torch pandas numpy scikit-learn matplotlib seaborn scipy
   jupyter notebook vehicle_demand_forecasting.ipynb
   ```

> ⚠️ **주의**: 셀은 반드시 위에서 아래로 순서대로 실행해야 합니다. `train_dl`, `model`, `scaler` 등의 변수가 이전 셀에서 정의되므로, 셀을 건너뛰면 `NameError`가 발생할 수 있습니다.

---

## 💡 한계 및 개선 방향

- 외부 변수 추가 필요 (경제지표, 금리, 신모델 출시 일정 등)
- 차종별 세분화 모델로 확장 가능
- CPLEX / Gurobi 도입 시 더 정교한 최적화 가능
- Transformer 계열 모델(Informer, PatchTST 등)과의 성능 비교 실험

---

## 📝 참고 데이터

- 시뮬레이션 데이터: 트렌드 + 계절성 + 코로나 영향 + 노이즈로 생성
- 실제 데이터: [Kaggle - Car Sales Report](https://www.kaggle.com/datasets/gauthamp10/car-sales-report)
