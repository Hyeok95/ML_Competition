## **1. 과제 개요**
* 평가지표 : MAE(Mean_Squared_Error
* 반도체 설비 제어 인자 조절에 따른 Target값을 예측하는 프로젝트

## **2. 사용 스택**
* Google Colab, Python, Pandas, Numpy, Matplotlib, Scikit-learn, Keras
  
## **4. 프로젝트 내용 요약**
1. 데이터 결측치와 이상치를 확인하고 분포를 확인하여 매우 적은 수의 데이터를 제거해줌.
2. X0~X5까지의 컬럼들이 5,6가지의 값들로 존재해서 원-핫 인코딩을 해주고, 6개의 컬럼을 32개의 컬럼으로 만들어줌
3. target값이 여러 값으로 존재하므로, 머신러닝 회귀모델과 Keras를 이용하여 살펴봄.

    → Keras의 Dense를 4개의 층으로 구성하여 사용하였을 때 가장 좋은 성능을 나타냄
    
4. 모델의 활성화 함수와 학습률, 층수를 조정하여 튜닝작업을 함.
  * 모델 코드
        ```python
        ohe = OneHotEncoder(sparse = False)
        skf = StratifiedKFold(n_splits = 15, random_state = 42, shuffle = True)
        es = EarlyStopping(monitor = 'val_acc', patience = 5, mode = 'max', verbose = 0)
        ```
        
```python
from tensorflow import keras

weight_decay = 1e-5
learning_rate = 0.0031

model = keras.Sequential([
    keras.layers.Dense(128, activation='tanh', input_shape=(X_train.shape[1],), kernel_regularizer=keras.regularizers.l2(weight_decay)),
    # keras.layers.BatchNormalization(),
    keras.layers.Dense(64, activation='tanh', kernel_regularizer=keras.regularizers.l2(weight_decay)),
    # keras.layers.BatchNormalization(),
    keras.layers.Dense(32, activation='tanh' ,kernel_regularizer=keras.regularizers.l2(weight_decay)),
    # keras.layers.BatchNormalization(),
    # keras.layers.Dense(16, activation='tanh' ,kernel_regularizer=keras.regularizers.l2(weight_decay)),
    keras.layers.Dropout(0),
    keras.layers.Dense(10)
])
```
        
## **4. 알게 된 점**
* Feature의 값이 5개의 특징적인 값들로 존재하다보니 원-핫 인코딩을 하고 모델링을 했을 때 성능이 좋아지는 것을 알게됨.
* 분포 수가 적은 이상치에도 성능 차이를 나타냄.
* Batch normalization을 사용할 때 오히려 더 loss가 높은 지점에서 수렴을 하여서 강제로 분포를 바꾸는 것은 학습이 잘되게 하는 것도 있지만 일종의 정보 소실로도 이어지고, 득보다 실이 더 큰 data가 주어진 상황이라고 생각함.
## **5. 결과**
### 전체 15명 중 5등


### **주체/주관**
마인즈앤 
