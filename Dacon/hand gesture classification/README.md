## **1. 대회설명**
* 대회 기간: 3월 7일 ~ 3월18일 18:00 PM 
* 대회명: 손동작 분류 경진대회

## **2. 과제개요**
데이콘 Basic| 정형 | Accuracy

* 문제정의<br>
손에 부착된 센서의 데이터를 통해 총 4개의 종류의 손동작을 분류

## **3. 사용 스택**
* Google Colab, Python, Pandas, Numpy, Scikit-learn, Keras, Matplotlib
  
## **4. 프로젝트 내용 요약**
1. EDA를 통하여 데이터를 살펴봄.
    - train data에 비해 test data가 4배 정도나 되는데, 성능이 워낙 잘나와서 train data를 줄인 것으로 확인하였고, sensor가 2의 배수인 32개인 것을 보면, 2D이미지의 좌표로 생각을 함.
        
        >> 이미지로 생각하여 **8*4 형태로 데이터**를 바꿔줌 
        
    - target이 0,123인 Multiclass이므로 **one-hot 인코딩**을 취해줌.
    - 이상치가 많이 존재하여 **RobustScaler**를 사용하여 이상치를 줄여줌.
        
        ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/66d8eebc-2d13-4eca-8bc5-0928de3a4446/Untitled.png)
        
        ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d8936dad-7211-4800-b13b-c410201a82c5/Untitled.png)
        
2. Model을 선택하여 최적의 Model 선택
    
    >> 층을 쌓고, 코드를 간편화하기 위해 **Keras를 사용**
    
    - LightGBM : 0.8
    - Cnn(conv3개) : 0.86 >> 층이 작은 Cnn일 때가 제일 성능이 좋음.
    - Dcnn (conv6개) : 0.84
    - LSTM : 0.85
    - GRU : 0.82
3. Cnn model HyperParameter 튜닝
    - Learning_rate를 0.4 → 0.004로 조정 (0.885) >> 높은 성능 향상
    - 층은 깊게하고 파라미터 수는 줄이게 하려고 1*1 conv를  추가(0.89)
        
        >> 성능 향상
        
    - activation 함수를 relu로 설정하고, Maxpooling 제거 (0.895) >> 성능 향상
    - K-fold를 10 → 15로 설정 (0.898~0.899)>> 성능 향상
    - Flatten 추가 (0.899) >> 성능 조금 향상
    - Dropout 설정 → 성능 저하
    - 모델 코드
        
        ```python
        ohe = OneHotEncoder(sparse = False)
        skf = StratifiedKFold(n_splits = 15, random_state = 42, shuffle = True)
        es = EarlyStopping(monitor = 'val_acc', patience = 5, mode = 'max', verbose = 0)
        ```
        
        ```python
        from sklearn.utils import shuffle
        
        cnn_acc = []
        cnn_pred = np.zeros((target.shape[0], 4))
        for i, (tr_idx, val_idx) in enumerate(skf.split(X, train.target)) :
            print(f'{i + 1} Fold Training.....')
            tr_x, tr_y = X[tr_idx], y[tr_idx]
            val_x, val_y = X[val_idx], y[val_idx]
            
            ### CNN 모델
            cnn = Sequential()
            cnn.add(Conv2D(32, (2, 2), padding = "same", activation = 'relu', input_shape = (8, 4, 1)))
            cnn.add(BatchNormalization())
            cnn.add(Conv2D(16, (1, 1), padding = "same", activation = 'relu'))
            cnn.add(BatchNormalization())
            cnn.add(Conv2D(32, (4, 4), padding = "same", activation = 'relu'))
            cnn.add(BatchNormalization())
            cnn.add(GlobalAveragePooling2D())
            cnn.add(Flatten())
            cnn.add(Dense(32, activation = 'relu'))
            # cnn.add(Dropout(0.25))
            cnn.add(Dense(4, activation = 'softmax'))
        
            ### ModelCheckPoint Fold마다 갱신
            mc = ModelCheckpoint(f'model_{i + 1}.h5', save_best_only = True, monitor = 'val_acc', mode = 'max', verbose = 0)
            
            ### 모델 compile
            cnn.compile(optimizer = Adam(learning_rate = 0.004), loss = 'categorical_crossentropy', metrics = ['acc'])
        
            cnn.fit(tr_x, tr_y, validation_data = (val_x, val_y), epochs = 100, batch_size = 32, callbacks = [es, mc], verbose = 0)
        
            ### 최고 성능 기록 모델 Load
            best = load_model(f'model_{i + 1}.h5')
            ### validation predict
            val_pred = best.predict(val_x)
            ### 확률값 중 최대값을 클래스로 매칭
            val_cls = np.argmax(val_pred, axis = 1)
            ### Fold별 정확도 산출
            fold_cnn_acc = accuracy_score(np.argmax(val_y, axis = 1), val_cls)
            cnn_acc.append(fold_cnn_acc)
            print(f'{i + 1} Fold ACC of CNN = {fold_cnn_acc}\n')
        
            ### Fold별 test 데이터에 대한 예측값 생성 및 앙상블
            fold_pred = best.predict(target) / skf.n_splits
            cnn_pred += fold_pred
        ```
        
4. 앙상블 Voting
    - 높은 점수가 나온 csv파일을 불러와 과반수 원리를 이용하여 Voting해줌
    
    → K-fold 15 설정 모델(0.899), Flatten 추가한 모델(0.899), epoch수 다르게한 것(0.898) 이용 >> 0.901로 성능 향상

## **5. 성능 향상 부분**
* 이상치가 많아서 이상치를 조정해 주기 위해 RobustScaler를 사용하게 되었습니다.
* 기존 CNN 코드에서 층 수를 줄였습니다.
* 활성화함수는 Relu로 설정하고, Maxpooling을 제거하고, Flatten을 추가해주었습니다. >> cnn size가 너무 작아서 MaxPooling을 제거
* learing_rate에 영향이 큰다는 것을 발견하여서 0.004가 제일 성능이 좋아서 조절해주었습니다.
* 층수를 늘린 DCNN, LSTM, GRU도 사용하여 성능을 비교해보았습니다. >> 센서 데이터를 시계열로 고려해서 LSTM을 썼더니 CNN과 비슷한 성능을 가지긴 하였습니다.
* 과반수 원리를 이용한 Voting을 사용해보았습니다 >> sklearn 라이브러리를 안쓰고 따로 과반수원리 코드를 짜서 사용했습니다.

## **6. 결과**
### public 5위 0.90194, private 8위 0.89811


### **주체/주관**
데이콘
