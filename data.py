import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping

# 데이터 경로 설정
data_dir = "/Users/leejaeyeong/Desktop/emotionproject/data"

# 이미지 크기 및 배치 크기 설정
img_size = (225, 225)
batch_size = 32

# 데이터 생성기 설정
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.3)

# 데이터 로드 및 전처리
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='training'  # 훈련 데이터
)
 
validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'  # 검증 데이터
)

# 클래스 인덱스 확인
print("클래스 인덱스:", train_generator.class_indices)

# 킥보드와 길의 이미지 수 확인
print("킥보드 이미지 수:", len(train_generator.labels) - sum(train_generator.labels))
print("길 이미지 수:", sum(train_generator.labels))

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(225, 225, 3)))
model.add(MaxPooling2D((2, 2)))

# 은닉층 1
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# 은닉층 2
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# 은닉층 3
model.add(Conv2D(128, (3, 3), activation='sigmoid'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='sigmoid'))
model.add(MaxPooling2D((2, 2)))

# Flatten 층: 3D 출력을 1D로 변환
model.add(Flatten())

# 은닉층 4
model.add(Dense(128, activation='relu'))

# 출력층
model.add(Dense(1, activation='sigmoid'))

# 모델 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early Stopping 설정
early_stopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)

# 모델 훈련
model.fit(train_generator, epochs=300, validation_data=validation_generator, callbacks=[early_stopping])

# 훈련 데이터 정확도 출력
train_accuracy = model.evaluate(train_generator)[1]
print(f'훈련 데이터 정확도: {train_accuracy * 100:.2f}%')

# 검증 데이터 정확도 출력
validation_accuracy = model.evaluate(validation_generator)[1]
print(f'검증 데이터 정확도: {validation_accuracy * 100:.2f}%')
