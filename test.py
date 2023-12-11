# import librosa
# import librosa.display
# import numpy as np
# import matplotlib.pyplot as plt

# def process_ui(file_path, target_length):
#     mfccs_processed = process_audio(file_path, target_length)

#     # 결과 시각화
#     print("Processed MFCC shape:", mfccs_processed.shape)
#     print("MFCC value")
#     print(mfccs_processed)

# def process_audio(file_path, target_length=100):
#     # 음성 파일 로드
#     y, sr = librosa.load(file_path)

#     # MFCC 추출
#     mfccs = librosa.feature.mfcc(y=y, sr=sr)

#     # 패딩
#     mfccs_padded = librosa.util.fix_length(mfccs, size= target_length, axis=1)

#     return mfccs_padded



# # 사용 예시
# file_path = "./dataset/3-1.wav"
# target_length = 200

# process_ui(file_path, target_length)

# file_path = "./dataset/4-1.wav"

# process_ui(file_path, target_length)

# file_path = "./dataset/5-1.wav"

# process_ui(file_path, target_length)

from flask import Flask, request, jsonify
import os
import time
import random  # random 모듈 임포트
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import librosa


def preprocess_mfcc(audio_file_path, n_mfcc=18, fixed_length=200):
    # Load the audio file
    y, sr = librosa.load(audio_file_path)
    
    # Compute MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    # Pad or truncate MFCCs to the fixed length
    if mfccs.shape[1] < fixed_length:
        mfccs_padded = np.pad(mfccs, ((0, 0), (0, fixed_length - mfccs.shape[1])), mode='constant')
    elif mfccs.shape[1] > fixed_length:
        mfccs_padded = mfccs[:, :fixed_length]
    else:
        mfccs_padded = mfccs
    
    return mfccs_padded

# request.args.get을 사용하여 'record' 파라미터 값 가져오기
# record_value = "1-1-p1.wav"
record_value = "6-10.wav"

    
# 녹음 파일이 있는지 확인
record_path = os.path.join('/Users/jihyeokchoi/Desktop/P-Project/AI/dataset', record_value)
file_exists = os.path.exists(record_path)
print(file_exists)
# 1초 동안 대기
# time.sleep(1)
if file_exists:
    y, sr = librosa.load(record_path)
    mfccs = preprocess_mfcc(record_path)
    mfccs = np.expand_dims(mfccs, axis=0)
    model = load_model('./model/WATERMELON_CNN-4.hdf5')
    result = model.predict(mfccs)
    result = result[0][0]
    print(result)
    
    # record_value가 None이 아니고 파일이 존재하면 0을 반환, 그렇지 않으면 랜덤으로 0 또는 1을 반환
    # result = 0 if record_value is not None and file_exists else random.choice([0, 1])
    
    # 결과를 JSON 형식으로 반환
    # print(result)