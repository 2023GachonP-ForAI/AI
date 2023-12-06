import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def process_ui(file_path, target_length):
    mfccs_processed = process_audio(file_path, target_length)

    # 결과 시각화
    print("Processed MFCC shape:", mfccs_processed.shape)
    print("MFCC value")
    print(mfccs_processed)

def process_audio(file_path, target_length=100):
    # 음성 파일 로드
    y, sr = librosa.load(file_path)

    # MFCC 추출
    mfccs = librosa.feature.mfcc(y=y, sr=sr)

    # 패딩
    mfccs_padded = librosa.util.fix_length(mfccs, size= target_length, axis=1)

    return mfccs_padded



# 사용 예시
file_path = "수박3-1.wav"
target_length = 200

process_ui(file_path, target_length)

file_path = "수박4-1.wav"

process_ui(file_path, target_length)

file_path = "수박5-1.wav"

process_ui(file_path, target_length)