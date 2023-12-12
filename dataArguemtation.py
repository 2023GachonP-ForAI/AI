import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import os
#windows ver입니다. 리눅스에서 돌릴거면 파일 수정이 조금 필요해요.

# 주파수 변형 함수
def pitch_shift(audio, sr, pitch_factor):
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_factor)

# 시간 쉬프트 함수
def time_shift(audio, shift_factor):
    return np.roll(audio, shift_factor)


# 폴더 경로 설정
folder_path = "./dataset"  # 폴더 경로를 적절히 변경해주세요.

# 폴더 내부의 모든 파일 목록 얻기
all_files = os.listdir('dataset')
print(all_files)
# .wav 파일만 선택하여 경로 저장
wav_files = [os.path.join(folder_path, file) for file in all_files if file.endswith(".wav")]

for wav_file in wav_files:
    # 예시 음성 데이터 불러오기
    file_path = wav_file
    audio, sr = librosa.load(file_path, sr=None)

    # 시각화를 위한 코드


    # 원본 음성 데이터 시각화 (파랑색)
    
    plt.figure(figsize=(16, 10))
    
    '''
    plt.subplot(7, 1, 1)
    librosa.display.waveshow(audio, sr=sr, color='blue', alpha=0.8, label='Original Audio')
    plt.title('Original Audio')
    plt.legend()
    '''
    

   
    # 주파수 변형 1
    pitch_factor = 2
    pitch_shifted_audio = pitch_shift(audio, sr, pitch_factor)
    pitch_shifted_path = file_path[:-4]+"-p1.wav" # 변조된 파일 저장 경로 설정
    sf.write(pitch_shifted_path, pitch_shifted_audio, sr)
    # 주파수 변조된 음성 데이터 시각화 (초록색)
   
    
    '''
    plt.subplot(7, 1, 2)
    librosa.display.waveshow(audio, sr=sr, color='blue', alpha=0.8,label='Original Audio')
    librosa.display.waveshow(pitch_shifted_audio, sr=sr, color='green', alpha=0.8, label='Pitch_shifted')
    plt.title('Pitch Shifted 1 Audio')
    plt.legend()
    '''
    
   

    # 주파수 변형 2
    pitch_factor = -2
    pitch_shifted_audio = pitch_shift(audio, sr, pitch_factor)
    pitch_shifted_path = file_path[:-4]+"-p2.wav" # 변조된 파일 저장 경로 설정
    sf.write(pitch_shifted_path, pitch_shifted_audio, sr)
    # 주파수 변조된 음성 데이터 시각화 (초록색)
   
    
    '''
    plt.subplot(7, 1, 3)
    librosa.display.waveshow(audio, sr=sr, color='blue', alpha=0.8,label='Original Audio')
    librosa.display.waveshow(pitch_shifted_audio, sr=sr, color='green', alpha=0.8, label='Pitch_shifted')
    plt.title('Pitch Shifted 2 Audio')
    plt.legend()
    '''
    
   

    # 시간 쉬프트 1
    shift_factor = 500
    time_shifted_audio = time_shift(audio, shift_factor)
    time_shifted_path = file_path[:-4]+"-t1.wav" # 주파수 변조된 파일 추출
    sf.write(time_shifted_path, time_shifted_audio, sr)
    # 시간 변조된 음성 데이터 시각화 (빨강색)
   
    
    '''
    plt.subplot(7, 1, 4)
    librosa.display.waveshow(audio, sr=sr, color='blue', alpha=0.8, label='Original Audio')
    librosa.display.waveshow(time_shifted_audio, sr=sr, color='red', alpha=0.8, label='Time_shifted')
    plt.title('Time Shifted 1 Audio')
    plt.legend()
    '''
    
   

    # 시간 쉬프트 2
    shift_factor = 1000
    time_shifted_audio = time_shift(audio, shift_factor)
    time_shifted_path = file_path[:-4]+"-t2.wav" # 주파수 변조된 파일 추출
    sf.write(time_shifted_path, time_shifted_audio, sr)
    # 시간 변조된 음성 데이터 시각화 (빨강색)
   
    
    '''
    plt.subplot(7, 1, 5)
    librosa.display.waveshow(audio, sr=sr, color='blue', alpha=0.8, label='Original Audio')
    librosa.display.waveshow(time_shifted_audio, sr=sr, color='red', alpha=0.8, label='Time_shifted')
    plt.title('Time Shifted 2 Audio')
    plt.legend()
    '''
    
   

    # 시간 쉬프트 3
    shift_factor = 1500
    time_shifted_audio = time_shift(audio, shift_factor)
    time_shifted_path = file_path[:-4]+"-t3.wav" # 주파수 변조된 파일 추출
    sf.write(time_shifted_path, time_shifted_audio, sr)
    # 시간 변조된 음성 데이터 시각화 (빨강색)

    


    '''
    plt.subplot(7, 1, 6)
    librosa.display.waveshow(audio, sr=sr, color='blue', alpha=0.8, label='Original Audio')
    librosa.display.waveshow(time_shifted_audio, sr=sr, color='red', alpha=0.8, label='Time_shifted')
    plt.title('Time Shifted 3 Audio')
    plt.legend()
    '''
    

    # 스케일링 1
    scaling_factor = 2.0  #
    scaling_audio = audio * scaling_factor
    scaling_path = file_path[:-4]+"-s1.wav"
    sf.write(scaling_path, scaling_audio, sr)

    
    '''
    plt.subplot(7, 1, 7)
    librosa.display.waveshow(audio, sr=sr, color='blue', alpha=0.8, label='Original Audio')
    librosa.display.waveshow(scaling_audio, sr=sr, color='grey', alpha=0.8, label='Time_shifted')
    plt.title('Time Shifted 3 Audio')
    plt.legend()
    plt.tight_layout()
    plt.show()
    '''
    
   




