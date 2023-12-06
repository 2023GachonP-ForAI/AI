import librosa
import numpy as np
import matplotlib.pyplot as plt
import random
import soundfile as sf  # soundfile 모듈 사용

def change_pitch(y, sr, semitone_steps):
    # 주파수 변형을 통한 음성의 피치 변경
    y_changed_pitch = librosa.effects.pitch_shift(y, n_steps=semitone_steps, sr=sr)
    return y_changed_pitch

# 음성 파일 로드
file_path = "수박3-1.wav"
y, sr = librosa.load(file_path)

# 주파수 변형을 통한 음성의 피치 변경
semitone_steps = random.uniform(-2, 2)
y_changed_pitch = change_pitch(y, sr, semitone_steps)

# 결과 시각화 (주파수 변형만 시각화)
plt.figure(figsize=(12, 6))

# 원본 음성
plt.subplot(2, 1, 1)
plt.plot(np.arange(len(y)) / sr, y, color='b')
plt.title('Original Audio')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

# 피치 변경된 음성
plt.subplot(2, 1, 2)
plt.plot(np.arange(len(y_changed_pitch)) / sr, y_changed_pitch, color='r')
plt.title('Pitch Changed Audio')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()

# 피치 변경된 음성을 파일로 저장 (soundfile 사용)
output_wav_path = "output_pitch_changed.wav"
sf.write(output_wav_path, y_changed_pitch, sr)
print(f"Pitch changed audio saved to: {output_wav_path}")
