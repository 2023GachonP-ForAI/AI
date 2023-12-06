'''
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Replace 'audio_file_path' with the path to your audio file
audio_file_path = './dataset/4-1.wav'
y, sr = librosa.load(audio_file_path)

# Compute the Mel spectrogram
# mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)

# Display the Mel spectrogram
# plt.figure(figsize=(10, 4))
# plt.subplot(2, 1, 1)  # Create a subplot with 2 rows and 1 column, and select the first plot
# librosa.display.specshow(librosa.power_to_db(mel_spectrogram, ref=np.max), y_axis='mel', x_axis='time')
# plt.colorbar(format='%+2.0f dB')
# plt.title('Mel Spectrogram')

# Compute MFCCs from the Mel spectrogram
# S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=18)
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=18)
# mfccs = librosa.feature.mfcc(S=librosa.power_to_db(S))

print(mfccs.shape)
# Display the MFCCs
# plt.subplot(2, 1, 2)  # Select the second plot
# librosa.display.specshow(mfccs, x_axis='time', cmap='viridis')  # You can choose a colormap of your choice
# plt.colorbar()
# plt.title('MFCCs')

# plt.tight_layout()  # Ensure plots do not overlap
# plt.show()

time = librosa.times_like(mfccs)

for i in range(mfccs.shape[0]):
    plt.plot(time, mfccs[i], label=f'MFCC {i + 1}')

plt.xlabel('Time (s)')
plt.ylabel('MFCC Values')
plt.title('MFCCs')
plt.legend()
plt.grid()
plt.show()
'''
"""
import librosa
import matplotlib.pyplot as plt

# List of audio file paths (replace with your file paths)
audio_file_paths = ['./dataset/4-1.wav', './dataset/4-2.wav', './dataset/4-3.wav']

# Create a subplot for each MFCC coefficient
plt.figure(figsize=(15, 5))

for j in range(18):  # Assuming you have 18 MFCC coefficients
    plt.subplot(3, 6, j + 1)  # 3 rows, 6 columns of subplots
    plt.title(f'MFCC {j + 1}')

    for i, audio_path in enumerate(audio_file_paths):
        y, sr = librosa.load(audio_path)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=18)
        time = librosa.times_like(mfccs)

        plt.plot(time, mfccs[j], label=f'Audio {i + 1}')

plt.xlabel('Time (s)')
plt.ylabel('MFCC Values')
plt.tight_layout()  # Ensure plots do not overlap
plt.show()
"""

import librosa
import matplotlib.pyplot as plt
import numpy as np

# Replace 'audio_file_path' with the path to your audio file
audio_file_path = './dataset/4-1.wav'

# Load the audio file
y, sr = librosa.load(audio_file_path)

# Create a time axis
time = librosa.times_like(y)

# Plot the audio waveform
plt.figure(figsize=(10, 4))
plt.plot(time, y, label='Audio Waveform', linewidth=0.7)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Audio Waveform')
plt.legend()
plt.grid()
plt.show()

mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=18)
time = librosa.times_like(mfccs)

# Display the MFCCs as a graph
plt.figure(figsize=(10, 6))
librosa.display.specshow(mfccs, x_axis='time', sr=sr, cmap='viridis')
plt.colorbar(format='%+2.0f dB')
plt.title('MFCCs')
plt.xlabel('Time (s)')
plt.ylabel('MFCC Coefficients')
plt.show()

# for i in range(mfccs.shape[0]):
#     plt.plot(time, mfccs[i], label=f'MFCC {i + 1}')
# plt.xlabel('Time (s)')
# plt.ylabel('MFCC Values')
# plt.title('MFCCs')
# plt.legend()
# plt.grid()
# plt.show()