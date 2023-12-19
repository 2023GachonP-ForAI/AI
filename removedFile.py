import os
# 폴더 경로 설정
# 폴더 내부의 모든 파일 목록 얻기
all_files = os.listdir('dataset')
print(all_files)

for wav_file in all_files:
	if "t" in wav_file or "p" in wav_file or "s" in wav_file:
		if ".ipynb_checkpoints" in wav_file:
			continue
		print("delete: ", wav_file)
		os.unlink('dataset/' + wav_file)