import subprocess
import os
# 1. 도커 이미지 로드 (이미 수행했다면 생략 가능)
# load_command = "docker load -i tta_01_26_042.tar.gz"
# subprocess.run(load_command, shell=True)

video_path='/videos'
preprocess_video_path='/preprocess_video'
# 2. 비디오 전처리 실행
# "비디오_폴더"와 "전처리_폴더" 경로를 실제 경로로 대체하세요.
preprocess_command = f"docker run -v /videos:/VIDEOS -v /preprocess_video:/PREPROCESSED tta_01_26_042 python scripts/preprocess.py -d /VIDEOS -o /PREPROCESSED/test.h5"
subprocess.run(preprocess_command, shell=True)

# 3. 비디오 요약 모델 실행
# 여기서도 "전처리_폴더" 경로를 실제 경로로 대체하세요.
summary_command = f"docker run --shm-size=32G -v {preprocess_video_path}:/PREPROCESSED tta_01_26_042 python3 scripts/training/summary/run.py -d labels -v /PREPROCESSED/test.h5 -s splits/1.json -w summary.ckpt"
subprocess.run(summary_command, shell=True)

# 결과 파일 처리는 모델 실행 후 생성된 파일을 기반으로 진행합니다.

'docker run -v /videos:/VIDEOS /preprocess_video:/PREPROCESSED tta_01_26_042 \python scripts/preprocess.py -d /VIDEOS -o /PREPROCESSED/test.h5'