import os
import shutil
from sklearn.model_selection import train_test_split

source_dir='datasets/images'
train_dir='datasets/train'
val_dir='datasets/val'
test_dir='datasets/test'

# 폴더 생성 함수
def create_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# 데이터를 분할하고 이동시키는 함수
def split_data(source_dir, train_dir, test_dir, val_dir, test_size=0.1, val_size=0.1):
    # 클래스 별로 처리 (여기서는 'dog'와 'cat')
    for class_name in ['dog', 'cat']:
        source_class_dir = os.path.join(source_dir, class_name)
        train_class_dir = os.path.join(train_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)

        # 디렉토리 생성
        create_dir_if_not_exists(train_class_dir)
        create_dir_if_not_exists(test_class_dir)
        create_dir_if_not_exists(val_class_dir)

        # 파일 리스트 가져오기
        files = [f for f in os.listdir(source_class_dir) if os.path.isfile(os.path.join(source_class_dir, f))]

        # 먼저 test 분할을 위해 train_test_split 사용
        train_files, test_files = train_test_split(files, test_size=test_size + val_size, random_state=42)

        # 나머지 파일들에서 val 분할
        train_files, val_files = train_test_split(train_files, test_size=val_size / (1 - (test_size + val_size)),
                                                  random_state=42)

        # 파일 이동
        for f in test_files:
            shutil.move(os.path.join(source_class_dir, f), os.path.join(test_class_dir, f))
        for f in val_files:
            shutil.move(os.path.join(source_class_dir, f), os.path.join(val_class_dir, f))
        for f in train_files:
            shutil.move(os.path.join(source_class_dir, f), os.path.join(train_class_dir, f))

