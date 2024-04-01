from pytube import YouTube
import os
os.environ["IMAGEIO_FFMPEG_EXE"] = "/opt/homebrew/bin/ffmpeg"
from moviepy.editor import VideoFileClip, concatenate_videoclips
from model.dataset_convert import VideoFramesDataset
from torchvision.transforms import Compose, Resize, ToTensor
import cv2
from transformers import ViTFeatureExtractor
from torch.utils.data import DataLoader, Dataset
from model.summarization import SummaryModel

import  torch

def download_video(url, path='./videos'):
    yt = YouTube(url)
    # 가장 높은 해상도의 스트림 선택
    ys = yt.streams.get_highest_resolution()
    # 영상 다운로드
    ys.download(path)
    print(f"다운로드 완료: {ys.default_filename}")
    return os.path.join(path, ys.default_filename)

def extract_video_features(extractor, video_file, sample_every=1):
    vc = cv2.VideoCapture(video_file)
    frames = []
    while vc.isOpened():
        success, frame = vc.read()
        if not success:
            break
        frames.append(frame)
    vc.release()

    features = extractor(images=frames, return_tensors="pt")
    return features["pixel_values"]




def load_model(ckpt_path, device='cpu'):
    model = SummaryModel.load_from_checkpoint(ckpt_path).to(device)
    model.eval()
    return model


def summarize_video(video_path, model, extractor, threshold=0.6):
    transform = Compose([
        Resize((224, 224)),  # ViT 입력 크기에 맞춰 조정
        ToTensor(),
    ])

    dataset = VideoFramesDataset(video_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    preds = []
    for batch in dataloader:
        features = extractor(images=batch, return_tensors="pt")["pixel_values"].to('cpu')
        with torch.no_grad():
            out = model(features).squeeze(1)
            pred = (torch.sigmoid(out) > threshold).nonzero(as_tuple=True)[0]
            preds.extend(pred + len(preds) * dataloader.batch_size)

    # 요약 비디오 생성
    clip = VideoFileClip(video_path)
    clips = [clip.subclip(max(frame.item() / clip.fps - 1, 0), min(frame.item() / clip.fps + 1, clip.duration)) for
             frame in preds]
    final_clip = concatenate_videoclips(clips)
    summary_path = "summary_video.mp4"
    final_clip.write_videofile(summary_path, codec="libx264", audio_codec="aac")
    print(f'Summary video saved to {summary_path}')

model_path = 'model/summary.ckpt'
youtube_url='https://www.youtube.com/watch?v=AkKgXCO6mBA&ab_channel=Popcorn%26CokeReview'
# 영상 다운로드
downloaded_video_path = download_video(youtube_url)

# 모델 및 특징 추출기 로딩
device = 'cpu' # 또는 'cuda' if GPU 사용
model = load_model(model_path, device)
extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

# 영상 요약 및 저장
summarize_video(downloaded_video_path, model, extractor)