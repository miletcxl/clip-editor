# scripts/extract_frames.py
import os
from moviepy.editor import VideoFileClip, concatenate_videoclips
from PIL import Image

video_dir = "assets"
output_dir = "frames"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(video_dir):
    if not filename.endswith(".mp4"):
        continue
    video_path = os.path.join(video_dir, filename)
    video = VideoFileClip(video_path)

    duration = int(video.duration)
    for t in range(0, duration, 10):
        frame = video.get_frame(t)
        img = Image.fromarray(frame)
        img.save(f"{output_dir}/{filename}_{t}s.jpg")

print("✅ 完成帧提取")

