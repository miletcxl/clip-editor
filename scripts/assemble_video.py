# scripts/assemble_video.py
import json
import os
from moviepy.editor import VideoFileClip, concatenate_videoclips


with open("index/result.json") as f:
    segments = json.load(f)

clips = []
for seg in segments:
    video_path = os.path.join("assets", seg["video"])
    t = seg["time"]
    clip = VideoFileClip(video_path).subclip(t, min(t + 5, VideoFileClip(video_path).duration))
    clips.append(clip)

final = concatenate_videoclips(clips)
os.makedirs("output", exist_ok=True)
final.write_videofile("output/final_video.mp4")
print("✅ 视频拼接完成，保存至 output/final_video.mp4")
