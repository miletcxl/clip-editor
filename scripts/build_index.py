# scripts/build_index.py
import os
import clip
import torch
import faiss
import json
import numpy as np
from PIL import Image
from moviepy.editor import VideoFileClip, concatenate_videoclips


device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

video_dir = "assets"
index_meta = []
vectors = []

for filename in os.listdir(video_dir):
    if not filename.endswith(".mp4"):
        continue
    video_path = os.path.join(video_dir, filename)
    video = VideoFileClip(video_path)

    duration = int(video.duration)
    for t in range(0, duration, 10):  # 每10秒截一帧
        frame = video.get_frame(t)
        image = Image.fromarray(frame)
        image_input = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_input).cpu().numpy()[0]

        vectors.append(image_features)
        index_meta.append({
            "video": filename,
            "time": t
        })

# 保存索引
os.makedirs("index", exist_ok=True)
vectors = np.array(vectors).astype("float32")
faiss_index = faiss.IndexFlatL2(vectors.shape[1])
faiss_index.add(vectors)
faiss.write_index(faiss_index, "index/clip.index")

with open("index/meta.json", "w") as f:
    json.dump(index_meta, f)

print("✅ 索引构建完成，共处理帧数:", len(index_meta))

