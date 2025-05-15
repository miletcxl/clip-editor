# app.py
import streamlit as st
import os
import clip
import torch
import faiss
import json
import numpy as np
from PIL import Image
from moviepy.editor import VideoFileClip, concatenate_videoclips
import imageio
imageio.plugins.ffmpeg.download()
import streamlit as st
import sys
st.write(f"🐍 当前 Python 版本: {sys.version}")

# 解决 OpenMP 冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 模型加载
device = "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)

st.set_page_config(page_title="AI 视频剪辑助手", layout="centered")
st.title("🎬 文本驱动 AI 自动剪辑")

# 上传视频
st.header("1️⃣ 上传视频素材")

assets_path = "assets"
os.makedirs(assets_path, exist_ok=True)

# 🔄 自动清空之前的素材
for f in os.listdir(assets_path):
    os.remove(os.path.join(assets_path, f))
st.info("🧹 已清空旧视频素材")

# ⬆️ 上传视频文件
uploaded_files = st.file_uploader("选择 MP4 视频", type=["mp4"], accept_multiple_files=True)

# 💾 保存视频文件
for file in uploaded_files:
    save_path = os.path.join(assets_path, file.name)
    with open(save_path, "wb") as f:
        f.write(file.read())
    st.success(f"✅ 成功保存: {file.name}")


# 🧠 检查是否需要重新构建索引
def needs_reindex(assets_dir, meta_path):
    if not os.path.exists(meta_path):
        return True
    with open(meta_path, "r") as f:
        meta = json.load(f)
    indexed_videos = {item['video'] for item in meta}
    current_videos = {f for f in os.listdir(assets_dir) if f.endswith(".mp4")}
    return indexed_videos != current_videos

if needs_reindex(assets_path, "index/meta.json"):
    st.warning("⚠️ 检测到视频素材变更，请点击『🔍 构建索引』重新生成索引。")

# 构建索引按钮
if st.button("🔍 构建索引"):
    index_meta = []
    vectors = []

    for filename in os.listdir(assets_path):
        if not filename.endswith(".mp4"):
            continue
        path = os.path.join(assets_path, filename)

        # 🛡️ 添加 try-except 防止坏文件导致程序崩溃
        try:
            video = VideoFileClip(path)
            duration = int(video.duration)
        except Exception as e:
            st.error(f"⚠️ 无法读取视频文件 {filename}，已跳过。原因：{e}")
            continue

        for t in range(0, duration, 10):
            try:
                frame = video.get_frame(t)
                image = Image.fromarray(frame)
                image_input = preprocess(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    image_features = model.encode_image(image_input).cpu().numpy()[0]

                vectors.append(image_features)
                index_meta.append({"video": filename, "time": t})
            except Exception as e:
                st.warning(f"⚠️ 跳过 {filename} 的第 {t}s 帧：{e}")
                continue

    if not vectors:
        st.error("❌ 所有视频都无法处理或没有有效帧，索引构建失败。")
    else:
        os.makedirs("index", exist_ok=True)
        vectors = np.array(vectors).astype("float32")
        faiss_index = faiss.IndexFlatL2(vectors.shape[1])
        faiss_index.add(vectors)
        faiss.write_index(faiss_index, "index/clip.index")
        with open("index/meta.json", "w") as f:
            json.dump(index_meta, f)
        st.success(f"✅ 索引构建完成，共索引 {len(index_meta)} 帧")


# 输入搜索文本
st.header("2️⃣ 输入文字，搜索片段")
query = st.text_input("请输入你想搜索的视频描述（例如：a smiling woman）")

# 生成剪辑
if st.button("✂️ 生成剪辑视频"):
    if not os.path.exists("index/clip.index"):
        st.error("❌ 请先构建索引")
    else:
        index = faiss.read_index("index/clip.index")
        with open("index/meta.json", "r") as f:
            meta = json.load(f)

        with torch.no_grad():
            text_features = model.encode_text(clip.tokenize([query]).to(device)).cpu().numpy()

        D, I = index.search(text_features, 3)
        selected = [meta[i] for i in I[0]]

        with open("index/result.json", "w") as f:
            json.dump(selected, f)

        clips = []
        for seg in selected:
            video_path = os.path.join(assets_path, seg["video"])
            t = seg["time"]
            clip_video = VideoFileClip(video_path).subclip(t, min(t + 5, VideoFileClip(video_path).duration))
            clips.append(clip_video)

        final = concatenate_videoclips(clips)
        os.makedirs("output", exist_ok=True)
        output_path = "output/final_video.mp4"
        if os.path.exists(output_path):
            os.remove(output_path)
        final.write_videofile(output_path)

        st.success("✅ 剪辑完成")
        st.video(output_path)
