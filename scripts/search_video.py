# scripts/search_video.py
import clip
import torch
import faiss
import json
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


device = "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)

# 输入查询
query = input("请输入你想找的场景描述：")
with torch.no_grad():
    text_features = model.encode_text(clip.tokenize([query]).to(device)).cpu().numpy()

# 加载索引
index = faiss.read_index("index/clip.index")
with open("index/meta.json", "r") as f:
    meta = json.load(f)

# 搜索
top_k = 3
D, I = index.search(text_features, top_k)

# 显示结果
selected = []
print("✅ 匹配到的片段：")
for idx in I[0]:
    item = meta[idx]
    print(f"{item['video']} @ {item['time']} 秒")
    selected.append(item)

# 保存结果
with open("index/result.json", "w") as f:
    json.dump(selected, f)

