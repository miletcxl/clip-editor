# 🎬 Clip Editor — 文本驱动的视频自动剪辑原型

这是一个使用 OpenAI CLIP 实现的多模态视频剪辑系统。你只需输入一段文字描述，系统即可从视频素材中找出语义相符的片段并拼接导出。

## 功能

- 使用 CLIP 对视频帧编码
- 基于语义相似度进行片段搜索
- 自动拼接成短视频

## 使用方法

1. 将视频放入 `assets/`
2. 运行 `build_index.py` 建立向量库
3. 使用 `search_video.py` 输入文字
4. 执行 `assemble_video.py` 输出视频

## 依赖安装

requirements.txt
