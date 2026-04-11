# -*- coding: utf-8 -*-
"""
下载 paraphrase-multilingual-MiniLM-L12-v2 模型到本地
用于离线或网络不稳定环境
"""

import os
from sentence_transformers import SentenceTransformer

# 模型名称
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# 本地保存路径
SAVE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "models", "sentence-transformer", MODEL_NAME)

def download_model():
    """下载并保存模型到本地"""
    print(f"正在下载模型: {MODEL_NAME}")
    print(f"保存路径: {SAVE_PATH}")
    
    # 创建目录
    os.makedirs(SAVE_PATH, exist_ok=True)
    
    # 下载模型
    model = SentenceTransformer(MODEL_NAME)
    
    # 保存模型
    model.save(SAVE_PATH)
    
    print(f"模型已保存到: {SAVE_PATH}")
    return SAVE_PATH

if __name__ == "__main__":
    download_model()
