"""
手动下载 bert-base-chinese 模型的脚本
从 HF Mirror 镜像站下载模型文件到本地
"""
import os
import urllib.request
import urllib.error
from pathlib import Path

# 模型保存路径（相对于项目根目录）
# 获取项目根目录（脚本位于 models/model-download/，向上两级到项目根目录）
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "bert-base-chinese"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# 镜像站基础URL
BASE_URL = "https://hf-mirror.com/google-bert/bert-base-chinese/resolve/main"

# 需要下载的文件列表（tokenizer必需的文件）
REQUIRED_FILES = [
    "tokenizer_config.json",
    "tokenizer.json", 
    "vocab.txt",
]

# 可选文件（如果需要完整模型）
OPTIONAL_FILES = [
    "config.json",
    "pytorch_model.bin",  # 如果需要完整模型，取消注释
    # "model.safetensors",  # 或者使用safetensors格式（更安全）
]

FILES_TO_DOWNLOAD = REQUIRED_FILES + OPTIONAL_FILES

def download_file(url, save_path):
    """下载文件"""
    print(f"正在下载: {url}")
    try:
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min((downloaded / total_size) * 100, 100)
                print(f"\r进度: {percent:.1f}% ({downloaded}/{total_size} bytes)", end='')
            else:
                print(f"\r已下载: {downloaded} bytes", end='')
        
        urllib.request.urlretrieve(url, save_path, reporthook=show_progress)
        print(f"\n✓ 已保存到: {save_path}")
        return True
    except urllib.error.URLError as e:
        print(f"\n✗ 下载失败 (网络错误): {e}")
        return False
    except Exception as e:
        print(f"\n✗ 下载失败: {e}")
        return False

def main():
    
    success_count = 0
    skip_count = 0
    
    for filename in FILES_TO_DOWNLOAD:
        file_path = MODEL_DIR / filename
        file_url = f"{BASE_URL}/{filename}"
        
        # 如果文件已存在，询问是否跳过
        if file_path.exists():
            print(f"\n文件已存在: {file_path}")
            response = input("是否重新下载? (y/n, 默认n): ").strip().lower()
            if response != 'y':
                print("跳过下载")
                skip_count += 1
                continue
        
        if download_file(file_url, file_path):
            success_count += 1
        else:
            print(f"警告: {filename} 下载失败，但程序可能仍可运行（如果只需要tokenizer）")

if __name__ == "__main__":
    main()

