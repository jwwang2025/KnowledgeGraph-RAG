"""
手动下载 uie-base 模型的脚本
从 Hugging Face 下载模型文件到本地
支持使用 transformers 库加载（PyTorch 格式）
"""
import os
import urllib.request
import urllib.error
from pathlib import Path

# 模型保存路径（相对于项目根目录）
# 获取项目根目录（脚本位于 models/model-download/，向上两级到项目根目录）
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "uie-base"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Hugging Face 模型仓库URL
# 使用镜像站（hf-mirror.com）以提高下载速度
BASE_URL = "https://hf-mirror.com/xusenlin/uie-base/resolve/main"

# 必需的文件列表
REQUIRED_FILES = [
    "config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.txt",
    "special_tokens_map.json",
    "added_tokens.json",
]

# 模型权重文件（大文件，约472MB）
MODEL_FILES = [
    "pytorch_model.bin",  # PyTorch 格式的模型权重
]

# 自定义代码文件（用于 transformers 库加载）
CUSTOM_CODE_FILES = [
    "modeling_uie.py",  # 自定义模型类
    "decode_utils.py",  # 解码工具
]

# 所有需要下载的文件
FILES_TO_DOWNLOAD = REQUIRED_FILES + MODEL_FILES + CUSTOM_CODE_FILES

def download_file(url, save_path):
    """下载文件"""
    print(f"正在下载: {url}")
    try:
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min((downloaded / total_size) * 100, 100)
                size_mb = downloaded / (1024 * 1024)
                total_mb = total_size / (1024 * 1024)
                print(f"\r进度: {percent:.1f}% ({size_mb:.1f}/{total_mb:.1f} MB)", end='')
            else:
                print(f"\r已下载: {downloaded / (1024 * 1024):.1f} MB", end='')
        
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
    print("=" * 60)
    print("开始下载 uie-base 模型文件")
    print("=" * 60)
    print(f"模型将保存到: {MODEL_DIR.absolute()}")
    print("=" * 60)
    
    success_count = 0
    skip_count = 0
    failed_files = []
    
    for filename in FILES_TO_DOWNLOAD:
        file_path = MODEL_DIR / filename
        file_url = f"{BASE_URL}/{filename}"
        
        # 如果文件已存在，询问是否跳过
        if file_path.exists():
            file_size = file_path.stat().st_size / (1024 * 1024)  # MB
            print(f"\n文件已存在: {file_path} ({file_size:.1f} MB)")
            response = input("是否重新下载? (y/n, 默认n): ").strip().lower()
            if response != 'y':
                print("跳过下载")
                skip_count += 1
                continue
        
        if download_file(file_url, file_path):
            success_count += 1
        else:
            failed_files.append(filename)
            print(f"警告: {filename} 下载失败")
    
    print("\n" + "=" * 60)
    print(f"下载完成!")
    print(f"成功: {success_count}, 跳过: {skip_count}, 失败: {len(failed_files)}")
    print(f"模型路径: {MODEL_DIR.absolute()}")
    print("=" * 60)
    
    # 检查必需文件
    missing_files = [f for f in REQUIRED_FILES if not (MODEL_DIR / f).exists()]
    
    if missing_files:
        print(f"\n警告: 以下必需文件缺失: {missing_files}")
        print("模型可能无法正常工作")
        print("\n您可以手动从以下地址下载:")
        for filename in missing_files:
            print(f"  {BASE_URL}/{filename}")
    else:
        print("\n✓ 所有必需文件已就绪!")
        
        # 检查模型权重文件
        if not (MODEL_DIR / "pytorch_model.bin").exists():
            print("\n⚠ 注意: 模型权重文件 (pytorch_model.bin) 未下载")
            print("模型无法加载，请确保下载了该文件")
        else:
            print("\n✓ 模型权重文件已就绪!")
        
        print(f"\n现在可以在代码中使用本地路径: {MODEL_DIR.absolute()}")
if __name__ == "__main__":
    main()

