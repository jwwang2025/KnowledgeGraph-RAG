"""
手动下载 ChatGLM-6B 模型的脚本
从 Hugging Face 下载模型文件到本地
支持使用 transformers 库加载（PyTorch 格式）
"""
import os
import urllib.request
import urllib.error
from pathlib import Path

# 模型保存路径（相对于项目根目录）
# 获取项目根目录（脚本位于 models/1_model-download/，向上两级到项目根目录）
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "chatglm-6b"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Hugging Face 模型仓库URL
# 使用镜像站（hf-mirror.com）以提高下载速度
# https://github.com/zai-org/ChatGLM-6B
# https://hf-mirror.com/zai-org/chatglm-6b/tree/main
BASE_URL = "https://hf-mirror.com/zai-org/chatglm-6b/resolve/main"

# 必需的配置/分词器文件（根据仓库实际文件列表）
REQUIRED_FILES = [
    "config.json",  # 模型配置
    "tokenizer_config.json",  # 分词器配置
    "ice_text.model",  # BPE 词表（2.71 MB）
    "pytorch_model.bin.index.json",  # 权重索引（指向8个分片）
]

# 模型权重分片（ChatGLM-6B 在该仓库为 8 片，总计约13.4 GB）
MODEL_FILES = [
    "pytorch_model-00001-of-00008.bin",  # 1.74 GB
    "pytorch_model-00002-of-00008.bin",  # 1.88 GB
    "pytorch_model-00003-of-00008.bin",  # 1.98 GB
    "pytorch_model-00004-of-00008.bin",  # 1.91 GB
    "pytorch_model-00005-of-00008.bin",  # 1.88 GB
    "pytorch_model-00006-of-00008.bin",  # 1.88 GB
    "pytorch_model-00007-of-00008.bin",  # 1.07 GB
    "pytorch_model-00008-of-00008.bin",  # 1.07 GB
]

# 自定义代码文件（trust_remote_code=True，必需）
CUSTOM_CODE_FILES = [
    "modeling_chatglm.py",  # 模型架构代码（57.6 kB）
    "configuration_chatglm.py",  # 配置类代码（4.28 kB）
    "tokenization_chatglm.py",  # 分词器代码（17 kB）
    "quantization.py",  # 量化支持代码（15.1 kB）
]

# 其他可选文件（许可证、文档、Git元数据和测试文件）
OPTIONAL_FILES = [
    "LICENSE",  # 许可证文件
    "MODEL_LICENSE",  # 模型许可证
    "README.md",  # 说明文档
    ".gitattributes",  # Git属性配置（Git LFS配置等，不影响模型运行）
    "test_modeling_chatglm.py",  # 测试文件（用于测试模型代码，不影响模型运行）
]

# 所有需要下载的文件
FILES_TO_DOWNLOAD = REQUIRED_FILES + MODEL_FILES + CUSTOM_CODE_FILES + OPTIONAL_FILES

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
    print("ChatGLM-6B 模型下载脚本")
    print("=" * 60)
    print(f"模型保存路径: {MODEL_DIR}")
    print(f"下载源: {BASE_URL}")
    print(f"需要下载的文件数: {len(FILES_TO_DOWNLOAD)}")
    print("=" * 60)
    print("\n注意: 模型权重文件分为8个分片，总计约13.4 GB，下载可能需要较长时间")
    print("建议使用稳定的网络连接，或考虑使用 Git LFS 下载")
    print("\n必需文件说明:")
    print("  - 配置文件: config.json, tokenizer_config.json")
    print("  - 分词器: ice_text.model (BPE词表)")
    print("  - 权重索引: pytorch_model.bin.index.json")
    print("  - 权重分片: 8个 pytorch_model-XXXXX-of-00008.bin 文件")
    print("  - 自定义代码: modeling_chatglm.py, configuration_chatglm.py, tokenization_chatglm.py, quantization.py")
    print("\n可选文件说明:")
    print("  - LICENSE, MODEL_LICENSE, README.md: 许可证和文档")
    print("  - .gitattributes: Git配置（不影响模型运行，但可保持仓库完整性）")
    print("  - test_modeling_chatglm.py: 测试文件（不影响模型运行，用于测试模型代码）")
    print("")
    
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
    print("下载完成统计")
    print("=" * 60)
    print(f"成功下载: {success_count} 个文件")
    print(f"跳过下载: {skip_count} 个文件")
    if failed_files:
        print(f"下载失败: {len(failed_files)} 个文件")
        print("失败文件列表:")
        for filename in failed_files:
            print(f"  - {filename}")
    else:
        print("所有文件下载成功！")
    print("=" * 60)
    
    # 验证关键文件是否存在
    print("\n验证关键文件...")
    critical_files = [
        "config.json",  # 模型配置（必需）
        "tokenizer_config.json",  # 分词器配置（必需）
        "ice_text.model",  # BPE词表（必需，ChatGLM使用自定义分词器）
        "pytorch_model.bin.index.json",  # 权重索引（必需，指向8个分片）
        "modeling_chatglm.py",  # 模型代码（必需，trust_remote_code=True）
        "configuration_chatglm.py",  # 配置代码（必需，trust_remote_code=True）
        "tokenization_chatglm.py",  # 分词器代码（必需，trust_remote_code=True）
    ]
    
    missing_files = []
    for filename in critical_files:
        file_path = MODEL_DIR / filename
        if not file_path.exists():
            missing_files.append(filename)
    
    # 检查所有8个权重分片文件是否存在
    missing_weight_files = []
    for i in range(1, 9):
        weight_file = MODEL_DIR / f"pytorch_model-{i:05d}-of-00008.bin"
        if not weight_file.exists():
            missing_weight_files.append(f"pytorch_model-{i:05d}-of-00008.bin")
    
    if missing_weight_files:
        missing_files.extend(missing_weight_files)
    
    if missing_files:
        print("⚠ 警告: 以下关键文件缺失:")
        for filename in missing_files:
            print(f"  - {filename}")
        print("\n模型可能无法正常加载，请检查下载是否完整。")
        print("注意: 所有8个权重分片文件都必须存在，缺少任何一个都会导致加载失败。")
    else:
        print("✓ 所有关键文件已就绪，模型可以正常加载。")
        print("✓ 所有8个权重分片文件已就绪。")
        print("\n文件完整性检查通过！可以使用以下代码加载模型：")
        print("  from transformers import AutoTokenizer, AutoModel")
        print(f"  tokenizer = AutoTokenizer.from_pretrained('{MODEL_DIR}', trust_remote_code=True)")
        print(f"  model = AutoModel.from_pretrained('{MODEL_DIR}', trust_remote_code=True)")
    
if __name__ == "__main__":
    main()

