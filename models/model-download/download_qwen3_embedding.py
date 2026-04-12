# -*- coding: utf-8 -*-
"""
Qwen3-Embedding-8B 模型下载脚本
支持离线下载和镜像加速
"""
import os
import sys

# 设置 HuggingFace 镜像（国内加速）
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_ENDPOINT"] = HF_ENDPOINT

def download_qwen3_embedding_model():
    """下载 Qwen3-Embedding-8B 模型"""
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    import torch
    
    model_name = "Qwen/Qwen3-Embedding-8B"
    
    # 获取保存路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    save_path = os.path.join(project_root, "models", "Qwen3-Embedding-8B")
    
    os.makedirs(save_path, exist_ok=True)
    
    print("=" * 60)
    print("Qwen3-Embedding-8B 模型下载工具")
    print("=" * 60)
    print(f"模型名称: {model_name}")
    print(f"保存路径: {save_path}")
    print(f"使用镜像: {HF_ENDPOINT}")
    print("=" * 60)
    
    # 检查是否已存在
    if os.path.exists(os.path.join(save_path, "config.json")):
        print(f"\n模型已存在于: {save_path}")
        response = input("是否重新下载？(y/N): ").strip().lower()
        if response != 'y':
            print("跳过下载")
            return save_path
    
    try:
        # 1. 下载分词器
        print("\n[1/3] 下载分词器...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        tokenizer.save_pretrained(save_path)
        print("分词器下载完成")
        
        # 2. 下载模型配置
        print("\n[2/3] 下载模型配置...")
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config.save_pretrained(save_path)
        print("配置下载完成")
        
        # 3. 下载模型权重
        print("\n[3/3] 下载模型权重...")
        print("提示: 模型较大，请耐心等待...")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model.save_pretrained(save_path)
        
        print("\n" + "=" * 60)
        print("模型下载完成！")
        print(f"保存路径: {save_path}")
        print("=" * 60)
        
        # 验证文件
        print("\n验证下载文件:")
        required_files = ["config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"]
        for f in required_files:
            filepath = os.path.join(save_path, f)
            if os.path.exists(filepath):
                size = os.path.getsize(filepath) / (1024 * 1024)
                print(f"  ✓ {f} ({size:.1f} MB)")
            else:
                print(f"  ✗ {f} (缺失)")
                
        return save_path
        
    except Exception as e:
        print(f"\n下载失败: {e}")
        print("\n可能的原因:")
        print("1. 网络连接问题")
        print("2. 磁盘空间不足")
        print("3. 模型名称拼写错误")
        return None


def check_model_exists():
    """检查模型是否已存在"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    model_path = os.path.join(project_root, "models", "Qwen3-Embedding-8B")
    
    config_file = os.path.join(model_path, "config.json")
    tokenizer_file = os.path.join(model_path, "tokenizer_config.json")
    
    if os.path.exists(config_file) and os.path.exists(tokenizer_file):
        return True, model_path
    return False, model_path


def get_model_info():
    """获取模型信息"""
    exists, model_path = check_model_exists()
    
    print("=" * 60)
    print("Qwen3-Embedding-8B 模型信息")
    print("=" * 60)
    print(f"本地路径: {model_path}")
    print(f"状态: {'已下载' if exists else '未下载'}")
    
    if exists:
        print("\n已下载的文件:")
        for f in os.listdir(model_path):
            filepath = os.path.join(model_path, f)
            if os.path.isfile(filepath):
                size = os.path.getsize(filepath) / (1024 * 1024)
                print(f"  - {f} ({size:.1f} MB)")
                
    print("=" * 60)
    return exists, model_path


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--check":
            get_model_info()
        elif sys.argv[1] == "--help":
            print("用法:")
            print("  python download_qwen3_embedding.py       # 下载模型")
            print("  python download_qwen3_embedding.py --check # 检查模型状态")
            print("  python download_qwen3_embedding.py --help  # 显示帮助")
    else:
        download_qwen3_embedding_model()
