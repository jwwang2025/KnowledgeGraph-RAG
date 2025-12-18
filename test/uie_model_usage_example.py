"""
UIE 模型使用示例
展示如何使用从 Hugging Face 下载的 uie-base 模型
"""
import os
from pathlib import Path

# 获取模型路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "uie-base"

def example_1_use_transformers():
    """
    示例1: 使用 transformers 库加载模型
    注意: 这个模型需要自定义的 modeling_uie.py 文件
    """
    print("=" * 60)
    print("示例1: 使用 transformers 库加载 UIE 模型")
    print("=" * 60)
    
    try:
        from transformers import AutoTokenizer, AutoModel
        
        model_path = str(MODEL_DIR.absolute())
        
        # 检查模型文件是否存在
        if not (MODEL_DIR / "pytorch_model.bin").exists():
            print(f"错误: 模型文件不存在，请先运行 download_uie_model.py 下载模型")
            return
        
        print(f"正在从 {model_path} 加载模型...")
        
        # 加载 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print("✓ Tokenizer 加载成功")
        
        # 加载模型（需要 trust_remote_code=True 因为使用了自定义的 modeling_uie.py）
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        print("✓ 模型加载成功")
        
        # 使用示例
        text = "张三在北京大学工作，他的研究方向是自然语言处理。"
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        
        print(f"\n输入文本: {text}")
        print(f"Tokenized: {inputs}")
        
        # 注意: 实际的 UIE 推理需要特定的解码逻辑（decode_utils.py）
        # 这里只是展示如何加载模型
        
    except ImportError:
        print("错误: 请先安装 transformers 库")
        print("运行: pip install transformers")
    except Exception as e:
        print(f"错误: {e}")
        print("\n提示:")
        print("1. 确保已下载所有模型文件（运行 download_uie_model.py）")
        print("2. 确保安装了 transformers 库: pip install transformers")
        print("3. 如果遇到自定义代码问题，确保 modeling_uie.py 和 decode_utils.py 在模型目录中")


def example_2_use_paddlenlp():
    """
    示例2: 使用 PaddleNLP 的 Taskflow（推荐方式）
    注意: PaddleNLP 会自动下载 PaddlePaddle 格式的模型
    如果要从本地加载，需要将 PyTorch 模型转换为 PaddlePaddle 格式
    """
    print("\n" + "=" * 60)
    print("示例2: 使用 PaddleNLP Taskflow（推荐）")
    print("=" * 60)
    
    try:
        from paddlenlp import Taskflow
        
        # 定义 schema（关系抽取模式）
        schema = {
            "人物": ["工作单位", "研究方向"],
            "工作单位": ["地点"],
        }
        
        print("正在初始化 UIE Taskflow...")
        # 使用默认的 uie-base 模型（PaddleNLP 会自动下载）
        ie = Taskflow("information_extraction", schema=schema, model="uie-base")
        print("✓ Taskflow 初始化成功")
        
        # 使用示例
        text = "张三在北京大学工作，他的研究方向是自然语言处理。"
        result = ie(text)
        
        print(f"\n输入文本: {text}")
        print(f"抽取结果: {result}")
        
    except ImportError:
        print("错误: 请先安装 PaddleNLP 库")
        print("运行: pip install paddlenlp")
    except Exception as e:
        print(f"错误: {e}")


def example_3_check_model_files():
    """检查模型文件是否完整"""
    print("\n" + "=" * 60)
    print("示例3: 检查模型文件")
    print("=" * 60)
    
    required_files = [
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.txt",
        "special_tokens_map.json",
        "added_tokens.json",
        "pytorch_model.bin",
        "modeling_uie.py",
        "decode_utils.py",
    ]
    
    print(f"模型目录: {MODEL_DIR.absolute()}")
    print("\n文件检查:")
    
    all_exist = True
    for filename in required_files:
        file_path = MODEL_DIR / filename
        if file_path.exists():
            size = file_path.stat().st_size / (1024 * 1024)  # MB
            print(f"  ✓ {filename} ({size:.1f} MB)")
        else:
            print(f"  ✗ {filename} (缺失)")
            all_exist = False
    
    if all_exist:
        print("\n✓ 所有文件完整，模型可以使用")
    else:
        print("\n⚠ 部分文件缺失，请运行 download_uie_model.py 下载")


if __name__ == "__main__":
    # 检查模型文件
    example_3_check_model_files()
    
    # 如果文件完整，尝试加载
    if (MODEL_DIR / "pytorch_model.bin").exists():
        print("\n" + "=" * 60)
        print("选择使用方式:")
        print("1. 使用 transformers 库（PyTorch 格式）")
        print("2. 使用 PaddleNLP Taskflow（推荐，自动下载 PaddlePaddle 格式）")
        print("=" * 60)
        
        choice = input("\n请输入选择 (1/2，直接回车跳过): ").strip()
        
        if choice == "1":
            example_1_use_transformers()
        elif choice == "2":
            example_2_use_paddlenlp()
        else:
            print("已跳过示例运行")
    else:
        print("\n请先运行 download_uie_model.py 下载模型文件")

