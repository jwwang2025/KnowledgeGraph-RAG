"""
快速测试 ChatGLM-6B 模型是否可用
检查模型文件完整性，尝试加载模型并进行简单对话测试
"""
import os
import sys
from pathlib import Path

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "chatglm-6b"

def check_model_files():
    """检查关键模型文件是否存在"""
    print("=" * 60)
    print("步骤 1: 检查模型文件完整性")
    print("=" * 60)
    
    # 必需的关键文件
    critical_files = [
        "config.json",
        "tokenizer_config.json",
        "ice_text.model",
        "pytorch_model.bin.index.json",
        "modeling_chatglm.py",
        "configuration_chatglm.py",
        "tokenization_chatglm.py",
    ]
    
    missing_files = []
    for filename in critical_files:
        file_path = MODEL_DIR / filename
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"✓ {filename} ({size_mb:.2f} MB)")
        else:
            print(f"✗ {filename} (缺失)")
            missing_files.append(filename)
    
    # 检查权重分片文件
    print("\n检查模型权重分片文件:")
    missing_weight_files = []
    for i in range(1, 9):
        weight_file = MODEL_DIR / f"pytorch_model-{i:05d}-of-00008.bin"
        if weight_file.exists():
            size_mb = weight_file.stat().st_size / (1024 * 1024)
            print(f"✓ pytorch_model-{i:05d}-of-00008.bin ({size_mb:.1f} MB)")
        else:
            print(f"✗ pytorch_model-{i:05d}-of-00008.bin (缺失)")
            missing_weight_files.append(f"pytorch_model-{i:05d}-of-00008.bin")
    
    if missing_files or missing_weight_files:
        print(f"\n⚠ 警告: 发现 {len(missing_files) + len(missing_weight_files)} 个缺失文件")
        return False
    else:
        print("\n✓ 所有关键文件检查通过")
        return True

def test_model_loading():
    """测试模型加载"""
    print("\n" + "=" * 60)
    print("步骤 2: 测试模型加载")
    print("=" * 60)
    
    if not MODEL_DIR.exists():
        print(f"✗ 模型目录不存在: {MODEL_DIR}")
        return None, None
    
    try:
        from transformers import AutoTokenizer, AutoModel
        import torch
        
        print(f"模型路径: {MODEL_DIR}")
        print("正在加载分词器...")
        tokenizer = AutoTokenizer.from_pretrained(
            str(MODEL_DIR), 
            trust_remote_code=True
        )
        print("✓ 分词器加载成功")
        
        print("正在加载模型（这可能需要几分钟，请耐心等待）...")
        model = AutoModel.from_pretrained(
            str(MODEL_DIR), 
            trust_remote_code=True
        )
        print("✓ 模型加载成功")
        
        # 检查CUDA是否可用
        if torch.cuda.is_available():
            print(f"检测到 GPU: {torch.cuda.get_device_name(0)}")
            print("正在将模型移至 GPU...")
            model = model.half().cuda()
            print("✓ 模型已移至 GPU")
        else:
            print("⚠ 未检测到 GPU，使用 CPU 模式（速度较慢）")
            model = model.float()
        
        model.eval()
        print("✓ 模型已设置为评估模式")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_chat(model, tokenizer):
    """测试模型对话功能"""
    print("\n" + "=" * 60)
    print("步骤 3: 测试模型对话功能")
    print("=" * 60)
    
    if model is None or tokenizer is None:
        print("✗ 模型或分词器未加载，无法进行对话测试")
        return False
    
    try:
        # 测试问题
        test_questions = [
            "你好",
            "请介绍一下你自己",
        ]
        
        history = []
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n测试问题 {i}: {question}")
            print("-" * 60)
            
            try:
                response, history = model.chat(tokenizer, question, history)
                print(f"模型回答: {response}")
                print("✓ 对话测试成功")
            except Exception as e:
                print(f"✗ 对话测试失败: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        print("\n" + "=" * 60)
        print("✓ 所有对话测试通过！")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"✗ 对话测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试流程"""
    print("\n" + "=" * 60)
    print("ChatGLM-6B 模型可用性测试")
    print("=" * 60)
    print(f"模型目录: {MODEL_DIR}")
    print()
    
    # 步骤1: 检查文件
    if not check_model_files():
        print("\n⚠ 模型文件不完整，请先完成下载")
        return
    
    # 步骤2: 加载模型
    model, tokenizer = test_model_loading()
    if model is None or tokenizer is None:
        print("\n✗ 模型加载失败，请检查错误信息")
        return
    
    # 步骤3: 测试对话
    success = test_chat(model, tokenizer)
    
    # 最终结果
    print("\n" + "=" * 60)
    if success:
        print("🎉 测试完成：模型可用！")
        print("=" * 60)
        print("\n模型已成功加载并可以正常对话。")
        print("你现在可以在项目中使用这个模型了。")
    else:
        print("❌ 测试失败：模型存在问题")
        print("=" * 60)
        print("\n请检查错误信息并修复问题。")
    print()

if __name__ == "__main__":
    main()

