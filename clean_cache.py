"""
清理 PaddleNLP Taskflow 缓存脚本
用于解决 inference.pdmodel 文件缺失的问题
"""
import os
import shutil

def clean_uie_cache():
    """清理 UIE 模型的缓存目录"""
    home_dir = os.path.expanduser("~")
    cache_dir = os.path.join(home_dir, ".paddlenlp", "taskflow", "information_extraction", "uie-base")
    
    if os.path.exists(cache_dir):
        print(f"找到缓存目录: {cache_dir}")
        try:
            shutil.rmtree(cache_dir)
            print("✓ 缓存清理成功！")
            print("下次运行程序时，模型将重新下载和转换。")
        except Exception as e:
            print(f"✗ 清理失败: {e}")
    else:
        print(f"缓存目录不存在: {cache_dir}")
        print("无需清理。")

if __name__ == "__main__":
    print("=" * 50)
    print("PaddleNLP Taskflow 缓存清理工具")
    print("=" * 50)
    clean_uie_cache()

