import os
import shutil
from data.schema import schema_v4

# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from paddle import inference as paddle_infer
from paddlenlp import Taskflow

# 全局变量，用于缓存 Taskflow 实例，避免重复初始化（提升性能）
_relation_ie_instance = None

def _get_cache_dir():
    """获取 PaddleNLP 缓存目录"""
    import os
    home_dir = os.path.expanduser("~")
    cache_dir = os.path.join(home_dir, ".paddlenlp", "taskflow", "information_extraction", "uie-base")
    return cache_dir

def _clean_cache_if_needed():
    """如果缓存目录存在但推理模型文件缺失，则清理缓存"""
    cache_dir = _get_cache_dir()
    static_dir = os.path.join(cache_dir, "static")
    inference_file = os.path.join(static_dir, "inference.pdmodel")
    
    if os.path.exists(cache_dir) and not os.path.exists(inference_file):
        print(f"检测到缓存目录存在但推理模型文件缺失，正在清理缓存: {cache_dir}")
        try:
            shutil.rmtree(cache_dir)
            print("缓存清理完成，模型将重新下载和转换")
        except Exception as e:
            print(f"清理缓存失败: {e}")

# 定义一个函数，用于关系抽取
def paddle_relation_ie(content):
    global _relation_ie_instance
    
    # 如果实例不存在，则创建（只初始化一次）
    if _relation_ie_instance is None:
        # 先检查并清理损坏的缓存
        _clean_cache_if_needed()
        
        # 尝试多种初始化方式
        try:
            # 方案1：优先使用 use_fast=True（动态图模式，避免模型转换问题）
            try:
                _relation_ie_instance = Taskflow(
                    "information_extraction", 
                    schema=schema_v4.schema, 
                    batch_size=2,
                    use_fast=True
                )
                print("成功使用 use_fast=True 模式初始化 Taskflow（动态图模式）")
            except (TypeError, ValueError) as e:
                # 如果 use_fast 参数不支持或值错误，尝试 use_fast=False
                print(f"use_fast=True 不可用: {e}，尝试 use_fast=False...")
                try:
                    _relation_ie_instance = Taskflow(
                        "information_extraction", 
                        schema=schema_v4.schema, 
                        batch_size=2,
                        use_fast=False
                    )
                    print("成功使用 use_fast=False 模式初始化 Taskflow")
                except TypeError:
                    # 如果 use_fast 参数完全不支持，尝试不使用该参数
                    print("当前版本不支持 use_fast 参数，尝试默认模式...")
                    _relation_ie_instance = Taskflow(
                        "information_extraction", 
                        schema=schema_v4.schema, 
                        batch_size=2
                    )
        except RuntimeError as e:
            if "inference.pdmodel" in str(e) or "Cannot open file" in str(e):
                # 如果仍然失败，清理缓存并使用 use_fast=True 重试
                print(f"初始化失败: {e}")
                print("清理缓存并使用动态图模式重试...")
                _clean_cache_if_needed()
                # 使用 use_fast=True 重试（避免模型转换问题）
                try:
                    _relation_ie_instance = Taskflow(
                        "information_extraction", 
                        schema=schema_v4.schema, 
                        batch_size=2,
                        use_fast=True
                    )
                    print("使用动态图模式初始化成功")
                except (TypeError, ValueError):
                    # 如果不支持 use_fast，使用默认模式
                    _relation_ie_instance = Taskflow(
                        "information_extraction", 
                        schema=schema_v4.schema, 
                        batch_size=2
                    )
            else:
                raise
    
    return _relation_ie_instance(content)


# 关系抽取并修改json文件
def rel_json(content):
    all_relations = [] # 定义一个空列表，用于存储每个chapter的关系信息
    res_relation = paddle_relation_ie(content)  # 传入文本进行关系识别
    for rel in res_relation:
        for sub_type, sub_rel in rel.items():
            for sub in sub_rel:
                if sub.get("relations") is None:
                    continue
                for rel_type, rel_obj in sub["relations"].items():
                    for obj in rel_obj:
                        if not sub['text'] or not obj['text']:
                            continue
                        rel_triple = {"em1Text": sub['text'],"em2Text": obj['text'],"label": rel_type}
                        all_relations.append(rel_triple)
    return all_relations


# 执行函数
# 
def uie_execute(texts):

    sent_id = 0
    all_items = []
    for line in texts:
        line = line.strip()
        all_relations = rel_json(line)

        item = {}
        item["id"] = sent_id
        item["sentText"] = line
        item["relationMentions"] = all_relations

        sent_id += 1
        if sent_id % 10 == 0 and sent_id != 0:
            print("Done {} lines".format(sent_id))

        all_items.append(item)

    return all_items
