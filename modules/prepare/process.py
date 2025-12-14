import os
from data.schema import schema_v4

# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from paddle import inference as paddle_infer
from paddlenlp import Taskflow

# ========== UIE 模型配置 ==========
# 如果需要使用自定义模型路径，请修改以下变量：
# 1. 使用默认模型（自动下载到缓存）: 设置为 None
# 2. 使用自定义路径: 设置为模型目录路径（包含 static/inference.pdmodel 的目录）
# 3. 使用特定模型名称: 设置为模型名称，如 "uie-base", "uie-medium" 等
UIE_MODEL_PATH = None  # 例如: "/path/to/model" 或 "uie-base" 或 None
UIE_MODEL_NAME = "uie-base"  # 当 UIE_MODEL_PATH 为 None 时使用的默认模型
# ===================================

# 定义一个函数，用于关系抽取
def paddle_relation_ie(content):
    # 构建 Taskflow 参数
    taskflow_kwargs = {
        "schema": schema_v4.schema,
        "batch_size": 2
    }
    
    # 如果指定了自定义路径，使用 task_path
    if UIE_MODEL_PATH is not None and os.path.exists(UIE_MODEL_PATH):
        taskflow_kwargs["task_path"] = UIE_MODEL_PATH
        print(f"使用自定义模型路径: {UIE_MODEL_PATH}")
    # 如果指定了模型名称，使用 model 参数
    elif UIE_MODEL_PATH is not None and isinstance(UIE_MODEL_PATH, str) and not os.path.exists(UIE_MODEL_PATH):
        taskflow_kwargs["model"] = UIE_MODEL_PATH
        print(f"使用模型名称: {UIE_MODEL_PATH}")
    # 否则使用默认模型
    else:
        taskflow_kwargs["model"] = UIE_MODEL_NAME
        print(f"使用默认模型: {UIE_MODEL_NAME}")
    
    relation_ie = Taskflow("information_extraction", **taskflow_kwargs)
    return relation_ie(content)


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
