from pathlib import Path
from typing import List

import torch
from transformers import AutoModel, AutoTokenizer
from config.settings import settings

# ========== UIE 模型配置（PyTorch 版） ==========
# 从配置系统获取 UIE 模型路径
UIE_MODEL_NAME = settings.UIE_MODEL_NAME
# 优先使用本地模型路径，如果不存在则使用模型名称（会从HuggingFace下载）
PROJECT_ROOT = Path(__file__).resolve().parents[2]
local_uie_path = PROJECT_ROOT / "models" / UIE_MODEL_NAME.split("/")[-1]
if local_uie_path.exists() and (local_uie_path / "tokenizer_config.json").exists():
    UIE_MODEL_PATH = str(local_uie_path)
    print(f"使用本地 UIE 模型路径: {UIE_MODEL_PATH}")
else:
    UIE_MODEL_PATH = UIE_MODEL_NAME
    print(f"使用 UIE 模型名称（将从HuggingFace下载）: {UIE_MODEL_PATH}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_tokenizer = None
_model = None
_model_device = DEVICE
# ==============================================


def _load_uie_model():
    """延迟加载 PyTorch UIE 模型与分词器，避免重复初始化。"""
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(
            UIE_MODEL_PATH,
            trust_remote_code=True,
        )
        _model = AutoModel.from_pretrained(
            UIE_MODEL_PATH,
            trust_remote_code=True,
        )
        _model.to(_model_device)
        _model.eval()
    return _model, _tokenizer


def torch_relation_ie(content: List[str]):
    model, tokenizer = _load_uie_model()
    if isinstance(content, str):
        content = [content]

    # 从配置系统获取 schema
    schema = settings.get_schema()

    return model.predict(
        tokenizer,
        content,
        schema=schema,
        batch_size=2,
        device=_model_device,
        show_progress_bar=False,
    )


# 关系抽取并修改json文件
def rel_json(content):
    all_relations = [] # 定义一个空列表，用于存储每个chapter的关系信息
    res_relation = torch_relation_ie(content)  # 传入文本进行关系识别
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
