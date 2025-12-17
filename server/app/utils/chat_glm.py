import os
import sys
sys.path.append('server/app')
import json
from opencc import OpenCC
from transformers import AutoTokenizer, AutoModel
from app.utils.image_searcher import ImageSearcher
from app.utils.query_wiki import WikiSearcher
from app.utils.ner import Ner
from app.utils.graph_utils import convert_graph_to_triples, search_node_item

model = None
tokenizer = None
init_history = None

ner = Ner()
image_searcher = ImageSearcher()
wiki_searcher = WikiSearcher()
cc = OpenCC('t2s')

def predict(user_input, history=None):
    global model, tokenizer, init_history
    if not history:
        history = init_history
    return model.chat(tokenizer, user_input, history)


def stream_predict(user_input, history=None):
    global model, tokenizer, init_history
    if not history:
        history = init_history

    print(f"[stream_predict] 收到用户输入: {user_input}")
    ref = ""

    # 获取实体
    graph = {}
    entities = []
    try:
        print("[stream_predict] 开始实体识别...")
        entities = ner.get_entities(user_input, etypes=["物体类", "人物类", "地点类", "组织机构类", "事件类", "世界地区类", "术语类"])
        print(f"[stream_predict] 识别到的实体: {entities}")
    except Exception as e:
        print(f"[stream_predict] 实体识别出错: {e}")
        entities = []

    # 获取实体的三元组
    triples = []
    try:
        print("[stream_predict] 开始查询知识图谱...")
        for entity in entities:
            graph = search_node_item(entity, graph if graph else None)

            if graph:
                triples += convert_graph_to_triples(graph, entity)
        print(f"[stream_predict] 找到 {len(triples)} 个三元组")
    except Exception as e:
        print(f"[stream_predict] 知识图谱查询出错: {e}")
        graph = {}
        triples = []

    triples_str = ""
    for t in triples:
        triples_str += f"({t[0]} {t[1]} {t[2]})；"

    if triples_str:
        ref += f"三元组信息：{triples_str}；"

    # 图片搜索
    try:
        print("[stream_predict] 开始图片搜索...")
        image = image_searcher.search(user_input)
    except Exception as e:
        print(f"[stream_predict] 图片搜索出错: {e}")
        image = None

    # Wikipedia 搜索
    wiki = None
    try:
        print("[stream_predict] 开始 Wikipedia 搜索...")
        for ent in entities + [user_input]:
            wiki = wiki_searcher.search(ent)
            if wiki is not None:
                break
    except Exception as e:
        print(f"[stream_predict] Wikipedia 搜索出错: {e}")
        wiki = None

    # 将Wikipedia搜索到的繁体转为简体
    if wiki:
        ref += cc.convert(wiki.summary)
        wiki = {
            "title": cc.convert(wiki.title),
            "summary": cc.convert(wiki.summary),
        }
        print(wiki)
    else:
        wiki = {
            "title": "无相关信息",
            "summary": "暂无相关描述",
        }

    if model is not None:
        if ref:
            chat_input = f"\n===参考资料===：\n{ref}；\n\n根据上面资料，用简洁且准确的话回答下面问题：\n{user_input}"
        else:
            chat_input = user_input

        clean_history = []
        for user_input, response in history:
            if "===参考资料===" in user_input:
                user_input = user_input.split("===参考资料===")[0]
            clean_history.append((user_input, response))

        print("chat_input: ", chat_input)
        for response, history in model.stream_chat(tokenizer, chat_input, clean_history):
            updates = {}
            for query, response in history:
                updates["query"] = query
                updates["response"] = response

            result = {
                "history": history,
                "updates": updates,
                "image": image,
                "graph": graph,
                "wiki": wiki
            }
            yield json.dumps(result, ensure_ascii=False).encode('utf8') + b'\n'

    else:
        updates = {
            "query": user_input,
            "response": "模型加载中，请稍后再试"
        }

        result = {
            "history": history,
            "updates": updates,
            "image": image,
            "graph": graph,
            "wiki": wiki
        }
        yield json.dumps(result, ensure_ascii=False).encode('utf8') + b'\n'

# 加载模型
def start_model():
    global model, tokenizer, init_history

    # 从环境变量获取模型路径，如果没有则尝试使用项目根目录下的 models/chatglm-6b
    # 如果本地路径不存在，则使用HuggingFace Hub上的模型ID
    default_local_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'models', 'chatglm-6b')
    default_local_path = os.path.abspath(default_local_path)
    
    model_path = os.getenv('CHATGLM_MODEL_PATH', default_local_path if os.path.exists(default_local_path) else 'THUDM/chatglm-6b')
    
    # 检查是否是本地路径
    if os.path.exists(model_path):
        print(f"使用本地模型路径: {model_path}")
    else:
        print(f"使用HuggingFace Hub模型: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    
    # 检查CUDA是否可用
    import torch
    if torch.cuda.is_available():
        print("使用GPU (CUDA)")
        model = model.half().cuda()
    else:
        print("使用CPU（注意：CPU模式速度较慢）")
        model = model.float()
    
    model.eval()

    pre_prompt = "你叫 ChatKG，是一个图谱问答机器人，此为背景。下面开始聊天吧！"
    _, history = predict(pre_prompt, [])
    init_history = history