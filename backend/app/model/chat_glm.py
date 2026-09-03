"""ChatGLM 模型调用与 Adaptive-RAG + Self-RAG + CoT + Citation + LangChain 集成，支持流式输出。"""

import os
import sys

# backend 目录与项目根目录加入 sys.path（config 位于项目根目录）
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PROJECT_ROOT = os.path.dirname(_BACKEND_DIR)
for _p in (_BACKEND_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import json
from typing import List, Optional, Generator, Tuple, Any, Dict
from config.settings import settings

model = None
tokenizer = None
init_history = None

rag_engine = None
langchain_adapter = None
langsmith_manager = None

NER_ETYPES = ["物体类", "人物类", "地点类", "组织机构类", "事件类", "世界地区类", "术语类"]
REF_MARKER = "===参考资料==="
FALLBACK_PROMPT_TEMPLATE = (
    "\n===参考资料===\n{ref}；\n\n"
    "根据上面资料，用简洁且准确的话回答下面问题：\n{query}"
)
PRE_PROMPT = "你叫 ChatKG，是一个图谱问答机器人，此为背景。下面开始聊天吧！"

# 模块级单例，避免每次请求重建
_ner = None
_wiki_searcher = None
_image_searcher = None
_opencc_t2s = None
_vector_searcher = None


def _get_ner():
    global _ner
    if _ner is None:
        from app.nlp import Ner
        _ner = Ner()
    return _ner


def _get_wiki_searcher():
    global _wiki_searcher
    if _wiki_searcher is None:
        from app.search import WikiSearcher
        _wiki_searcher = WikiSearcher()
    return _wiki_searcher


def _get_image_searcher():
    global _image_searcher
    if _image_searcher is None:
        from app.search import ImageSearcher
        _image_searcher = ImageSearcher()
    return _image_searcher


def _get_opencc_t2s():
    global _opencc_t2s
    if _opencc_t2s is None:
        from opencc import OpenCC
        _opencc_t2s = OpenCC('t2s')
    return _opencc_t2s


def _get_vector_searcher():
    global _vector_searcher
    if _vector_searcher is None:
        from app.search import VectorSearcher
        _vector_searcher = VectorSearcher(
            collection_name="project_v1_docs",
            persist_dir="./data/vector_db"
        )
    return _vector_searcher


def init_rag_engine():
    """初始化 Adaptive-RAG + CoT 引擎（单例）"""
    global rag_engine
    if rag_engine is None:
        from app.rag import AdaptiveRAGEngine
        rag_engine = AdaptiveRAGEngine(
            project_name="project_v1",
            vector_db_path="./data/vector_db",
            enable_evaluation=True,       # 启用 Self-RAG 评估
            enable_iteration=False,       # 可选：启用迭代检索
            enable_cot=True,              # 启用 CoT 思维链
            default_cot_mode="zero_shot"  # 默认 Zero-shot CoT 模式
        )
    return rag_engine


def init_langsmith():
    """初始化 LangSmith 追踪（读取 LANGSMITH_API_KEY 等环境变量）"""
    global langsmith_manager
    if langsmith_manager is None:
        from app.rag.langsmith_integration import get_langsmith_manager

        langsmith_manager = get_langsmith_manager()
        if langsmith_manager.is_enabled():
            print(f"[LangSmith] 初始化完成，项目: {langsmith_manager.config.project_name}")
        else:
            print("[LangSmith] 未配置 API Key，追踪功能已禁用")

    return langsmith_manager


def init_langchain_adapter():
    """初始化 LangChain 适配器（单例）"""
    global langchain_adapter
    if langchain_adapter is None:
        from app.rag.langchain_components import LangChainAdapter, create_langchain_vectorstore

        vectorstore = create_langchain_vectorstore(
            collection_name="project_v1_docs",
            persist_dir="./data/vector_db"
        )
        langchain_adapter = LangChainAdapter(vectorstore=vectorstore)
        print("[LangChain] 适配器初始化完成")

    return langchain_adapter


def predict(user_input: str, history: List[Tuple[str, str]] = None) -> Tuple[str, List[Tuple[str, str]]]:
    """同步对话预测，返回 (response, new_history)"""
    global model, tokenizer, init_history
    if not history:
        history = init_history or []
    return model.chat(tokenizer, user_input, history)


def stream_predict(user_input: str, history: List[Tuple[str, str]] = None,
                   use_adaptive_rag: bool = True,
                   enable_evaluation: bool = True,
                   enable_cot: bool = True,
                   include_citations: bool = True,
                   use_langchain: bool = False) -> Generator[bytes, None, None]:
    """流式对话预测 (Adaptive-RAG + Self-RAG + CoT + Citation + LangChain)，yield JSON 字节流"""
    global model, tokenizer, init_history

    if not history:
        history = init_history or []

    base_result = {
        "history": history,
        "query": user_input,
        "rag_context": None,
        "evaluation": None,
        "citations": None,
        "image": None,
        "graph": {},
        "wiki": None,
        "metadata": {}
    }

    chat_input = user_input
    use_retrieval = False
    rag_metadata = {}

    if use_adaptive_rag and model is not None:
        try:
            engine = init_rag_engine()
            context = engine.process(user_input, history)

            plan = context.retrieval_plan
            retrieval = context.retrieval_result
            rag_metadata = {
                "question_type": plan.question_type.value if plan else None,
                "confidence": plan.confidence if plan else 0,
                "sources_used": retrieval.total_sources_used if retrieval else 0,
                "total_time": f"{context.total_time:.2f}s",
                "stages": context.stage_history,
                "final_action": context.final_action.value,
                "use_retrieval": context.use_retrieval,
                "use_cot": plan.use_cot if plan else False,
                "cot_mode": plan.cot_mode if plan else "direct",
                "reasoning_depth": plan.reasoning_depth if plan else 0,
                "langchain_enabled": use_langchain
            }

            base_result["rag_context"] = {
                "sources": list(retrieval.results.keys()) if retrieval else [],
                "triples_count": len(retrieval.triples) if retrieval else 0,
                "docs_count": len(retrieval.documents) if retrieval else 0,
                "has_wiki": retrieval.wiki_summary is not None if retrieval else False,
                "has_image": retrieval.image_url is not None if retrieval else False,
                "use_cot": plan.use_cot if plan else False,
                "cot_mode": plan.cot_mode if plan else "direct",
                "reasoning_steps": len(context.reasoning_chain.steps) if context.reasoning_chain else 0,
                "langchain_enabled": use_langchain
            }

            if enable_evaluation and context.evaluation_report:
                base_result["evaluation"] = context.evaluation_report.get_summary()
                base_result["metadata"]["evaluation"] = context.evaluation_report.get_summary()

            if context.reasoning_chain and enable_cot:
                base_result["metadata"]["cot"] = {
                    "mode": context.reasoning_chain.mode.value,
                    "steps": len(context.reasoning_chain.steps),
                    "depth": context.reasoning_chain.depth
                }

            base_result["metadata"]["rag"] = rag_metadata

            if include_citations and context.citation_set:
                try:
                    from app.rag.citation import CitationEmbedder
                except ImportError:
                    CitationEmbedder = None

                if CitationEmbedder:
                    citation_embedder = CitationEmbedder(format_type="superscript")
                    base_result["citations"] = citation_embedder.format_citations_for_api(
                        context.citation_set.citations
                    )
                    print(f"[Citation] 返回 {len(context.citation_set.citations)} 条引用")

            if context.use_retrieval and context.assembled_prompt:
                chat_input = context.assembled_prompt
                use_retrieval = True
                print(f"[Adaptive-RAG] 使用检索增强，问题类型: {rag_metadata['question_type']}")
                if rag_metadata["use_cot"]:
                    print(f"[CoT] 启用思维链推理，模式: {rag_metadata['cot_mode']}")
            else:
                print("[Adaptive-RAG] 无需检索，直接生成")

        except Exception as e:
            print(f"[Adaptive-RAG 错误] {e}")
            base_result["metadata"]["rag_error"] = str(e)

    if use_langchain and not use_retrieval:
        try:
            init_langchain_adapter()
            base_result["metadata"]["langchain_note"] = "LangChain 适配器已就绪，需要配置 LLM"
            print("[LangChain] RAG Chain 已准备就绪")
        except Exception as e:
            print(f"[LangChain 错误] {e}")
            base_result["metadata"]["langchain_error"] = str(e)

    if not use_retrieval and model is not None:
        try:
            ref = _fallback_retrieval(user_input)
            if ref:
                chat_input = FALLBACK_PROMPT_TEMPLATE.format(ref=ref, query=user_input)
                use_retrieval = True
        except Exception as e:
            print(f"[备用检索错误] {e}")

    try:
        base_result["image"] = _get_image_searcher().search(user_input)
    except Exception as e:
        print(f"[图像搜索错误] {e}")

    wiki = None
    if not use_retrieval or not rag_metadata.get("has_wiki"):
        try:
            cc = _get_opencc_t2s()
            entities = _get_ner().get_entities(user_input, etypes=NER_ETYPES)

            wiki_searcher = _get_wiki_searcher()
            for ent in entities + [user_input]:
                wiki = wiki_searcher.search(ent)
                if wiki is not None:
                    break

            if wiki:
                wiki = {
                    "title": cc.convert(wiki.title),
                    "summary": cc.convert(wiki.summary)[:500],
                }
                print(f"[Wikipedia] {wiki['title']}")
        except Exception as e:
            print(f"[Wikipedia 搜索错误] {e}")

    base_result["wiki"] = wiki or {"title": "无相关信息", "summary": "暂无相关描述"}

    if model is not None:
        # 清理历史记录中的参考资料部分
        clean_history = [
            (q.split(REF_MARKER)[0] if REF_MARKER in q else q, r)
            for q, r in history
        ]

        print(f"[ChatGLM] 输入: {chat_input[:100]}..." if len(chat_input) > 100 else f"[ChatGLM] 输入: {chat_input}")

        for response, history in model.stream_chat(tokenizer, chat_input, clean_history):
            updates = {"query": history[-1][0], "response": history[-1][1]} if history else {}

            result = base_result.copy()
            result["history"] = history
            result["updates"] = updates

            yield json.dumps(result, ensure_ascii=False).encode('utf8') + b'\n'
    else:
        result = base_result.copy()
        result["updates"] = {"query": user_input, "response": "模型加载中，请稍后再试"}
        result["history"] = history

        yield json.dumps(result, ensure_ascii=False).encode('utf8') + b'\n'


def _fallback_retrieval(user_input: str) -> str:
    """备用检索方案：知识图谱三元组 + 向量数据库（向后兼容降级路径）"""
    ref = ""

    try:
        from app.kg import search_node_item, convert_graph_to_triples

        entities = _get_ner().get_entities(user_input, etypes=NER_ETYPES)
        print(f"[备用检索] 实体: {entities}")

        triples = []
        graph = {'nodes': [], 'links': [], 'sents': []}

        for entity in entities[:5]:
            graph = search_node_item(entity, graph if graph['nodes'] else None)
            if graph:
                triples += convert_graph_to_triples(graph, entity)

        MAX_TRIPLES = 10
        if len(triples) > MAX_TRIPLES:
            triples = triples[:MAX_TRIPLES]

        if triples:
            triples_str = "；".join(f"({t[0]} {t[1]} {t[2]})" for t in triples)
            ref += f"三元组信息：{triples_str}；"

        try:
            search_results = _get_vector_searcher().search(user_input, top_k=3)
            if search_results and search_results.get('documents') and search_results['documents'][0]:
                docs = search_results['documents'][0]
                docs_str = "；".join(docs[:3])
                ref += f"相关文档：{docs_str}；"
                print(f"[备用检索] 检索到 {len(docs)} 条文档")
        except Exception as e:
            print(f"[向量检索错误] {e}")

    except Exception as e:
        print(f"[备用检索错误] {e}")

    return ref


def get_langchain_rag_chain(
    template_type: str = "rag",
    use_history: bool = False,
    use_cot: bool = False
):
    """获取 LangChain RAG Chain（需先配置 LLM，否则返回 None）"""
    try:
        adapter = init_langchain_adapter()
        return adapter.build_rag_chain(
            template_type=template_type,
            use_history=use_history,
            use_cot=use_cot
        )
    except Exception as e:
        print(f"[LangChain] 获取 RAG Chain 失败: {e}")
        return None


def invoke_langchain_rag(query: str, history: List[tuple] = None) -> Dict[str, Any]:
    """使用 LangChain RAG Chain 处理查询"""
    chain = get_langchain_rag_chain()
    if chain is None:
        return {"error": "LangChain RAG Chain 未就绪"}

    try:
        result = chain.invoke({"question": query, "history": history or []})
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


def start_model():
    """启动并加载 ChatGLM 模型，并预热 RAG / LangChain / LangSmith 组件"""
    global model, tokenizer, init_history

    model_path = settings.CHATGLM_MODEL_PATH

    # 相对路径转绝对路径（相对于项目根目录）
    if not os.path.isabs(model_path) and not model_path.startswith(("http://", "https://")):
        abs_model_path = os.path.join(_PROJECT_ROOT, model_path.lstrip("./"))
        if os.path.exists(abs_model_path):
            model_path = abs_model_path

    print(f"使用本地模型路径: {model_path}" if os.path.exists(model_path)
          else f"使用 HuggingFace Hub 模型: {model_path}")

    import types
    import importlib.util

    if os.path.exists(model_path):
        # 直接导入 ChatGLM 自定义模块（避免 transformers trust_remote_code 的路径问题）
        parent_module = types.ModuleType("transformers_modules")
        parent_module.__path__ = [model_path]
        sys.modules["transformers_modules"] = parent_module

        for mod_name, filename in [
            ("configuration_chatglm", "configuration_chatglm.py"),
            ("tokenization_chatglm", "tokenization_chatglm.py"),
            ("modeling_chatglm", "modeling_chatglm.py"),
        ]:
            spec = importlib.util.spec_from_file_location(
                f"transformers_modules.{mod_name}", os.path.join(model_path, filename))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            sys.modules[f"transformers_modules.{mod_name}"] = module

        ChatGLMConfig = sys.modules["transformers_modules.configuration_chatglm"].ChatGLMConfig
        ChatGLMTokenizer = sys.modules["transformers_modules.tokenization_chatglm"].ChatGLMTokenizer
        ChatGLMForConditionalGeneration = sys.modules["transformers_modules.modeling_chatglm"].ChatGLMForConditionalGeneration

        config = ChatGLMConfig.from_pretrained(model_path)
        tokenizer = ChatGLMTokenizer.from_pretrained(model_path)
        model = ChatGLMForConditionalGeneration.from_pretrained(model_path, config=config)
    else:
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True)

    import torch
    if torch.cuda.is_available():
        print("使用 GPU (CUDA)")
        model = model.half().cuda()
    else:
        print("使用 CPU（注意：CPU 模式速度较慢）")
        model = model.float()

    model.eval()

    _, history = predict(PRE_PROMPT, [])
    init_history = history

    print("预热 Adaptive-RAG 引擎...")
    _ = init_rag_engine()
    print("Adaptive-RAG 引擎就绪")

    print("预热 LangChain 组件...")
    try:
        _ = init_langchain_adapter()
        print("LangChain 组件就绪")
    except Exception as e:
        print(f"[警告] LangChain 组件初始化失败: {e}")

    print("初始化 LangSmith 追踪...")
    try:
        _ = init_langsmith()
        print("LangSmith 追踪就绪")
    except Exception as e:
        print(f"[警告] LangSmith 初始化失败: {e}")


def get_rag_stats() -> dict:
    """获取 RAG 统计信息"""
    return rag_engine.get_stats() if rag_engine else {}


def reset_rag_stats():
    """重置 RAG 统计信息"""
    if rag_engine:
        rag_engine.reset_stats()


def get_langchain_stats() -> dict:
    """获取 LangChain 统计信息"""
    return {"adapter_ready": True} if langchain_adapter else {"adapter_ready": False}


def get_langsmith_stats() -> dict:
    """获取 LangSmith 统计信息"""
    if langsmith_manager and langsmith_manager.is_enabled():
        return {"enabled": True, "project": langsmith_manager.config.project_name}
    return {"enabled": False}


def is_langsmith_enabled() -> bool:
    """检查 LangSmith 是否启用"""
    if langsmith_manager is None:
        return init_langsmith().is_enabled()
    return langsmith_manager.is_enabled()
