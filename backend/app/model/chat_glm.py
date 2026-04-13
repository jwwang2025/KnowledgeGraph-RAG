"""
ChatGLM 模型调用与 Adaptive-RAG + Self-RAG + CoT 集成模块
支持引用溯源机制
支持 LangChain 集成

本模块整合了:
1. ChatGLM-6B 模型调用
2. Adaptive-RAG 智能检索增强
3. Self-RAG 结果评估与反思
4. CoT 思维链推理
5. Citation 引用溯源
6. LangChain RAG Chain 集成

支持流式输出，实时返回对话状态和引用信息
"""

import os
import sys
sys.path.append('server/app')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
from typing import List, Optional, Generator, Tuple, Any, Dict
from config.settings import settings

# 全局变量
model = None
tokenizer = None
init_history = None

# Adaptive-RAG 组件 (延迟初始化)
rag_engine = None

# LangChain 组件 (延迟初始化)
langchain_adapter = None


def init_rag_engine():
    """初始化 Adaptive-RAG + CoT 引擎"""
    global rag_engine
    if rag_engine is None:
        from app.rag import AdaptiveRAGEngine
        rag_engine = AdaptiveRAGEngine(
            project_name="project_v1",
            vector_db_path="./data/vector_db",
            enable_evaluation=True,    # 启用 Self-RAG 评估
            enable_iteration=False,   # 可选：启用迭代检索
            enable_cot=True,         # 启用 CoT 思维链
            default_cot_mode="zero_shot"  # 默认 Zero-shot CoT 模式
        )
    return rag_engine


def init_langchain_adapter():
    """
    初始化 LangChain 适配器
    
    提供 LangChain 风格的 RAG 接口
    """
    global langchain_adapter
    if langchain_adapter is None:
        from app.rag.langchain_components import LangChainAdapter, create_langchain_vectorstore
        
        # 创建向量存储
        vectorstore = create_langchain_vectorstore(
            collection_name="project_v1_docs",
            persist_dir="./data/vector_db"
        )
        
        langchain_adapter = LangChainAdapter(vectorstore=vectorstore)
        print("[LangChain] 适配器初始化完成")
    
    return langchain_adapter


def predict(user_input: str, history: List[Tuple[str, str]] = None) -> Tuple[str, List[Tuple[str, str]]]:
    """
    同步版本的对话预测
    
    Args:
        user_input: 用户输入
        history: 对话历史
        
    Returns:
        (response, new_history): 响应内容和更新后的历史
    """
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
    """
    流式对话预测 (集成 Adaptive-RAG + Self-RAG + CoT + Citation + LangChain)

    Args:
        user_input: 用户输入
        history: 对话历史
        use_adaptive_rag: 是否使用 Adaptive-RAG (默认 True)
        enable_evaluation: 是否启用结果评估 (默认 True)
        enable_cot: 是否启用思维链 CoT (默认 True)
        include_citations: 是否包含引用信息 (默认 True)
        use_langchain: 是否使用 LangChain RAG Chain (默认 False)

    Yields:
        JSON 编码的流式响应 (包含 citations 字段)
    """
    global model, tokenizer, init_history
    
    if not history:
        history = init_history or []

    # 构建基础返回数据
    base_result = {
        "history": history,
        "query": user_input,
        "rag_context": None,       # RAG 上下文信息
        "evaluation": None,        # 评估报告
        "citations": None,         # 引用溯源信息 (新增)
        "image": None,
        "graph": {},
        "wiki": None,
        "metadata": {}             # 元数据
    }

    # ==================== Adaptive-RAG 检索阶段 ====================
    chat_input = user_input
    use_retrieval = False
    rag_metadata = {}

    if use_adaptive_rag and model is not None:
        try:
            # 获取 RAG 引擎
            engine = init_rag_engine()
            
            # 处理查询，获取检索上下文
            context = engine.process(user_input, history)
            
            # 更新 RAG 元数据
            rag_metadata = {
                "question_type": context.retrieval_plan.question_type.value if context.retrieval_plan else None,
                "confidence": context.retrieval_plan.confidence if context.retrieval_plan else 0,
                "sources_used": context.retrieval_result.total_sources_used if context.retrieval_result else 0,
                "total_time": f"{context.total_time:.2f}s",
                "stages": context.stage_history,
                "final_action": context.final_action.value,
                "use_retrieval": context.use_retrieval,
                # CoT 相关元数据
                "use_cot": context.retrieval_plan.use_cot if context.retrieval_plan else False,
                "cot_mode": context.retrieval_plan.cot_mode if context.retrieval_plan else "direct",
                "reasoning_depth": context.retrieval_plan.reasoning_depth if context.retrieval_plan else 0,
                # LangChain 集成标记
                "langchain_enabled": use_langchain
            }
            
            # 添加检索上下文到返回数据
            base_result["rag_context"] = {
                "sources": list(context.retrieval_result.results.keys()) if context.retrieval_result else [],
                "triples_count": len(context.retrieval_result.triples) if context.retrieval_result else 0,
                "docs_count": len(context.retrieval_result.documents) if context.retrieval_result else 0,
                "has_wiki": context.retrieval_result.wiki_summary is not None if context.retrieval_result else False,
                "has_image": context.retrieval_result.image_url is not None if context.retrieval_result else False,
                # CoT 信息
                "use_cot": context.retrieval_plan.use_cot if context.retrieval_plan else False,
                "cot_mode": context.retrieval_plan.cot_mode if context.retrieval_plan else "direct",
                "reasoning_steps": len(context.reasoning_chain.steps) if context.reasoning_chain else 0,
                # LangChain 信息
                "langchain_enabled": use_langchain
            }
            
            # 添加评估报告 (如果启用)
            if enable_evaluation and context.evaluation_report:
                base_result["evaluation"] = context.evaluation_report.get_summary()
                base_result["metadata"]["evaluation"] = context.evaluation_report.get_summary()
            
            # 添加 CoT 推理链信息到元数据
            if context.reasoning_chain and enable_cot:
                base_result["metadata"]["cot"] = {
                    "mode": context.reasoning_chain.mode.value,
                    "steps": len(context.reasoning_chain.steps),
                    "depth": context.reasoning_chain.depth
                }
            
            # 添加到元数据
            base_result["metadata"]["rag"] = rag_metadata
            
            # 添加引用溯源信息 (新增)
            if include_citations and context.citation_set:
                CitationEmbedder = None
                try:
                    from app.rag.citation import CitationEmbedder as CitationEmbedderClass
                    CitationEmbedder = CitationEmbedderClass
                except ImportError:
                    pass
                
                if CitationEmbedder:
                    citation_embedder = CitationEmbedder(format_type="superscript")
                    base_result["citations"] = citation_embedder.format_citations_for_api(
                        context.citation_set.citations
                    )
                    print(f"[Citation] 返回 {len(context.citation_set.citations)} 条引用")
            
            # 根据决策决定是否使用检索结果
            if context.use_retrieval and context.assembled_prompt:
                chat_input = context.assembled_prompt
                use_retrieval = True
                print(f"[Adaptive-RAG] 使用检索增强，问题类型: {rag_metadata['question_type']}")
                if rag_metadata["use_cot"]:
                    print(f"[CoT] 启用思维链推理，模式: {rag_metadata['cot_mode']}")
            else:
                print(f"[Adaptive-RAG] 无需检索，直接生成")
                
        except Exception as e:
            print(f"[Adaptive-RAG 错误] {e}")
            # 出错时回退到原始查询
            base_result["metadata"]["rag_error"] = str(e)
    
    # ==================== LangChain RAG Chain (可选) ====================
    if use_langchain and not use_retrieval:
        try:
            adapter = init_langchain_adapter()
            
            # 使用 LangChain RAG Chain
            # 注意：需要 LLM，可以通过 adapter 设置
            # 这里展示 LangChain 集成的能力，实际使用时需要配置 LLM
            base_result["metadata"]["langchain_note"] = "LangChain 适配器已就绪，需要配置 LLM"
            print("[LangChain] RAG Chain 已准备就绪")
            
        except Exception as e:
            print(f"[LangChain 错误] {e}")
            base_result["metadata"]["langchain_error"] = str(e)
    
    # ==================== 备用检索方案 (当 Adaptive-RAG 未启用或失败时) ====================
    if not use_retrieval and model is not None:
        try:
            ref = _fallback_retrieval(user_input)
            if ref:
                chat_input = (
                    f"\n===参考资料===\n{ref}；\n\n"
                    f"根据上面资料，用简洁且准确的话回答下面问题：\n{user_input}"
                )
                use_retrieval = True
        except Exception as e:
            print(f"[备用检索错误] {e}")
    
    # ==================== 图像搜索 (始终执行) ====================
    image = None
    try:
        from app.search import ImageSearcher
        image_searcher = ImageSearcher()
        image = image_searcher.search(user_input)
        base_result["image"] = image
    except Exception as e:
        print(f"[图像搜索错误] {e}")
    
    # ==================== Wikipedia 搜索 (当无检索结果时) ====================
    wiki = None
    if not use_retrieval or not rag_metadata.get("has_wiki"):
        try:
            from app.nlp import Ner
            from app.search import WikiSearcher
            from opencc import OpenCC
            
            ner = Ner()
            wiki_searcher = WikiSearcher()
            cc = OpenCC('t2s')
            
            # 尝试实体搜索
            entities = ner.get_entities(
                user_input,
                etypes=["物体类", "人物类", "地点类", "组织机构类", "事件类", "世界地区类", "术语类"]
            )
            
            wiki = None
            for ent in entities + [user_input]:
                wiki = wiki_searcher.search(ent)
                if wiki is not None:
                    break
            
            if wiki:
                summary = cc.convert(wiki.summary)[:500]
                wiki = {
                    "title": cc.convert(wiki.title),
                    "summary": summary,
                }
                print(f"[Wikipedia] {wiki['title']}")
        except Exception as e:
            print(f"[Wikipedia 搜索错误] {e}")
    
    base_result["wiki"] = wiki or {"title": "无相关信息", "summary": "暂无相关描述"}
    
    # ==================== 模型生成阶段 ====================
    if model is not None:
        # 清理历史记录中的参考资料部分
        clean_history = []
        for q, r in history:
            if "===参考资料===" in q:
                q = q.split("===参考资料===")[0]
            clean_history.append((q, r))
        
        print(f"[ChatGLM] 输入: {chat_input[:100]}..." if len(chat_input) > 100 else f"[ChatGLM] 输入: {chat_input}")
        
        # 流式生成
        for response, history in model.stream_chat(tokenizer, chat_input, clean_history):
            # 构建响应
            updates = {}
            for q, r in history:
                updates["query"] = q
                updates["response"] = r
            
            result = base_result.copy()
            result["history"] = history
            result["updates"] = updates
            
            yield json.dumps(result, ensure_ascii=False).encode('utf8') + b'\n'
    else:
        # 模型未加载
        updates = {
            "query": user_input,
            "response": "模型加载中，请稍后再试"
        }
        result = base_result.copy()
        result["updates"] = updates
        result["history"] = history
        
        yield json.dumps(result, ensure_ascii=False).encode('utf8') + b'\n'


def _fallback_retrieval(user_input: str) -> str:
    """
    备用检索方案 (当 Adaptive-RAG 未启用时的简单检索)
    
    用于向后兼容和降级处理
    """
    ref = ""
    
    try:
        from app.nlp import Ner
        from app.kg import search_node_item, convert_graph_to_triples
        from app.search import VectorSearcher
        
        ner = Ner()
        
        # 1. NER 实体识别
        entities = ner.get_entities(
            user_input,
            etypes=["物体类", "人物类", "地点类", "组织机构类", "事件类", "世界地区类", "术语类"]
        )
        print(f"[备用检索] 实体: {entities}")
        
        # 2. 知识图谱三元组检索
        triples = []
        graph = {'nodes': [], 'links': [], 'sents': []}
        
        for entity in entities[:5]:
            graph = search_node_item(entity, graph if graph['nodes'] else None)
            if graph:
                triples += convert_graph_to_triples(graph, entity)
        
        # 限制数量
        MAX_TRIPLES = 10
        if len(triples) > MAX_TRIPLES:
            triples = triples[:MAX_TRIPLES]
        
        if triples:
            triples_str = "；".join([f"({t[0]} {t[1]} {t[2]})" for t in triples])
            ref += f"三元组信息：{triples_str}；"
        
        # 3. 向量数据库检索
        try:
            vector_searcher = VectorSearcher(
                collection_name="project_v1_docs",
                persist_dir="./data/vector_db"
            )
            search_results = vector_searcher.search(user_input, top_k=3)
            
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


# ==================== LangChain RAG Chain 便捷函数 ====================

def get_langchain_rag_chain(
    template_type: str = "rag",
    use_history: bool = False,
    use_cot: bool = False
):
    """
    获取 LangChain RAG Chain
    
    Args:
        template_type: 模板类型 ("rag", "citation", "cot", "comparison", "definition")
        use_history: 是否使用对话历史
        use_cot: 是否启用思维链
    
    Returns:
        LangChain RunnableSequence
    
    Note:
        需要先配置 LLM，否则返回 None
    """
    global langchain_adapter
    
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
    """
    使用 LangChain RAG Chain 处理查询
    
    Args:
        query: 用户查询
        history: 对话历史
    
    Returns:
        处理结果字典
    """
    chain = get_langchain_rag_chain()
    if chain is None:
        return {"error": "LangChain RAG Chain 未就绪"}
    
    try:
        input_data = {"question": query, "history": history or []}
        result = chain.invoke(input_data)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


# ==================== 生命周期管理 ====================

def start_model():
    """
    启动并加载 ChatGLM 模型
    
    从配置系统获取模型路径，支持本地路径和 HuggingFace Hub
    同时初始化 LangChain 组件
    """
    global model, tokenizer, init_history

    # 从配置系统获取模型路径
    model_path = settings.CHATGLM_MODEL_PATH
    
    # 如果是相对路径，转换为绝对路径
    if not os.path.isabs(model_path) and not model_path.startswith(("http://", "https://")):
        # 获取项目根目录（从 server/app/utils 向上四级）
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        abs_model_path = os.path.join(project_root, model_path.lstrip("./"))
        if os.path.exists(abs_model_path):
            model_path = abs_model_path
    
    # 检查是否是本地路径
    if os.path.exists(model_path):
        print(f"使用本地模型路径: {model_path}")
    else:
        print(f"使用 HuggingFace Hub 模型: {model_path}")
    
    import sys
    import types
    import importlib.util
    
    # 直接导入 ChatGLM 自定义模块（避免 transformers trust_remote_code 的路径问题）
    if os.path.exists(model_path):
        # 创建假的父包模块用于相对导入
        parent_module = types.ModuleType("transformers_modules")
        parent_module.__path__ = [model_path]
        sys.modules["transformers_modules"] = parent_module
        
        # 动态导入配置模块
        config_path = os.path.join(model_path, "configuration_chatglm.py")
        config_spec = importlib.util.spec_from_file_location("transformers_modules.configuration_chatglm", config_path)
        config_module = importlib.util.module_from_spec(config_spec)
        config_spec.loader.exec_module(config_module)
        sys.modules["transformers_modules.configuration_chatglm"] = config_module
        
        # 动态导入 tokenizer 模块
        tokenizer_path = os.path.join(model_path, "tokenization_chatglm.py")
        tokenizer_spec = importlib.util.spec_from_file_location("transformers_modules.tokenization_chatglm", tokenizer_path)
        tokenizer_module = importlib.util.module_from_spec(tokenizer_spec)
        tokenizer_spec.loader.exec_module(tokenizer_module)
        sys.modules["transformers_modules.tokenization_chatglm"] = tokenizer_module
        
        # 动态导入模型模块
        modeling_path = os.path.join(model_path, "modeling_chatglm.py")
        modeling_spec = importlib.util.spec_from_file_location("transformers_modules.modeling_chatglm", modeling_path)
        modeling_module = importlib.util.module_from_spec(modeling_spec)
        modeling_spec.loader.exec_module(modeling_module)
        sys.modules["transformers_modules.modeling_chatglm"] = modeling_module
        
        # 使用自定义模块加载
        ChatGLMConfig = config_module.ChatGLMConfig
        ChatGLMTokenizer = tokenizer_module.ChatGLMTokenizer
        ChatGLMForConditionalGeneration = modeling_module.ChatGLMForConditionalGeneration
        
        # 加载配置和模型
        config = ChatGLMConfig.from_pretrained(model_path)
        tokenizer = ChatGLMTokenizer.from_pretrained(model_path)
        model = ChatGLMForConditionalGeneration.from_pretrained(model_path, config=config)
    else:
        # 从 HuggingFace Hub 加载（使用 trust_remote_code）
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    
    # 检查 CUDA 是否可用
    import torch
    if torch.cuda.is_available():
        print("使用 GPU (CUDA)")
        model = model.half().cuda()
    else:
        print("使用 CPU（注意：CPU 模式速度较慢）")
        model = model.float()
    
    model.eval()

    # 初始化对话
    pre_prompt = "你叫 ChatKG，是一个图谱问答机器人，此为背景。下面开始聊天吧！"
    _, history = predict(pre_prompt, [])
    init_history = history
    
    # 预热 RAG 引擎
    print("预热 Adaptive-RAG 引擎...")
    _ = init_rag_engine()
    print("Adaptive-RAG 引擎就绪")
    
    # 预热 LangChain 适配器 (可选)
    print("预热 LangChain 组件...")
    try:
        _ = init_langchain_adapter()
        print("LangChain 组件就绪")
    except Exception as e:
        print(f"[警告] LangChain 组件初始化失败: {e}")


def get_rag_stats() -> dict:
    """获取 RAG 统计信息"""
    global rag_engine
    if rag_engine:
        return rag_engine.get_stats()
    return {}


def reset_rag_stats():
    """重置 RAG 统计信息"""
    global rag_engine
    if rag_engine:
        rag_engine.reset_stats()


def get_langchain_stats() -> dict:
    """获取 LangChain 统计信息"""
    global langchain_adapter
    if langchain_adapter:
        return {"adapter_ready": True}
    return {"adapter_ready": False}
