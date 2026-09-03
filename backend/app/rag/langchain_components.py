"""LangChain 封装层：Embeddings / VectorStore / Retriever / RAG Chain 组件与工具函数。"""

import os
import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    Runnable,
    RunnablePassthrough,
    RunnableLambda,
    RunnableSequence,
)
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun


# =============================================================================
# Part 1: Embeddings 封装层
# =============================================================================

class SentenceTransformerEmbeddings(Embeddings):
    """LangChain 兼容的 SentenceTransformer Embeddings 封装。"""

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        local_model_path: Optional[str] = None,
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
        encode_kwargs: Optional[Dict[str, Any]] = None
    ):
        self.model_name = model_name
        self.local_model_path = local_model_path
        self.normalize_embeddings = normalize_embeddings
        self.encode_kwargs = encode_kwargs or {}
        self._model = None
        self._device = device

    @property
    def model(self):
        """延迟加载模型"""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            model_path = self.local_model_path
            if model_path and os.path.exists(model_path):
                self._model = SentenceTransformer(model_path)
            elif model_path:
                print(f"[警告] 本地模型不存在: {model_path}，将尝试在线下载")
                self._model = SentenceTransformer(self.model_name)
            else:
                self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=True,
            **self.encode_kwargs
        )
        return embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings

    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode(
            text,
            normalize_embeddings=self.normalize_embeddings,
            **self.encode_kwargs
        )
        return embedding.tolist() if hasattr(embedding, 'tolist') else embedding

    def __call__(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)


class Qwen3Embeddings(Embeddings):
    """LangChain 兼容的 Qwen3-Embedding 封装，支持多表征嵌入。"""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-8B",
        local_model_path: Optional[str] = None,
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
        batch_size: int = 8,
        max_length: int = 8192
    ):
        self.model_name = model_name
        self.local_model_path = local_model_path
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self.max_length = max_length
        self._encoder = None

        # 自动选择设备
        if device is None:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    @property
    def encoder(self):
        """延迟加载编码器"""
        if self._encoder is None:
            from app.search.qwen3_embedding import Qwen3EmbeddingEncoder
            self._encoder = Qwen3EmbeddingEncoder(
                model_name=self.model_name,
                local_model_path=self.local_model_path,
                device=self.device,
                normalize_embeddings=self.normalize_embeddings,
                batch_size=self.batch_size,
                max_length=self.max_length
            )
        return self._encoder

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.encoder.encode(texts)
        return embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings

    def embed_query(self, text: str) -> List[float]:
        embedding = self.encoder.encode(text, prompt_name="asymmetric")
        result = embedding[0] if len(embedding.shape) > 1 else embedding
        return result.tolist() if hasattr(result, 'tolist') else result

    def embed_query_multi_representation(self, text: str) -> Dict[str, List[float]]:
        results = self.encoder.encode_multi_representation(text)
        return {k: v.tolist() if hasattr(v, 'tolist') else v
                for k, v in results.items()}


# =============================================================================
# Part 2: 自定义 VectorStore
# =============================================================================

class KnowledgeGraphVectorStore(VectorStore):
    """知识图谱增强的 VectorStore：同时支持向量检索与图谱数据。"""

    def __init__(
        self,
        collection_name: str = "knowledge_base",
        persist_dir: str = "./data/vector_db",
        embedding_function: Optional[Embeddings] = None,
        client: Any = None
    ):
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self.embedding_function = embedding_function
        self._client = client
        self._collection = None

        # 初始化 ChromaDB
        if client is None:
            import chromadb
            from chromadb.config import Settings
            self._client = chromadb.PersistentClient(
                path=persist_dir,
                settings=Settings(anonymized_telemetry=False)
            )

        # 获取或创建集合
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Knowledge Graph RAG vector store"}
        )

    @property
    def embeddings(self):
        return self.embedding_function

    def _docs_from_results(self, results: Dict[str, Any]) -> List[Document]:
        """将 ChromaDB 查询结果转换为 Document 列表"""
        documents = []
        if results.get("documents") and results["documents"][0]:
            metadatas = results.get("metadatas")
            for i, doc in enumerate(results["documents"][0]):
                metadata = metadatas[0][i] if metadatas else {}
                documents.append(Document(page_content=doc, metadata=metadata))
        return documents

    def _query_by_embedding(self, embedding: List[float], k: int,
                            filter: Optional[Dict[str, Any]],
                            include: List[str]) -> Dict[str, Any]:
        return self._collection.query(
            query_embeddings=[embedding],
            n_results=k,
            where=filter,
            include=include
        )

    def add_texts(
        self,
        texts: Iterable[str],
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        **kwargs
    ) -> List[str]:
        """添加文本到向量库"""
        texts = list(texts)
        ids = ids or [f"doc_{i}" for i in range(len(texts))]

        # 如果没有提供嵌入，使用 embedding_function
        if embeddings is None and self.embedding_function is not None:
            embeddings = self.embedding_function.embed_documents(texts)

        self._collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

        return ids

    def add_documents(self, documents: List[Document], **kwargs) -> List[str]:
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return self.add_texts(texts, metadatas=metadatas, **kwargs)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Document]:
        """相似度搜索"""
        embedding = self.embedding_function.embed_query(query)
        results = self._query_by_embedding(
            embedding, k, filter, ["documents", "metadatas", "distances"]
        )
        return self._docs_from_results(results)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Tuple[Document, float]]:
        """带分数的相似度搜索（距离转换为相似度分数）"""
        embedding = self.embedding_function.embed_query(query)
        results = self._query_by_embedding(
            embedding, k, filter, ["documents", "metadatas", "distances"]
        )

        documents = self._docs_from_results(results)
        distances = results.get("distances", [[]])[0] if results.get("distances") else []
        return [
            (doc, 1.0 / (1.0 + (distances[i] if i < len(distances) else 0.0)))
            for i, doc in enumerate(documents)
        ]

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Document]:
        """通过向量搜索"""
        results = self._query_by_embedding(
            embedding, k, filter, ["documents", "metadatas"]
        )
        return self._docs_from_results(results)

    def delete(self, ids: Optional[List[str]] = None, **kwargs) -> None:
        if ids:
            self._collection.delete(ids=ids)

    @classmethod
    def fromTexts(
        cls,
        texts: Iterable[str],
        embedding: Optional[Embeddings] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        **kwargs
    ) -> "KnowledgeGraphVectorStore":
        """从文本创建 VectorStore"""
        persist_dir = kwargs.get("persist_dir", "./data/vector_db")
        collection_name = kwargs.get("collection_name", "knowledge_base")

        vectorstore = cls(
            collection_name=collection_name,
            persist_dir=persist_dir,
            embedding_function=embedding
        )

        vectorstore.add_texts(texts, metadatas=metadatas, ids=ids)
        return vectorstore

    @classmethod
    def fromDocuments(
        cls,
        documents: List[Document],
        embedding: Optional[Embeddings] = None,
        **kwargs
    ) -> "KnowledgeGraphVectorStore":
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return cls.fromTexts(texts, embedding, metadatas=metadatas, **kwargs)

    def as_retriever(self, **kwargs) -> BaseRetriever:
        return KnowledgeGraphRetriever(vectorstore=self, **kwargs)

    @property
    def collection(self):
        return self._collection


class KnowledgeGraphRetriever(BaseRetriever):
    """知识图谱增强的 Retriever：结合向量检索和知识图谱三元组。"""

    vectorstore: KnowledgeGraphVectorStore
    search_type: str = "similarity"
    k: int = 4
    score_threshold: Optional[float] = None

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs
    ) -> List[Document]:
        if self.search_type == "mmr":
            documents = self.vectorstore.max_marginal_relevance_search(query, k=self.k, **kwargs)
        else:
            documents = self.vectorstore.similarity_search(query, k=self.k)

        # 可选：按分数过滤
        if self.score_threshold:
            if self.search_type == "similarity":
                docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=self.k)
                documents = [doc for doc, score in docs_with_scores if score >= self.score_threshold]

        return documents


# =============================================================================
# Part 3: 自定义 Retriever
# =============================================================================

def _mark_source(documents: List[Document], source: str, weight: float) -> List[Document]:
    """为文档标注来源与权重"""
    for doc in documents:
        doc.metadata["source"] = source
        doc.metadata["weight"] = weight
    return documents


class MultiSourceRetriever(BaseRetriever):
    """多源检索器：支持向量数据库、知识图谱、Wikipedia。"""

    vector_retriever: Optional[BaseRetriever] = None
    kg_retriever: Optional[Callable] = None
    wiki_wrapper: Optional[WikipediaAPIWrapper] = None

    k: int = 4
    source_weights: Dict[str, float] = field(default_factory=lambda: {
        "vector": 0.4,
        "kg": 0.4,
        "wiki": 0.2
    })

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs
    ) -> List[Document]:
        all_documents = []

        # 1. 向量数据库检索
        if self.vector_retriever:
            try:
                vector_docs = self.vector_retriever.get_relevant_documents(query, **kwargs)
                all_documents.extend(_mark_source(
                    vector_docs, "vector", self.source_weights.get("vector", 0.4)
                ))
            except Exception as e:
                print(f"[MultiSourceRetriever] 向量检索失败: {e}")

        # 2. 知识图谱检索
        if self.kg_retriever:
            try:
                kg_docs = self.kg_retriever(query, top_k=self.k)
                all_documents.extend(_mark_source(
                    kg_docs, "kg", self.source_weights.get("kg", 0.4)
                ))
            except Exception as e:
                print(f"[MultiSourceRetriever] 知识图谱检索失败: {e}")

        # 3. Wikipedia 检索
        if self.wiki_wrapper:
            try:
                wiki_result = self.wiki_wrapper.run(query)
                if wiki_result:
                    all_documents.append(Document(
                        page_content=wiki_result,
                        metadata={"source": "wiki", "weight": self.source_weights.get("wiki", 0.2)}
                    ))
            except Exception as e:
                print(f"[MultiSourceRetriever] Wikipedia 检索失败: {e}")

        # 按权重排序并限制数量
        all_documents.sort(key=lambda x: x.metadata.get("weight", 0), reverse=True)
        return all_documents[:self.k]


# =============================================================================
# Part 4: 自定义 RAG Chain 组件
# =============================================================================

@dataclass
class RetrievalResult:
    """LangChain 风格的检索结果"""
    documents: List[Document]
    source_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def _format_docs(docs: List[Document]) -> str:
    """格式化文档列表为上下文字符串"""
    return "\n\n".join([doc.page_content for doc in docs])


class RAGFusionRetriever(BaseRetriever):
    """RRF (Reciprocal Rank Fusion) 检索器：融合多个检索器结果。"""

    retrievers: List[BaseRetriever]
    k: int = 60  # RRF 平滑因子

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs
    ) -> List[Document]:
        # 获取所有检索器结果
        all_results: Dict[str, List[Tuple[int, Document]]] = {}

        for i, retriever in enumerate(self.retrievers):
            try:
                docs = retriever.get_relevant_documents(query, **kwargs)
                all_results[f"retriever_{i}"] = list(enumerate(docs))
            except Exception as e:
                print(f"[RRF Retriever] 检索器 {i} 失败: {e}")
                all_results[f"retriever_{i}"] = []

        # RRF 融合（文档内容哈希作为唯一标识）
        fused_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        for results in all_results.values():
            for rank, (_, doc) in enumerate(results):
                doc_key = hashlib.md5(doc.page_content.encode()).hexdigest()
                fused_scores[doc_key] = fused_scores.get(doc_key, 0.0) + 1.0 / (self.k + rank + 1)
                doc_map[doc_key] = doc

        # 按分数排序，返回 top 10
        sorted_keys = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)

        fused_documents = []
        for doc_key in sorted_keys[:10]:
            doc = doc_map[doc_key]
            doc.metadata["fusion_score"] = fused_scores[doc_key]
            fused_documents.append(doc)

        return fused_documents


# =============================================================================
# Part 5: 提示模板
# =============================================================================

class RAGPromptTemplates:
    """RAG 场景使用的提示模板"""

    RAG_TEMPLATE = """基于以下参考资料回答问题。如果参考资料中没有相关信息，请说明你不知道。

===参考资料===
{context}

===问题===
{question}

===回答==="""

    CITATION_TEMPLATE = """基于以下参考资料回答问题，并在回答中标注引用的来源。

===参考资料===
{context}

===问题===
{question}

===回答===
（请在回答中的相关陈述后标注来源，例如：[来源1: 知识图谱]）
"""

    COT_TEMPLATE = """让我们一步步思考：

===参考资料===
{context}

===问题===
{question}

===推理过程===
1. 首先，理解问题：{question}
2. 分析参考资料中相关信息
3. 逐步推理得出结论
4. 最终回答
"""

    COMPARISON_TEMPLATE = """基于以下参考资料，比较分析所问的事物。

===参考资料===
{context}

===问题===
{question}

请从以下角度进行比较：
1. 基本定义/概念
2. 主要特征/特点
3. 优缺点对比
4. 适用场景
"""

    DEFINITION_TEMPLATE = """基于以下参考资料，给出准确的定义和解释。

===参考资料===
{context}

===问题===
{question}

请按照以下格式回答：
【定义】
【特征】
【例子】
"""

    _TEMPLATES = {
        "rag": RAG_TEMPLATE,
        "citation": CITATION_TEMPLATE,
        "cot": COT_TEMPLATE,
        "comparison": COMPARISON_TEMPLATE,
        "definition": DEFINITION_TEMPLATE,
    }

    @classmethod
    def get_template(cls, template_type: str = "rag") -> str:
        """获取指定类型的模板"""
        return cls._TEMPLATES.get(template_type, cls.RAG_TEMPLATE)

    @classmethod
    def create_prompt_template(cls, template_type: str = "rag") -> PromptTemplate:
        """创建 LangChain PromptTemplate"""
        return PromptTemplate.from_template(cls.get_template(template_type))


# =============================================================================
# Part 6: RAG Chain 构建器
# =============================================================================

class RAGChainBuilder:
    """RAG Chain 构建器：基于 LangChain Runnable 接口构建 RAG 流程。"""

    def __init__(
        self,
        llm: BaseLanguageModel,
        retriever: BaseRetriever,
        prompt_template: Optional[str] = None,
        output_key: str = "answer"
    ):
        self.llm = llm
        self.retriever = retriever
        self.prompt_template = prompt_template or self.RAG_TEMPLATE
        self.output_key = output_key

        self._prompt = PromptTemplate.from_template(self.prompt_template)

    def build_basic_chain(self) -> RunnableSequence:
        """构建基础 RAG Chain"""
        return (
            RunnablePassthrough.assign(context=lambda x: _format_docs(x["context"]))
            | self._prompt
            | self.llm
        )

    def build_with_history(self) -> RunnableSequence:
        """构建带历史记录的 RAG Chain"""
        def get_history(x: Dict) -> str:
            history = x.get("history", [])
            if not history:
                return "没有对话历史"
            return "\n".join([f"用户: {q}\n助手: {r}" for q, r in history[-3:]])

        prompt = PromptTemplate.from_template(
            self.prompt_template + "\n\n===对话历史===\n{history}"
        )

        return (
            RunnablePassthrough.assign(
                context=lambda x: _format_docs(x["context"]),
                history=lambda x: get_history(x)
            )
            | prompt
            | self.llm
        )

    def build_cot_chain(self) -> RunnableSequence:
        """构建 CoT RAG Chain"""
        prompt = PromptTemplate.from_template(RAGPromptTemplates.COT_TEMPLATE)

        return (
            RunnablePassthrough.assign(context=lambda x: _format_docs(x["context"]))
            | prompt
            | self.llm
        )

    def build_multi_stage_chain(
        self,
        question_router: Optional[Runnable] = None,
        result_evaluator: Optional[Runnable] = None
    ) -> RunnableSequence:
        """构建多阶段 RAG Chain：路由 -> 检索 -> 评估 -> 生成"""
        stages = []

        if question_router:
            stages.append(question_router)

        # 检索阶段
        stages.append(RunnableLambda(
            lambda x: {"context": self.retriever.get_relevant_documents(x["question"])}
        ))

        if result_evaluator:
            stages.append(result_evaluator)

        # 生成阶段
        stages.append(
            RunnablePassthrough.assign(context=lambda x: _format_docs(x["context"]))
            | self._prompt
            | self.llm
        )

        return RunnableSequence(stages)


# =============================================================================
# Part 7: 工具函数
# =============================================================================

def _create_vectorstore(
    embeddings: Embeddings,
    collection_name: str,
    persist_dir: str
) -> KnowledgeGraphVectorStore:
    """用给定嵌入模型创建 KnowledgeGraphVectorStore"""
    return KnowledgeGraphVectorStore(
        collection_name=collection_name,
        persist_dir=persist_dir,
        embedding_function=embeddings
    )


def create_langchain_vectorstore(
    collection_name: str = "knowledge_base",
    persist_dir: str = "./data/vector_db",
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
    local_model_path: Optional[str] = None
) -> KnowledgeGraphVectorStore:
    """创建 LangChain 兼容的向量数据库"""
    embeddings = SentenceTransformerEmbeddings(
        model_name=embedding_model,
        local_model_path=local_model_path
    )
    return _create_vectorstore(embeddings, collection_name, persist_dir)


def create_qwen3_vectorstore(
    collection_name: str = "knowledge_base",
    persist_dir: str = "./data/vector_db",
    local_model_path: Optional[str] = None
) -> KnowledgeGraphVectorStore:
    """使用 Qwen3-Embedding 创建向量数据库"""
    return _create_vectorstore(
        Qwen3Embeddings(local_model_path=local_model_path),
        collection_name,
        persist_dir
    )


def create_wikipedia_retriever(top_k_results: int = 3) -> WikipediaQueryRun:
    """创建 Wikipedia 检索工具"""
    return WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=top_k_results))


def documents_to_langchain(
    texts: List[str],
    metadatas: Optional[List[Dict[str, Any]]] = None,
    ids: Optional[List[str]] = None
) -> List[Document]:
    """将原始文本转换为 LangChain Document 对象"""
    documents = []
    for i, text in enumerate(texts):
        metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
        metadata["doc_id"] = ids[i] if ids and i < len(ids) else f"doc_{i}"
        documents.append(Document(page_content=text, metadata=metadata))
    return documents


def format_retrieval_results(
    documents: List[Document],
    include_metadata: bool = True
) -> str:
    """格式化检索结果为上下文字符串"""
    if include_metadata:
        return "\n\n".join(
            f"[{i+1}] 来源: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
            for i, doc in enumerate(documents)
        )
    return "\n\n".join(doc.page_content for doc in documents)


# =============================================================================
# Part 8: LangChain 集成适配器
# =============================================================================

class LangChainAdapter:
    """LangChain 集成适配器：将 KnowledgeGraph-RAG 组件接入 LangChain 框架。"""

    def __init__(
        self,
        llm: Optional[BaseLanguageModel] = None,
        vectorstore: Optional[KnowledgeGraphVectorStore] = None
    ):
        self.llm = llm
        self.vectorstore = vectorstore
        self._chain: Optional[RunnableSequence] = None

    def build_rag_chain(
        self,
        template_type: str = "rag",
        use_history: bool = False,
        use_cot: bool = False
    ) -> RunnableSequence:
        """构建 RAG Chain"""
        if not self.llm or not self.vectorstore:
            raise ValueError("需要先设置 llm 和 vectorstore")

        builder = RAGChainBuilder(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(),
            prompt_template=RAGPromptTemplates.get_template(template_type)
        )

        if use_cot:
            self._chain = builder.build_cot_chain()
        elif use_history:
            self._chain = builder.build_with_history()
        else:
            self._chain = builder.build_basic_chain()

        return self._chain

    def invoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        if not self._chain:
            raise ValueError("需要先调用 build_rag_chain")
        return self._chain.invoke(input_data)

    def stream(self, input_data: Dict[str, Any]) -> Iterable[str]:
        if not self._chain:
            raise ValueError("需要先调用 build_rag_chain")
        return self._chain.stream(input_data)


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    # Embeddings
    'SentenceTransformerEmbeddings',
    'Qwen3Embeddings',
    # VectorStore
    'KnowledgeGraphVectorStore',
    'KnowledgeGraphRetriever',
    # Retriever
    'MultiSourceRetriever',
    'RAGFusionRetriever',
    # Prompt Templates
    'RAGPromptTemplates',
    # Chain Builder
    'RAGChainBuilder',
    'LangChainAdapter',
    # Utilities
    'create_langchain_vectorstore',
    'create_qwen3_vectorstore',
    'create_wikipedia_retriever',
    'documents_to_langchain',
    'format_retrieval_results',
    'RetrievalResult',
]
