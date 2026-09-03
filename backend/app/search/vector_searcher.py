"""向量数据库检索器 (Vector Searcher)：基于 ChromaDB + LangChain 的语义检索。"""
import os
from typing import List, Dict, Any, Optional, Union

import chromadb
from chromadb.config import Settings
from langchain_core.documents import Document
from langchain_chroma import Chroma

# 模块级 ChromaDB 客户端缓存：同一持久化目录全局复用一个客户端
_CLIENT_CACHE: Dict[str, chromadb.PersistentClient] = {}


def _get_chroma_client(persist_dir: str) -> chromadb.PersistentClient:
    """获取（或创建）ChromaDB PersistentClient"""
    if persist_dir not in _CLIENT_CACHE:
        os.makedirs(persist_dir, exist_ok=True)
        _CLIENT_CACHE[persist_dir] = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
    return _CLIENT_CACHE[persist_dir]


class VectorSearcher:
    """ChromaDB 向量数据库检索器，支持 SentenceTransformer / Qwen3 嵌入与 LangChain 接口。"""

    def __init__(
        self,
        collection_name: str = "knowledge_base",
        persist_dir: str = "./data/vector_db",
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        local_model_path: Optional[str] = None,
        embedding_type: str = "sentence_transformer"
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.model_name = model_name
        self.local_model_path = local_model_path
        self.embedding_type = embedding_type

        self._embedding_function = None
        self._vectorstore = None
        self._langchain_store = None

        # ChromaDB 客户端与 Collection（复用，不随查询重建）
        self._init_chroma_client()

    def _init_chroma_client(self):
        """初始化 ChromaDB 客户端 (兼容旧接口)"""
        self.client = _get_chroma_client(self.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Knowledge Graph RAG vector store"}
        )
        self._encoder = None

    @property
    def embedding_function(self):
        """LangChain 兼容的嵌入函数（懒加载）"""
        if self._embedding_function is None:
            # 延迟导入，避免与 app.rag 的循环依赖
            from app.rag.langchain_components import SentenceTransformerEmbeddings, Qwen3Embeddings
            if self.embedding_type == "qwen3":
                self._embedding_function = Qwen3Embeddings(
                    local_model_path=self.local_model_path
                )
            else:
                self._embedding_function = SentenceTransformerEmbeddings(
                    model_name=self.model_name,
                    local_model_path=self.local_model_path
                )
        return self._embedding_function

    @property
    def encoder(self):
        """嵌入编码器 (兼容旧接口，懒加载)"""
        if self._encoder is None:
            if self.embedding_type == "qwen3":
                from app.search.qwen3_embedding import Qwen3EmbeddingEncoder
                self._encoder = Qwen3EmbeddingEncoder(
                    local_model_path=self.local_model_path
                )
            else:
                from sentence_transformers import SentenceTransformer
                model_path = self.local_model_path
                if model_path and os.path.exists(model_path):
                    self._encoder = SentenceTransformer(model_path)
                else:
                    if model_path:
                        print(f"[警告] 本地模型不存在: {model_path}，将尝试在线下载")
                    self._encoder = SentenceTransformer(self.model_name)
        return self._encoder

    @property
    def langchain_vectorstore(self) -> Chroma:
        """LangChain Chroma 向量存储（懒加载）"""
        if self._langchain_store is None:
            self._langchain_store = Chroma(
                client=self.client,
                collection_name=self.collection_name,
                embedding_function=self.embedding_function
            )
        return self._langchain_store

    @property
    def kg_vectorstore(self) -> "KnowledgeGraphVectorStore":
        """知识图谱增强的 VectorStore（懒加载）"""
        if self._vectorstore is None:
            from app.rag.langchain_components import KnowledgeGraphVectorStore
            self._vectorstore = KnowledgeGraphVectorStore(
                collection_name=self.collection_name,
                persist_dir=self.persist_dir,
                embedding_function=self.embedding_function,
                client=self.client
            )
        return self._vectorstore

    def get_embedding(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """获取文本的向量表示 (兼容旧接口)"""
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.encoder.encode(texts)
        return embeddings.tolist() if hasattr(embeddings, 'tolist') else list(embeddings)

    def add_documents(
        self,
        texts: List[str],
        ids: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 1000
    ) -> List[str]:
        """批量添加文档到向量数据库"""
        if not texts:
            return []

        docs_metadatas = metadatas if metadatas else [{}] * len(texts)
        documents = [
            Document(page_content=text, metadata=metadata or {})
            for text, metadata in zip(texts, docs_metadatas)
        ]
        return self.langchain_vectorstore.add_documents(documents, ids=ids)

    def add_document(
        self,
        text: str,
        doc_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """添加单个文档"""
        doc = Document(page_content=text, metadata=metadata or {})
        ids = self.langchain_vectorstore.add_documents([doc], ids=[doc_id] if doc_id else None)
        return ids[0] if ids else None

    def search(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """语义检索，返回 documents/distances/metadatas/ids"""
        filter_condition = filter or where

        kwargs = {"filter": filter_condition} if filter_condition else {}
        results = self.langchain_vectorstore.similarity_search_with_score(query, k=top_k, **kwargs)

        documents, distances, metadatas, ids = [], [], [], []
        for doc, score in results:
            documents.append(doc.page_content)
            distances.append(1.0 - score)  # 分数转距离
            metadatas.append(doc.metadata)
            ids.append(doc.metadata.get("doc_id", ""))

        # 兼容旧格式：单查询结果包裹一层列表
        return {
            "documents": [documents],
            "distances": [distances],
            "metadatas": [metadatas],
            "ids": [ids],
        }

    def similarity_search(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.7
    ) -> Dict[str, Any]:
        """相似度检索，过滤距离超过 threshold 的结果"""
        results = self.search(query, top_k)

        filtered_results = {"documents": [], "distances": [], "metadatas": [], "ids": []}
        for doc, dist, meta, doc_id in zip(
            results.get("documents", [[]])[0],
            results.get("distances", [[]])[0],
            results.get("metadatas", [[]])[0],
            results.get("ids", [[]])[0],
        ):
            if dist <= threshold:
                filtered_results["documents"].append(doc)
                filtered_results["distances"].append(dist)
                filtered_results["metadatas"].append(meta)
                filtered_results["ids"].append(doc_id)

        return filtered_results

    def similarity_search_langchain(
        self,
        query: str,
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """LangChain 风格的相似度检索"""
        kwargs = {"filter": filter} if filter else {}
        return self.langchain_vectorstore.similarity_search(query, k=top_k, **kwargs)

    def as_retriever(
        self,
        search_type: str = "similarity",
        search_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """转换为 LangChain Retriever"""
        search_kwargs = search_kwargs or {}
        search_kwargs.setdefault("k", 5)

        return self.langchain_vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs,
            **kwargs
        )

    def get_langchain_chain(self, prompt_template: Optional[str] = None):
        """获取 LangChain RAG Chain（LLM 需在调用时配置）"""
        from app.rag.langchain_components import RAGChainBuilder, RAGPromptTemplates

        class LazyRAGChainBuilder(RAGChainBuilder):
            def __init__(self, *args, **kwargs):
                super().__init__(llm=None, retriever=None, prompt_template=prompt_template)

        return LazyRAGChainBuilder(
            retriever=self.as_retriever(),
            prompt_template=prompt_template or RAGPromptTemplates.RAG_TEMPLATE
        )

    def delete_collection(self):
        """删除 collection 并重建客户端引用"""
        self.client.delete_collection(self.collection_name)
        self._init_chroma_client()
        self._langchain_store = None
        self._vectorstore = None

    def get_collection_info(self) -> Dict[str, Any]:
        """获取 collection 信息"""
        return {
            "name": self.collection.name,
            "count": self.collection.count(),
            "metadata": self.collection.metadata
        }

    def reset(self):
        """重置数据库"""
        self.delete_collection()


def create_vector_searcher(
    collection_name: str = "knowledge_base",
    persist_dir: str = "./data/vector_db",
    embedding_type: str = "sentence_transformer",
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
) -> VectorSearcher:
    """创建 VectorSearcher 实例"""
    return VectorSearcher(
        collection_name=collection_name,
        persist_dir=persist_dir,
        embedding_type=embedding_type,
        model_name=model_name
    )


def create_langchain_chroma(
    collection_name: str = "knowledge_base",
    persist_dir: str = "./data/vector_db",
    embedding_function: Optional[Any] = None
) -> Chroma:
    """直接创建 LangChain Chroma 实例"""
    if embedding_function is None:
        from app.rag.langchain_components import SentenceTransformerEmbeddings
        embedding_function = SentenceTransformerEmbeddings()

    return Chroma(
        collection_name=collection_name,
        persist_directory=persist_dir,
        embedding_function=embedding_function
    )
