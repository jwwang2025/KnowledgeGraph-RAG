"""
向量数据库检索器 (Vector Searcher)
基于 ChromaDB 的语义检索，支持 LangChain 集成

使用 LangChain 的 Chroma 封装，提供标准化的向量检索接口
"""

import os
from typing import List, Dict, Any, Optional, Union, Tuple, Iterable
from langchain_core.documents import Document

# LangChain Chroma
from langchain_chroma import Chroma

# 导入 LangChain 封装层
from app.rag.langchain_components import (
    SentenceTransformerEmbeddings,
    Qwen3Embeddings,
    KnowledgeGraphVectorStore,
)


class VectorSearcher:
    """
    ChromaDB 向量数据库检索器
    
    支持:
    - SentenceTransformer 嵌入
    - Qwen3-Embedding-8B 嵌入
    - LangChain 兼容接口
    - 多表征技术
    - 层级向量索引
    """
    
    def __init__(
        self,
        collection_name: str = "knowledge_base",
        persist_dir: str = "./data/vector_db",
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        local_model_path: Optional[str] = None,
        embedding_type: str = "sentence_transformer"
    ):
        """
        初始化 VectorSearcher
        
        Args:
            collection_name: 集合名称
            persist_dir: 持久化目录
            model_name: 嵌入模型名称
            local_model_path: 本地模型路径
            embedding_type: 嵌入类型 ("sentence_transformer" / "qwen3")
        """
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.model_name = model_name
        self.local_model_path = local_model_path
        self.embedding_type = embedding_type
        
        # 初始化嵌入函数
        self._embedding_function = None
        self._vectorstore = None
        self._langchain_store = None
        
        # 初始化 ChromaDB 客户端 (兼容旧接口)
        self._init_chroma_client()
    
    def _init_chroma_client(self):
        """初始化 ChromaDB 客户端 (兼容旧接口)"""
        import chromadb
        from chromadb.config import Settings
        
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Knowledge Graph RAG vector store"}
        )
        self._encoder = None
    
    @property
    def embedding_function(self):
        """获取 LangChain 兼容的嵌入函数"""
        if self._embedding_function is None:
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
        """获取嵌入编码器 (兼容旧接口)"""
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
                elif model_path:
                    print(f"[警告] 本地模型不存在: {model_path}，将尝试在线下载")
                    self._encoder = SentenceTransformer(self.model_name)
                else:
                    self._encoder = SentenceTransformer(self.model_name)
        return self._encoder
    
    @property
    def langchain_vectorstore(self) -> Chroma:
        """获取 LangChain Chroma 向量存储"""
        if self._langchain_store is None:
            self._langchain_store = Chroma(
                client=self.client,
                collection_name=self.collection_name,
                embedding_function=self.embedding_function
            )
        return self._langchain_store
    
    @property
    def kg_vectorstore(self) -> KnowledgeGraphVectorStore:
        """获取知识图谱增强的 VectorStore"""
        if self._vectorstore is None:
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
        """
        批量添加文档到向量数据库
        
        Args:
            texts: 文档文本列表
            ids: 文档 ID 列表
            metadatas: 元数据列表
            batch_size: 每批处理的文档数量
        
        Returns:
            添加的文档 ID 列表
        """
        if not texts:
            return []
        
        # 使用 LangChain Chroma 接口
        documents = [
            Document(page_content=text, metadata=metadata or {})
            for text, metadata in zip(texts, metadatas or [{}] * len(texts))
        ] if metadatas else [
            Document(page_content=text)
            for text in texts
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
        """
        语义检索相关文档
        
        Args:
            query: 查询文本
            top_k: 返回数量
            where: 元数据过滤条件
            where_document: 文档内容过滤条件
            filter: 统一的过滤条件 (LangChain 风格)
        
        Returns:
            dict: 包含 documents, distances, metadatas, ids
        """
        # 统一过滤条件
        filter_condition = filter or where
        
        # 使用 LangChain Chroma 检索
        if filter_condition:
            results = self.langchain_vectorstore.similarity_search_with_score(
                query, k=top_k, filter=filter_condition
            )
        else:
            results = self.langchain_vectorstore.similarity_search_with_score(
                query, k=top_k
            )
        
        # 格式化为旧接口格式
        documents = []
        distances = []
        metadatas = []
        ids = []
        
        for doc, score in results:
            documents.append(doc.page_content)
            distances.append(1.0 - score)  # 转换分数为距离
            metadatas.append(doc.metadata)
            ids.append(doc.metadata.get("doc_id", ""))
        
        return {
            "documents": [documents] if documents else [[]],  # 兼容旧格式
            "distances": [distances] if distances else [[]],
            "metadatas": [metadatas] if metadatas else [[]],
            "ids": [ids] if ids else [[]]
        }
    
    def similarity_search(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.7
    ) -> Dict[str, Any]:
        """相似度检索，过滤低相似度结果"""
        results = self.search(query, top_k)
        
        filtered_results = {
            "documents": [],
            "distances": [],
            "metadatas": [],
            "ids": []
        }
        
        for i, dist in enumerate(results.get("distances", [[]])[0]):
            if dist <= threshold:
                for key in filtered_results:
                    if key in results and results[key] and results[key][0] and i < len(results[key][0]):
                        filtered_results[key].append(results[key][0][i])
        
        return filtered_results
    
    def similarity_search_langchain(
        self,
        query: str,
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        LangChain 风格的相似度检索
        
        Args:
            query: 查询文本
            top_k: 返回数量
            filter: 过滤条件
        
        Returns:
            Document 列表
        """
        if filter:
            return self.langchain_vectorstore.similarity_search(
                query, k=top_k, filter=filter
            )
        return self.langchain_vectorstore.similarity_search(query, k=top_k)
    
    def as_retriever(
        self,
        search_type: str = "similarity",
        search_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        转换为 LangChain Retriever
        
        Args:
            search_type: 检索类型 ("similarity", "mmr")
            search_kwargs: 检索参数
            **kwargs: 其他参数
        
        Returns:
            Retriever 实例
        """
        search_kwargs = search_kwargs or {}
        search_kwargs.setdefault("k", 5)
        
        return self.langchain_vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs,
            **kwargs
        )
    
    def get_langchain_chain(self, prompt_template: Optional[str] = None):
        """
        获取 LangChain RAG Chain (需要配合 LLM 使用)
        
        Args:
            prompt_template: 提示模板
        
        Returns:
            RAGChainBuilder 实例
        """
        from app.rag.langchain_components import RAGChainBuilder, RAGPromptTemplates
        
        # 注意：这里需要传入 LLM，可以在调用时设置
        class LazyRAGChainBuilder(RAGChainBuilder):
            def __init__(self, *args, **kwargs):
                super().__init__(llm=None, retriever=None, prompt_template=prompt_template)
        
        return LazyRAGChainBuilder(
            retriever=self.as_retriever(),
            prompt_template=prompt_template or RAGPromptTemplates.RAG_TEMPLATE
        )
    
    def delete_collection(self):
        """删除 collection"""
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


# =============================================================================
# 便捷函数
# =============================================================================

def create_vector_searcher(
    collection_name: str = "knowledge_base",
    persist_dir: str = "./data/vector_db",
    embedding_type: str = "sentence_transformer",
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
) -> VectorSearcher:
    """
    创建 VectorSearcher 实例
    
    Args:
        collection_name: 集合名称
        persist_dir: 持久化目录
        embedding_type: 嵌入类型
        model_name: 模型名称
    
    Returns:
        VectorSearcher 实例
    """
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
    """
    直接创建 LangChain Chroma 实例
    
    Args:
        collection_name: 集合名称
        persist_dir: 持久化目录
        embedding_function: 嵌入函数
    
    Returns:
        Chroma 实例
    """
    if embedding_function is None:
        embedding_function = SentenceTransformerEmbeddings()
    
    return Chroma(
        collection_name=collection_name,
        persist_directory=persist_dir,
        embedding_function=embedding_function
    )
