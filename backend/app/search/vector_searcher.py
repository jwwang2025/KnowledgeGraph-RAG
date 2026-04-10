import os
import chromadb
from chromadb.config import Settings


class VectorSearcher:
    """ChromaDB 向量数据库检索器"""
    
    def __init__(self, collection_name="knowledge_base", persist_dir="./data/vector_db"):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self._init_client()
        
    def _init_client(self):
        """初始化 ChromaDB 客户端"""
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
    def encoder(self):
        """延迟加载编码器模型"""
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        return self._encoder
        
    def get_embedding(self, texts):
        """获取文本的向量表示"""
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.encoder.encode(texts)
        return embeddings.tolist() if hasattr(embeddings, 'tolist') else list(embeddings)
        
    def add_documents(self, texts, ids, metadatas=None):
        """批量添加文档到向量数据库"""
        if not texts:
            return
            
        embeddings = self.get_embedding(texts)
        metadatas = metadatas or [{}] * len(texts)
        
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            ids=ids,
            metadatas=metadatas
        )
        
    def add_document(self, text, doc_id, metadata=None):
        """添加单个文档"""
        self.add_documents([text], [doc_id], [metadata] if metadata else None)
        
    def search(self, query, top_k=5, where=None, where_document=None):
        """语义检索相关文档
        
        Args:
            query: 查询文本
            top_k: 返回数量
            where: 元数据过滤条件
            where_document: 文档内容过滤条件
            
        Returns:
            dict: 包含 documents, distances, metadatas, ids
        """
        query_embedding = self.get_embedding(query)
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            where=where,
            where_document=where_document,
            include=["documents", "distances", "metadatas", "ids"]
        )
        
        return self._format_results(results)
        
    def similarity_search(self, query, top_k=5, threshold=0.7):
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
                    if key in results and results[key] and i < len(results[key][0]):
                        filtered_results[key].append(results[key][0][i])
                        
        return filtered_results
        
    def _format_results(self, results):
        """格式化���索结果"""
        if not results or not results.get('ids'):
            return {
                "documents": [],
                "distances": [],
                "metadatas": [],
                "ids": []
            }
            
        formatted = {
            "documents": results.get('documents', [[]]),
            "distances": results.get('distances', [[]]),
            "metadatas": results.get('metadatas', [[]]),
            "ids": results.get('ids', [[]])
        }
        
        return formatted
        
    def delete_collection(self):
        """删除 collection"""
        self.client.delete_collection(self.collection_name)
        self._init_client()
        
    def get_collection_info(self):
        """获取 collection 信息"""
        return {
            "name": self.collection.name,
            "count": self.collection.count(),
            "metadata": self.collection.metadata
        }
        
    def reset(self):
        """重置数据库"""
        self.client.delete_collection(self.collection_name)
        self._init_client()
