"""
层级向量索引模块
支持 Sentence、Chunk、Document 三级索引
分布式 ChromaDB 存储
"""
import os
import json
import uuid
import hashlib
from typing import List, Dict, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import chromadb
from chromadb.config import Settings
from chromadb.api.models.Collection import Collection
import numpy as np

from .qwen3_embedding import Qwen3EmbeddingEncoder, MultiRepresentationBuilder
from .qwen3_embedding import HybridEncoder


@dataclass
class IndexMetadata:
    """索引元数据"""
    index_id: str
    index_type: str  # "sentence", "chunk", "document"
    representation_type: str  # "dense", "sparse", "hybrid"
    created_at: str
    document_count: int
    total_vectors: int
    embedding_dim: int
    parent_index_id: Optional[str] = None
    children_index_ids: List[str] = field(default_factory=list)
    extra_info: Dict[str, Any] = field(default_factory=dict)


class HierarchicalVectorIndex:
    """
    层级向量索引
    
    支持三级索引结构:
    - Level 1: Sentence (句子级)
    - Level 2: Chunk (分块级)
    - Level 3: Document (文档级)
    
    支持多种表征类型:
    - dense: 稠密向量 (Qwen3-Embedding)
    - sparse: 稀疏向量 (BM25)
    - hybrid: 混合向量
    """
    
    def __init__(
        self,
        project_name: str = "project_v1",
        persist_dir: str = "./data/vector_db",
        encoder: Optional[Qwen3EmbeddingEncoder] = None,
        enable_multi_representation: bool = True,
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ):
        """
        初始化层级向量索引
        
        Args:
            project_name: 项目名称
            persist_dir: 持久化存储目录
            encoder: 向量编码器
            enable_multi_representation: 是否启用多表征
            chunk_size: 分块大小
            chunk_overlap: 块间重叠
        """
        self.project_name = project_name
        self.persist_dir = persist_dir
        self.encoder = encoder or Qwen3EmbeddingEncoder()
        self.enable_multi_representation = enable_multi_representation
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.multi_repr_builder = MultiRepresentationBuilder(self.encoder)
        
        # 索引层级映射
        self.levels = {
            "sentence": 1,
            "chunk": 2,
            "document": 3
        }
        
        # ChromaDB 客户端和 Collection 映射
        self._client = None
        self._collections = {}
        self._index_metadata = {}
        
        # 索引目录
        self.index_dir = os.path.join(persist_dir, project_name, "hierarchical_index")
        self.metadata_dir = os.path.join(self.index_dir, "metadata")
        
    @property
    def client(self) -> chromadb.PersistentClient:
        """获取 ChromaDB 客户端"""
        if self._client is None:
            os.makedirs(self.persist_dir, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=Settings(anonymized_telemetry=False)
            )
        return self._client
        
    def _get_collection_name(self, level: str, repr_type: str = "dense") -> str:
        """获取 Collection 名称"""
        return f"{self.project_name}_{level}_{repr_type}"
        
    def _get_or_create_collection(
        self,
        level: str,
        repr_type: str = "dense",
        metadata: Optional[Dict] = None
    ) -> Collection:
        """获取或创建 Collection"""
        name = self._get_collection_name(level, repr_type)
        
        if name not in self._collections:
            collection_meta = {
                "level": level,
                "representation_type": repr_type,
                "project": self.project_name,
                "created_at": datetime.now().isoformat()
            }
            if metadata:
                collection_meta.update(metadata)
                
            self._collections[name] = self.client.get_or_create_collection(
                name=name,
                metadata=collection_meta
            )
            
        return self._collections[name]
        
    def _generate_doc_id(self, level: str, content_hash: str) -> str:
        """生成文档 ID"""
        prefix_map = {"sentence": "sent", "chunk": "chunk", "document": "doc"}
        prefix = prefix_map.get(level, "item")
        return f"{prefix}_{content_hash[:16]}"
        
    def _compute_content_hash(self, content: str) -> str:
        """计算内容哈希"""
        return hashlib.md5(content.encode()).hexdigest()
        
    def _split_into_sentences(self, text: str) -> List[str]:
        """将文本分割为句子"""
        import re
        # 按中文标点分句
        sentences = re.split(r'[。！？；\n]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
        
    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """将文本分块"""
        if len(text) <= chunk_size:
            return [text]
            
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
            
        return chunks
        
    def index_document(
        self,
        document: str,
        doc_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        index_level: str = "all",
        representations: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        索引单个文档
        
        Args:
            document: 文档文本
            doc_id: 文档 ID（可选，自动生成）
            metadata: 元数据
            index_level: 索引层级 ("all", "sentence", "chunk", "document")
            representations: 表征类型列表
            
        Returns:
            Dict: 索引结果
        """
        doc_id = doc_id or f"doc_{uuid.uuid4().hex[:12]}"
        metadata = metadata or {}
        metadata["doc_id"] = doc_id
        metadata["indexed_at"] = datetime.now().isoformat()
        
        result = {
            "doc_id": doc_id,
            "levels": {}
        }
        
        # 确定要索引的层级
        if index_level == "all":
            levels_to_index = ["sentence", "chunk", "document"]
        else:
            levels_to_index = [index_level]
            
        # 1. 句子级索引
        if "sentence" in levels_to_index:
            sentences = self._split_into_sentences(document)
            if sentences:
                sent_result = self._index_level(
                    "sentence",
                    sentences,
                    doc_id,
                    metadata,
                    representations
                )
                result["levels"]["sentence"] = sent_result
                
        # 2. 分块级索引
        if "chunk" in levels_to_index:
            chunks = self._chunk_text(document, self.chunk_size, self.chunk_overlap)
            if chunks:
                chunk_result = self._index_level(
                    "chunk",
                    chunks,
                    doc_id,
                    metadata,
                    representations
                )
                result["levels"]["chunk"] = chunk_result
                
        # 3. 文档级索引
        if "document" in levels_to_index:
            doc_result = self._index_level(
                "document",
                [document],
                doc_id,
                metadata,
                representations
            )
            result["levels"]["document"] = doc_result
            
        return result
        
    def _index_level(
        self,
        level: str,
        items: List[str],
        doc_id: str,
        metadata: Dict,
        representations: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """索引特定层级的项目"""
        result = {
            "count": len(items),
            "items": []
        }
        
        # 默认表征类型
        if representations is None:
            representations = ["original"]
            
        for i, item in enumerate(items):
            item_id = f"{doc_id}_{level}_{i}"
            item_meta = {
                **metadata,
                "level": level,
                "item_index": i,
                "item_id": item_id
            }
            
            # 获取嵌入
            if self.enable_multi_representation and len(representations) > 1:
                # 多表征
                embeddings_dict = self.encoder.encode_multi_representation(
                    item,
                    representations
                )
                
                for repr_type, embedding in embeddings_dict.items():
                    collection = self._get_or_create_collection(level, repr_type)
                    
                    collection.add(
                        embeddings=embedding.tolist(),
                        documents=[item],
                        ids=[item_id],
                        metadatas=[item_meta]
                    )
                    
                result["items"].append({
                    "item_id": item_id,
                    "representations": list(embeddings_dict.keys())
                })
            else:
                # 单表征
                embedding = self.encoder.encode(item)
                repr_type = representations[0] if representations else "original"
                
                collection = self._get_or_create_collection(level, repr_type)
                
                collection.add(
                    embeddings=embedding.tolist(),
                    documents=[item],
                    ids=[item_id],
                    metadatas=[item_meta]
                )
                
                result["items"].append({
                    "item_id": item_id,
                    "representations": [repr_type]
                })
                
        return result
        
    def index_documents_batch(
        self,
        documents: List[str],
        metadata_list: Optional[List[Dict]] = None,
        index_level: str = "all",
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """
        批量索引文档
        
        Args:
            documents: 文档列表
            metadata_list: 元数据列表
            index_level: 索引层级
            batch_size: 批处理大小
            
        Returns:
            Dict: 索引结果
        """
        if metadata_list is None:
            metadata_list = [{} for _ in documents]
            
        total_results = {
            "total_documents": len(documents),
            "results": []
        }
        
        for i, doc in enumerate(documents):
            doc_id = f"doc_{uuid.uuid4().hex[:12]}"
            result = self.index_document(
                doc,
                doc_id=doc_id,
                metadata=metadata_list[i] if i < len(metadata_list) else {},
                index_level=index_level
            )
            total_results["results"].append(result)
            
            if (i + 1) % batch_size == 0:
                print(f"[进度] 已索引 {i + 1}/{len(documents)} 文档")
                
        return total_results
        
    def search(
        self,
        query: str,
        top_k: int = 5,
        level: str = "chunk",
        representation: str = "original",
        filter_metadata: Optional[Dict] = None,
        search_across_levels: bool = True
    ) -> Dict[str, Any]:
        """
        搜索向量索引
        
        Args:
            query: 查询文本
            top_k: 返回数量
            level: 搜索层级
            representation: 表征类型
            filter_metadata: 元数据过滤条件
            search_across_levels: 是否跨层级搜索
            
        Returns:
            Dict: 搜索结果
        """
        # 获取查询向量
        query_embedding = self.encoder.encode(query)
        
        results = {}
        
        if search_across_levels:
            # 跨层级搜索
            for lvl in ["sentence", "chunk", "document"]:
                if level != "all" and lvl != level:
                    continue
                    
                collection_name = self._get_collection_name(lvl, representation)
                try:
                    collection = self.client.get_collection(collection_name)
                    
                    search_results = collection.query(
                        query_embeddings=query_embedding.tolist(),
                        n_results=top_k,
                        where=filter_metadata,
                        include=["documents", "distances", "metadatas", "ids"]
                    )
                    
                    if search_results["ids"]:
                        results[lvl] = {
                            "ids": search_results["ids"][0],
                            "documents": search_results["documents"][0],
                            "distances": search_results["distances"][0],
                            "metadatas": search_results["metadatas"][0]
                        }
                except Exception as e:
                    print(f"[警告] 搜索层级 {lvl} 失败: {e}")
        else:
            # 单层级搜索
            collection_name = self._get_collection_name(level, representation)
            try:
                collection = self.client.get_collection(collection_name)
                
                search_results = collection.query(
                    query_embeddings=query_embedding.tolist(),
                    n_results=top_k,
                    where=filter_metadata,
                    include=["documents", "distances", "metadatas", "ids"]
                )
                
                if search_results["ids"]:
                    results[level] = {
                        "ids": search_results["ids"][0],
                        "documents": search_results["documents"][0],
                        "distances": search_results["distances"][0],
                        "metadatas": search_results["metadatas"][0]
                    }
            except Exception as e:
                print(f"[警告] 搜索失败: {e}")
                
        return results
        
    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.7,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        混合搜索：结合稠密和稀疏向量
        
        Args:
            query: 查询文本
            top_k: 返回数量
            alpha: 稠密向量权重 (1-alpha 为稀疏向量权重)
            filter_metadata: 元数据过滤
            
        Returns:
            List: 混合搜索结果
        """
        # 获取混合编码器
        hybrid_encoder = HybridEncoder(self.encoder)
        
        # 获取稠密和稀疏向量
        hybrid_result = hybrid_encoder.encode_hybrid(query, return_sparse=True)
        
        # 从各层级获取结果
        all_results = []
        
        for level in ["sentence", "chunk", "document"]:
            collection_name = self._get_collection_name(level, "dense")
            try:
                dense_collection = self.client.get_collection(collection_name)
                
                # 获取稠密向量搜索结果
                dense_results = dense_collection.query(
                    query_embeddings=hybrid_result["dense"].tolist(),
                    n_results=top_k * 2,
                    where=filter_metadata,
                    include=["documents", "distances", "metadatas", "ids"]
                )
                
                # 融合稀疏向量分数
                if dense_results["ids"]:
                    # 计算综合分数
                    for i, (doc_id, doc, dist, meta) in enumerate(zip(
                        dense_results["ids"][0],
                        dense_results["documents"][0],
                        dense_results["distances"][0],
                        dense_results["metadatas"][0]
                    )):
                        # 距离转换为相似度
                        dense_score = 1.0 / (1.0 + dist)
                        
                        # 简化的稀疏分数计算
                        sparse_score = self._compute_sparse_similarity(
                            query, doc
                        )
                        
                        # 混合分数
                        final_score = alpha * dense_score + (1 - alpha) * sparse_score
                        
                        all_results.append({
                            "doc_id": doc_id,
                            "document": doc,
                            "distance": dist,
                            "dense_score": dense_score,
                            "sparse_score": sparse_score,
                            "final_score": final_score,
                            "level": level,
                            "metadata": meta
                        })
            except Exception as e:
                print(f"[警告] 搜索层级 {level} 失败: {e}")
                
        # 按最终分数排序
        all_results.sort(key=lambda x: x["final_score"], reverse=True)
        
        return all_results[:top_k]
        
    def _compute_sparse_similarity(self, query: str, document: str) -> float:
        """计算稀疏向量相似度（简化 BM25）"""
        try:
            import jieba
            query_terms = set(jieba.cut(query))
            doc_terms = set(jieba.cut(document))
            
            # Jaccard 相似度
            intersection = len(query_terms & doc_terms)
            union = len(query_terms | doc_terms)
            
            return intersection / union if union > 0 else 0.0
        except:
            return 0.0
            
    def get_collection_info(self, level: str, representation: str = "original") -> Dict:
        """获取 Collection 信息"""
        collection_name = self._get_collection_name(level, representation)
        try:
            collection = self.client.get_collection(collection_name)
            return {
                "name": collection.name,
                "count": collection.count(),
                "metadata": collection.metadata
            }
        except Exception as e:
            return {"error": str(e)}
            
    def delete_collection(self, level: str, representation: str = "original"):
        """删除 Collection"""
        collection_name = self._get_collection_name(level, representation)
        try:
            self.client.delete_collection(collection_name)
            if collection_name in self._collections:
                del self._collections[collection_name]
            print(f"[完成] 已删除 Collection: {collection_name}")
        except Exception as e:
            print(f"[警告] 删除 Collection 失败: {e}")
            
    def reset(self, confirm: bool = False):
        """重置所有索引"""
        if not confirm:
            print("[警告] 此操作将删除所有索引数据，请确认 (confirm=True)")
            return
            
        for level in ["sentence", "chunk", "document"]:
            for repr_type in ["original", "keywords", "entities", "triples", "summary", "query"]:
                self.delete_collection(level, repr_type)
                
        print("[完成] 所有索引已重置")
        
    def save_index_metadata(self):
        """保存索引元数据"""
        os.makedirs(self.metadata_dir, exist_ok=True)
        
        metadata = {
            "project_name": self.project_name,
            "saved_at": datetime.now().isoformat(),
            "config": {
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "enable_multi_representation": self.enable_multi_representation
            },
            "collections": {}
        }
        
        for level in ["sentence", "chunk", "document"]:
            collection_info = self.get_collection_info(level)
            if "error" not in collection_info:
                metadata["collections"][level] = collection_info
                
        metadata_path = os.path.join(self.metadata_dir, "index_metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
            
        print(f"[完成] 索引元数据已保存: {metadata_path}")
        

class DistributedVectorIndex:
    """
    分布式向量索引
    
    支持多个 ChromaDB 实例的分片存储
    """
    
    def __init__(
        self,
        project_name: str = "project_v1",
        base_dir: str = "./data/vector_db",
        num_shards: int = 4
    ):
        """
        初始化分布式向量索引
        
        Args:
            project_name: 项目名称
            base_dir: 基础存储目录
            num_shards: 分片数量
        """
        self.project_name = project_name
        self.base_dir = base_dir
        self.num_shards = num_shards
        self.shards = []
        
        # 初始化各分片
        for i in range(num_shards):
            shard_dir = os.path.join(base_dir, f"shard_{i}")
            os.makedirs(shard_dir, exist_ok=True)
            
            shard = HierarchicalVectorIndex(
                project_name=f"{project_name}_shard_{i}",
                persist_dir=shard_dir
            )
            self.shards.append(shard)
            
    def _get_shard_index(self, key: str) -> int:
        """根据键获取分片索引"""
        return hash(key) % self.num_shards
        
    def index_document(
        self,
        document: str,
        doc_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        shard_key: Optional[str] = None
    ) -> Dict:
        """索引文档到指定分片"""
        if shard_key is None:
            shard_key = doc_id or str(uuid.uuid4())
            
        shard_idx = self._get_shard_index(shard_key)
        shard = self.shards[shard_idx]
        
        return shard.index_document(document, doc_id, metadata)
        
    def search(
        self,
        query: str,
        top_k: int = 5,
        search_all_shards: bool = True,
        **kwargs
    ) -> Dict:
        """搜索所有分片"""
        if search_all_shards:
            # 并行搜索所有分片
            import concurrent.futures
            
            all_results = []
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_shards) as executor:
                futures = {
                    executor.submit(shard.search, query, top_k * 2, **kwargs): i
                    for i, shard in enumerate(self.shards)
                }
                
                for future in concurrent.futures.as_completed(futures):
                    results = future.result()
                    for level, level_results in results.items():
                        if level_results["documents"]:
                            all_results.extend([
                                {**r, "shard": futures[future]}
                                for r in zip(
                                    level_results["documents"],
                                    level_results["distances"],
                                    level_results["ids"],
                                    level_results["metadatas"]
                                )
                            ])
                            
            # 合并并排序
            all_results.sort(key=lambda x: x.get("distance", float("inf")))
            return {"results": all_results[:top_k]}
        else:
            # 搜索单个分片
            shard_idx = self._get_shard_index(query)
            return self.shards[shard_idx].search(query, top_k, **kwargs)
            
    def get_shard_info(self) -> List[Dict]:
        """获取所有分片信息"""
        return [
            {
                "shard_id": i,
                "shard_dir": shard.persist_dir,
                "collections": shard.get_collection_info("chunk")
            }
            for i, shard in enumerate(self.shards)
        ]


def create_hierarchical_index(
    data_source: str,
    project_name: str = "project_v1",
    persist_dir: str = "./data/vector_db",
    **kwargs
) -> HierarchicalVectorIndex:
    """
    从数据源创建层级向量索引
    
    Args:
        data_source: 数据源路径（JSON 文件或目录）
        project_name: 项目名称
        persist_dir: 持久化目录
        **kwargs: 其他参数
        
    Returns:
        HierarchicalVectorIndex: 层级向量索引实例
    """
    index = HierarchicalVectorIndex(
        project_name=project_name,
        persist_dir=persist_dir,
        **kwargs
    )
    
    # 加载数据
    if os.path.isfile(data_source):
        with open(data_source, "r", encoding="utf-8") as f:
            if data_source.endswith(".json"):
                data = json.load(f)
            elif data_source.endswith(".jsonl"):
                data = [json.loads(line) for line in f]
            else:
                data = [line.strip() for line in f if line.strip()]
    else:
        # 目录中的所有文本文件
        data = []
        for filename in os.listdir(data_source):
            filepath = os.path.join(data_source, filename)
            if os.path.isfile(filepath):
                with open(filepath, "r", encoding="utf-8") as f:
                    data.extend([line.strip() for line in f if line.strip()])
                    
    # 索引数据
    if isinstance(data, list):
        if data and isinstance(data[0], dict):
            documents = [item.get("text", item.get("content", str(item))) for item in data]
            metadata_list = [item for item in data if isinstance(item, dict)]
        else:
            documents = data
            metadata_list = None
            
        index.index_documents_batch(documents, metadata_list)
    else:
        index.index_document(str(data))
        
    # 保存元数据
    index.save_index_metadata()
    
    return index


if __name__ == "__main__":
    # 测试层级索引
    index = HierarchicalVectorIndex(
        project_name="test_project",
        persist_dir="./data/vector_db"
    )
    
    # 测试文档索引
    documents = [
        "知识图谱是一种用图来表达实体和它们之间关系的技术。",
        "人工智能是计算机科学的一个分支。",
        "机器学习是人工智能的一个子领域。"
    ]
    
    result = index.index_documents_batch(documents)
    print(f"索引结果: {result}")
    
    # 测试搜索
    search_result = index.search("什么是知识图谱？", top_k=2)
    print(f"搜索结果: {search_result}")
