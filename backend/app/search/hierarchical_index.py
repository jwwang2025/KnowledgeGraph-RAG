"""层级向量索引模块：Sentence/Chunk/Document 三级索引，支持分布式 ChromaDB 存储。"""
import os
import json
import uuid
import hashlib
import concurrent.futures
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

import chromadb
from chromadb.config import Settings
from chromadb.api.models.Collection import Collection

from .qwen3_embedding import (
    Qwen3EmbeddingEncoder,
    MultiRepresentationBuilder,
    HybridEncoder,
    split_sentences,
    chunk_text,
)

# 查询时使用的 include 字段（避免重复字面量）
_QUERY_INCLUDE = ["documents", "distances", "metadatas", "ids"]


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
    层级向量索引：三级结构（sentence/chunk/document），
    支持 dense/sparse/hybrid 多种表征。
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
        self.project_name = project_name
        self.persist_dir = persist_dir
        self.encoder = encoder or Qwen3EmbeddingEncoder()
        self.enable_multi_representation = enable_multi_representation
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.multi_repr_builder = MultiRepresentationBuilder(self.encoder)

        self.levels = {"sentence": 1, "chunk": 2, "document": 3}

        self._client = None
        self._collections: Dict[str, Collection] = {}
        self._index_metadata = {}
        self._hybrid_encoder = None

        self.index_dir = os.path.join(persist_dir, project_name, "hierarchical_index")
        self.metadata_dir = os.path.join(self.index_dir, "metadata")

    @property
    def client(self) -> chromadb.PersistentClient:
        """懒加载 ChromaDB 客户端（实例内复用）"""
        if self._client is None:
            os.makedirs(self.persist_dir, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=Settings(anonymized_telemetry=False)
            )
        return self._client

    def _get_collection_name(self, level: str, repr_type: str = "dense") -> str:
        return f"{self.project_name}_{level}_{repr_type}"

    def _get_or_create_collection(
        self,
        level: str,
        repr_type: str = "dense",
        metadata: Optional[Dict] = None
    ) -> Collection:
        """获取或创建 Collection（实例内缓存复用）"""
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
        prefix = {"sentence": "sent", "chunk": "chunk", "document": "doc"}.get(level, "item")
        return f"{prefix}_{content_hash[:16]}"

    @staticmethod
    def _compute_content_hash(content: str) -> str:
        return hashlib.md5(content.encode()).hexdigest()

    @staticmethod
    def _split_into_sentences(text: str) -> List[str]:
        """按中文标点分句（含分号）"""
        return split_sentences(text, r'[。！？；\n]+')

    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        return chunk_text(text, chunk_size, overlap)

    def index_document(
        self,
        document: str,
        doc_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        index_level: str = "all",
        representations: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """索引单个文档，可选层级 ("all"/"sentence"/"chunk"/"document")"""
        doc_id = doc_id or f"doc_{uuid.uuid4().hex[:12]}"
        metadata = metadata or {}
        metadata["doc_id"] = doc_id
        metadata["indexed_at"] = datetime.now().isoformat()

        result = {"doc_id": doc_id, "levels": {}}

        levels_to_index = ["sentence", "chunk", "document"] if index_level == "all" else [index_level]
        level_texts = {
            "sentence": self._split_into_sentences(document) if "sentence" in levels_to_index else None,
            "chunk": self._chunk_text(document, self.chunk_size, self.chunk_overlap) if "chunk" in levels_to_index else None,
            "document": [document] if "document" in levels_to_index else None,
        }

        for level in ["sentence", "chunk", "document"]:
            items = level_texts[level]
            if items:
                result["levels"][level] = self._index_level(
                    level, items, doc_id, metadata, representations
                )

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
        if representations is None:
            representations = ["original"]

        result = {"count": len(items), "items": []}

        for i, item in enumerate(items):
            item_id = f"{doc_id}_{level}_{i}"
            item_meta = {
                **metadata,
                "level": level,
                "item_index": i,
                "item_id": item_id
            }

            if self.enable_multi_representation and len(representations) > 1:
                embeddings_dict = self.encoder.encode_multi_representation(item, representations)
            else:
                repr_type = representations[0] if representations else "original"
                embeddings_dict = {repr_type: self.encoder.encode(item)}

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

        return result

    def index_documents_batch(
        self,
        documents: List[str],
        metadata_list: Optional[List[Dict]] = None,
        index_level: str = "all",
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """批量索引文档"""
        if metadata_list is None:
            metadata_list = [{} for _ in documents]

        total_results = {"total_documents": len(documents), "results": []}

        for i, doc in enumerate(documents):
            result = self.index_document(
                doc,
                doc_id=f"doc_{uuid.uuid4().hex[:12]}",
                metadata=metadata_list[i] if i < len(metadata_list) else {},
                index_level=index_level
            )
            total_results["results"].append(result)

            if (i + 1) % batch_size == 0:
                print(f"[进度] 已索引 {i + 1}/{len(documents)} 文档")

        return total_results

    def _query_collection(self, collection_name: str, query_embedding, top_k: int, filter_metadata: Optional[Dict]):
        """查询单个 Collection，失败时返回 None"""
        try:
            collection = self.client.get_collection(collection_name)
            search_results = collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=top_k,
                where=filter_metadata,
                include=_QUERY_INCLUDE
            )
            if search_results["ids"]:
                return {
                    "ids": search_results["ids"][0],
                    "documents": search_results["documents"][0],
                    "distances": search_results["distances"][0],
                    "metadatas": search_results["metadatas"][0]
                }
        except Exception as e:
            print(f"[警告] 查询 {collection_name} 失败: {e}")
        return None

    def search(
        self,
        query: str,
        top_k: int = 5,
        level: str = "chunk",
        representation: str = "original",
        filter_metadata: Optional[Dict] = None,
        search_across_levels: bool = True
    ) -> Dict[str, Any]:
        """搜索向量索引（支持跨层级）"""
        query_embedding = self.encoder.encode(query)
        results = {}

        levels = ["sentence", "chunk", "document"] if search_across_levels else [level]
        for lvl in levels:
            if search_across_levels and level != "all" and lvl != level:
                continue

            level_result = self._query_collection(
                self._get_collection_name(lvl, representation),
                query_embedding, top_k, filter_metadata
            )
            if level_result:
                results[lvl] = level_result

        return results

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.7,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """混合搜索：稠密向量检索 + 简化 BM25 稀疏分数融合"""
        if self._hybrid_encoder is None:
            self._hybrid_encoder = HybridEncoder(self.encoder)
        hybrid_result = self._hybrid_encoder.encode_hybrid(query, return_sparse=True)

        all_results = []
        for level in ["sentence", "chunk", "document"]:
            collection_name = self._get_collection_name(level, "dense")
            try:
                dense_collection = self.client.get_collection(collection_name)
                dense_results = dense_collection.query(
                    query_embeddings=hybrid_result["dense"].tolist(),
                    n_results=top_k * 2,
                    where=filter_metadata,
                    include=_QUERY_INCLUDE
                )

                if dense_results["ids"]:
                    for doc_id, doc, dist, meta in zip(
                        dense_results["ids"][0],
                        dense_results["documents"][0],
                        dense_results["distances"][0],
                        dense_results["metadatas"][0]
                    ):
                        dense_score = 1.0 / (1.0 + dist)
                        sparse_score = self._compute_sparse_similarity(query, doc)
                        all_results.append({
                            "doc_id": doc_id,
                            "document": doc,
                            "distance": dist,
                            "dense_score": dense_score,
                            "sparse_score": sparse_score,
                            "final_score": alpha * dense_score + (1 - alpha) * sparse_score,
                            "level": level,
                            "metadata": meta
                        })
            except Exception as e:
                print(f"[警告] 搜索层级 {level} 失败: {e}")

        all_results.sort(key=lambda x: x["final_score"], reverse=True)
        return all_results[:top_k]

    @staticmethod
    def _compute_sparse_similarity(query: str, document: str) -> float:
        """简化 BM25 相似度（jieba 分词 + Jaccard）"""
        try:
            import jieba
            query_terms = set(jieba.cut(query))
            doc_terms = set(jieba.cut(document))
            intersection = len(query_terms & doc_terms)
            union = len(query_terms | doc_terms)
            return intersection / union if union > 0 else 0.0
        except Exception:
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
            self._collections.pop(collection_name, None)
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
    """分布式向量索引：多个 ChromaDB 实例的分片存储"""

    def __init__(
        self,
        project_name: str = "project_v1",
        base_dir: str = "./data/vector_db",
        num_shards: int = 4
    ):
        self.project_name = project_name
        self.base_dir = base_dir
        self.num_shards = num_shards
        self.shards = []

        for i in range(num_shards):
            shard_dir = os.path.join(base_dir, f"shard_{i}")
            os.makedirs(shard_dir, exist_ok=True)
            self.shards.append(HierarchicalVectorIndex(
                project_name=f"{project_name}_shard_{i}",
                persist_dir=shard_dir
            ))

    def _get_shard_index(self, key: str) -> int:
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

        shard = self.shards[self._get_shard_index(shard_key)]
        return shard.index_document(document, doc_id, metadata)

    def search(
        self,
        query: str,
        top_k: int = 5,
        search_all_shards: bool = True,
        **kwargs
    ) -> Dict:
        """搜索分片（可并行搜索全部或按 key 定位单个）"""
        if not search_all_shards:
            return self.shards[self._get_shard_index(query)].search(query, top_k, **kwargs)

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

        all_results.sort(key=lambda x: x.get("distance", float("inf")))
        return {"results": all_results[:top_k]}

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
    """从数据源（JSON/JSONL/文本文件/目录）创建层级向量索引"""
    index = HierarchicalVectorIndex(
        project_name=project_name,
        persist_dir=persist_dir,
        **kwargs
    )

    if os.path.isfile(data_source):
        with open(data_source, "r", encoding="utf-8") as f:
            if data_source.endswith(".json"):
                data = json.load(f)
            elif data_source.endswith(".jsonl"):
                data = [json.loads(line) for line in f]
            else:
                data = [line.strip() for line in f if line.strip()]
    else:
        data = []
        for filename in os.listdir(data_source):
            filepath = os.path.join(data_source, filename)
            if os.path.isfile(filepath):
                with open(filepath, "r", encoding="utf-8") as f:
                    data.extend([line.strip() for line in f if line.strip()])

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

    index.save_index_metadata()
    return index
