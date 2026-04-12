"""
Search 模块 - 多种检索源适配器

包含:
- vector_searcher: 向量数据库检索 (ChromaDB)
- wiki_searcher: Wikipedia 搜索
- image_searcher: 图像搜索
- qwen3_embedding: Qwen3-Embedding-8B 向量化编码器
- hierarchical_index: 层级向量索引
"""

from .vector_searcher import VectorSearcher
from .wiki_searcher import WikiSearcher
from .image_searcher import ImageSearcher
from .qwen3_embedding import (
    Qwen3EmbeddingEncoder,
    HybridEncoder,
    MultiRepresentationBuilder,
    download_qwen3_embedding_model
)
from .hierarchical_index import (
    HierarchicalVectorIndex,
    DistributedVectorIndex,
    create_hierarchical_index
)

__all__ = [
    # 基础检索器
    'VectorSearcher',
    'WikiSearcher',
    'ImageSearcher',
    # Qwen3-Embedding
    'Qwen3EmbeddingEncoder',
    'HybridEncoder',
    'MultiRepresentationBuilder',
    'download_qwen3_embedding_model',
    # 层级索引
    'HierarchicalVectorIndex',
    'DistributedVectorIndex',
    'create_hierarchical_index',
]
