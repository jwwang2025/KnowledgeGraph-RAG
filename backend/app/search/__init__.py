"""Search 模块：向量检索 / Wikipedia / 图像搜索 / 向量化编码 / 层级索引。"""

from .vector_searcher import VectorSearcher, create_vector_searcher, create_langchain_chroma
from .wiki_searcher import WikiSearcher, WikipediaDocumentLoader, create_wikipedia_retriever, search_wikipedia
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
# LangChain 集成
from app.rag.langchain_components import (
    SentenceTransformerEmbeddings,
    Qwen3Embeddings,
    KnowledgeGraphVectorStore,
    create_langchain_vectorstore,
    create_qwen3_vectorstore,
)

__all__ = [
    # 基础检索器
    'VectorSearcher', 'WikiSearcher', 'ImageSearcher',
    # 创建函数
    'create_vector_searcher', 'create_langchain_chroma',
    'create_wikipedia_retriever', 'search_wikipedia', 'WikipediaDocumentLoader',
    # Qwen3-Embedding
    'Qwen3EmbeddingEncoder', 'HybridEncoder', 'MultiRepresentationBuilder',
    'download_qwen3_embedding_model',
    # 层级索引
    'HierarchicalVectorIndex', 'DistributedVectorIndex', 'create_hierarchical_index',
    # LangChain 集成
    'SentenceTransformerEmbeddings', 'Qwen3Embeddings', 'KnowledgeGraphVectorStore',
    'create_langchain_vectorstore', 'create_qwen3_vectorstore',
]
