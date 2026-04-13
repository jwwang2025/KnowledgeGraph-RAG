"""
RAG 模块 - 检索增强生成核心组件

包含:
|- query_router: 问题路由
|- retrieval_decider: 检索决策
|- result_evaluator: 结果评估
|- cot_reasoner: 思维链推理
|- adaptive_rag_engine: 核心引擎
|- citation: 引用溯源机制
|- fusion: RRF 融合算法 (Self-RAG 多轮检索第一轮)
|- reranker: Cohere 语义重排序 (Self-RAG 多轮检索第二轮)
|- langchain_components: LangChain 集成组件
"""

from .query_router import QueryRouter, QuestionType, RetrievalPlan
from .retrieval_decider import RetrievalDecider, MultiSourceRetrievalResult, RetrievalStatus
from .result_evaluator import ResultEvaluator, EvaluationReport, ActionDecision, RelevanceLevel
from .cot_reasoner import CoTReasoner, CoTMode, ReasoningChain, ReasoningStep
from .adaptive_rag_engine import AdaptiveRAGEngine, RetrievalContext, ProcessStage
from .citation import (
    Citation,
    CitationSet,
    CitationContext,
    CitationSource,
    CitationType,
    CitationGenerator,
    CitationEmbedder,
)
from .fusion import (
    RRFusion,
    FusionResult,
    FusionItem,
    ResultType,
    MultiRoundRetrieval,
    fuse_results,
    deduplicate_items,
)
from .reranker import (
    CohereReranker,
    RerankModel,
    RerankResult,
    RerankReport,
    SelfRAGRefiner,
    create_reranker,
    quick_rerank,
)
# LangChain 集成组件
from .langchain_components import (
    SentenceTransformerEmbeddings,
    Qwen3Embeddings,
    KnowledgeGraphVectorStore,
    KnowledgeGraphRetriever,
    MultiSourceRetriever,
    RAGFusionRetriever,
    RAGPromptTemplates,
    RAGChainBuilder,
    LangChainAdapter,
    create_langchain_vectorstore,
    create_qwen3_vectorstore,
    create_wikipedia_retriever,
    documents_to_langchain,
    format_retrieval_results,
    RetrievalResult,
)

__all__ = [
    # 核心引擎
    'AdaptiveRAGEngine',
    'RetrievalContext',
    'ProcessStage',
    # 问题路由
    'QueryRouter',
    'QuestionType',
    'RetrievalPlan',
    # 检索决策
    'RetrievalDecider',
    'MultiSourceRetrievalResult',
    'RetrievalStatus',
    # 结果评估
    'ResultEvaluator',
    'EvaluationReport',
    'ActionDecision',
    'RelevanceLevel',
    # 思维链
    'CoTReasoner',
    'CoTMode',
    'ReasoningChain',
    'ReasoningStep',
    # 引用溯源
    'Citation',
    'CitationSet',
    'CitationContext',
    'CitationSource',
    'CitationType',
    'CitationGenerator',
    'CitationEmbedder',
    # RRF 融合 (多轮检索第一轮)
    'RRFusion',
    'FusionResult',
    'FusionItem',
    'ResultType',
    'MultiRoundRetrieval',
    'fuse_results',
    'deduplicate_items',
    # Cohere 重排序 (多轮检索第二轮)
    'CohereReranker',
    'RerankModel',
    'RerankResult',
    'RerankReport',
    'SelfRAGRefiner',
    'create_reranker',
    'quick_rerank',
    # ========== LangChain 集成组件 ==========
    # Embeddings
    'SentenceTransformerEmbeddings',
    'Qwen3Embeddings',
    # VectorStore & Retriever
    'KnowledgeGraphVectorStore',
    'KnowledgeGraphRetriever',
    'MultiSourceRetriever',
    'RAGFusionRetriever',
    # Prompt Templates & Chain
    'RAGPromptTemplates',
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
