"""app/utils 模块：向后兼容的统一导出（推荐直接使用 app.rag / app.search / app.nlp / app.model / app.kg）。"""

# 从各子模块导入，保持向后兼容
from app.rag import (
    # 核心引擎
    AdaptiveRAGEngine, RetrievalContext, ProcessStage,
    # 问题路由
    QueryRouter, QuestionType, RetrievalPlan,
    # 检索决策
    RetrievalDecider, MultiSourceRetrievalResult, RetrievalStatus,
    # 结果评估
    ResultEvaluator, EvaluationReport, ActionDecision, RelevanceLevel,
    # 思维链
    CoTReasoner, CoTMode, ReasoningChain, ReasoningStep,
)

from app.search import (
    VectorSearcher,
    WikiSearcher,
    ImageSearcher,
)

from app.nlp import (
    Ner,
)

from app.model import (
    start_model,
    stream_predict,
    predict,
    init_rag_engine,
)

from app.kg import (
    search_node_item,
    convert_graph_to_triples,
    load_knowledge_graph,
)

# 保留 logger
from app.logger import setup_logger, get_logger

__all__ = [
    # RAG 模块
    'AdaptiveRAGEngine', 'RetrievalContext', 'ProcessStage',
    'QueryRouter', 'QuestionType', 'RetrievalPlan',
    'RetrievalDecider', 'MultiSourceRetrievalResult', 'RetrievalStatus',
    'ResultEvaluator', 'EvaluationReport', 'ActionDecision', 'RelevanceLevel',
    'CoTReasoner', 'CoTMode', 'ReasoningChain', 'ReasoningStep',
    # 搜索模块
    'VectorSearcher', 'WikiSearcher', 'ImageSearcher',
    # NLP 模块
    'Ner',
    # 模型模块
    'start_model', 'stream_predict', 'predict', 'init_rag_engine',
    # KG 模块
    'search_node_item', 'convert_graph_to_triples',
    'load_knowledge_graph',
    # 日志模块
    'setup_logger', 'get_logger',
]
