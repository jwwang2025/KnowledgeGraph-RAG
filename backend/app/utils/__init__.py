"""app/utils 模块：向后兼容的统一导出（推荐直接使用 app.rag / app.search / app.nlp / app.model / app.kg）。"""

from app.rag import (
    AdaptiveRAGEngine, RetrievalContext, ProcessStage,
    QueryRouter, QuestionType, RetrievalPlan,
    RetrievalDecider, MultiSourceRetrievalResult, RetrievalStatus,
    ResultEvaluator, EvaluationReport, ActionDecision, RelevanceLevel,
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

from app.logger import setup_logger, get_logger

__all__ = [
    'AdaptiveRAGEngine', 'RetrievalContext', 'ProcessStage',
    'QueryRouter', 'QuestionType', 'RetrievalPlan',
    'RetrievalDecider', 'MultiSourceRetrievalResult', 'RetrievalStatus',
    'ResultEvaluator', 'EvaluationReport', 'ActionDecision', 'RelevanceLevel',
    'CoTReasoner', 'CoTMode', 'ReasoningChain', 'ReasoningStep',
    'VectorSearcher', 'WikiSearcher', 'ImageSearcher',
    'Ner',
    'start_model', 'stream_predict', 'predict', 'init_rag_engine',
    'search_node_item', 'convert_graph_to_triples',
    'load_knowledge_graph',
    'setup_logger', 'get_logger',
]
