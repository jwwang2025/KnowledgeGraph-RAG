"""
RAG 模块 - 检索增强生成核心组件

包含:
- query_router: 问题路由
- retrieval_decider: 检索决策
- result_evaluator: 结果评估
- cot_reasoner: 思维链推理
- adaptive_rag_engine: 核心引擎
"""

from .query_router import QueryRouter, QuestionType, RetrievalPlan
from .retrieval_decider import RetrievalDecider, MultiSourceRetrievalResult, RetrievalStatus
from .result_evaluator import ResultEvaluator, EvaluationReport, ActionDecision, RelevanceLevel
from .cot_reasoner import CoTReasoner, CoTMode, ReasoningChain, ReasoningStep
from .adaptive_rag_engine import AdaptiveRAGEngine, RetrievalContext, ProcessStage

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
]
