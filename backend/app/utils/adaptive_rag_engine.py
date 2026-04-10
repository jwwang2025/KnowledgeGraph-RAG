"""
Adaptive-RAG 核心编排引擎
融合 Adaptive-RAG、Self-RAG 和 CoT 思维链，实现智能问答

核心流程:
1. QueryRouter - 分析问题类型，决定检索策略
2. RetrievalDecider - 执行多源自适应检索
3. ResultEvaluator - 评估检索结果的相关性
4. CoTReasoner - 构建思维链 prompt
5. PromptAssembler - 组装增强后的 prompt
"""

import time
import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Generator
from enum import Enum

from .query_router import QueryRouter, QuestionType, RetrievalPlan
from .retrieval_decider import RetrievalDecider, MultiSourceRetrievalResult, RetrievalStatus
from .result_evaluator import ResultEvaluator, EvaluationReport, ActionDecision, RelevanceLevel
from .cot_reasoner import CoTReasoner, CoTMode, ReasoningChain
from config.settings import settings


class ProcessStage(Enum):
    """处理阶段枚举"""
    INITIALIZING = "initializing"
    ROUTING = "routing"
    RETRIEVING = "retrieving"
    EVALUATING = "evaluating"
    REASONING = "reasoning"  # 新增: CoT 推理阶段
    ASSEMBLING = "assembling"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class RetrievalContext:
    """
    检索上下文 - 存储整个检索流程的中间状态
    
    包含从问题分析到结果评估的所有信息
    """
    # 阶段追踪
    current_stage: ProcessStage = ProcessStage.INITIALIZING
    stage_history: List[str] = field(default_factory=list)
    
    # 原始输入
    query: str = ""
    history: List[tuple] = field(default_factory=list)
    
    # 路由阶段
    retrieval_plan: Optional[RetrievalPlan] = None
    
    # 检索阶段
    retrieval_result: Optional[MultiSourceRetrievalResult] = None
    
    # 评估阶段
    evaluation_report: Optional[EvaluationReport] = None
    
    # CoT 推理阶段 (新增)
    reasoning_chain: Optional[ReasoningChain] = None
    cot_prompt: str = ""
    
    # 组装阶段
    assembled_prompt: str = ""
    knowledge_context: str = ""
    
    # 决策
    use_retrieval: bool = True
    final_action: ActionDecision = ActionDecision.USE_RETRIEVAL
    
    # 元数据
    total_time: float = 0.0
    stage_times: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 错误信息
    error: Optional[str] = None
    
    def add_stage(self, stage: ProcessStage, duration: float = 0.0):
        """记录阶段"""
        self.stage_history.append(stage.value)
        self.current_stage = stage
        if duration > 0:
            self.stage_times[stage.value] = duration
    
    def to_summary(self) -> Dict[str, Any]:
        """转换为摘要字典"""
        return {
            "query": self.query,
            "stage": self.current_stage.value,
            "stage_history": self.stage_history,
            "total_time": f"{self.total_time:.2f}s",
            "use_retrieval": self.use_retrieval,
            "final_action": self.final_action.value,
            "use_cot": self.retrieval_plan.use_cot if self.retrieval_plan else False,
            "cot_mode": self.retrieval_plan.cot_mode if self.retrieval_plan else "direct",
            "has_error": self.error is not None
        }


class AdaptiveRAGEngine:
    """
    Adaptive-RAG + Self-RAG + CoT 核心引擎
    
    融合三种先进思想的智能问答引擎:
    
    Adaptive-RAG 思想:
    - 问题路由: 分析问题类型，决定检索策略
    - 自适应检索: 根据问题类型选择知识源和检索深度
    - 迭代优化: 根据评估结果决定是否需要重新检索
    
    Self-RAG 思想:
    - 结果评估: 评估每条检索结果的相关性
    - 反思决策: 决定是否使用检索结果
    - 质量控制: 只使用高质量的检索结果
    
    CoT (Chain of Thought) 思想:
    - Zero-shot CoT: "让我们一步步思考"引导推理
    - Few-shot CoT: 基于示例的推理学习
    - Self-Consistency: 多路径推理取最优
    """
    
    def __init__(self, project_name: str = "project_v1",
                 vector_db_path: str = "./data/vector_db",
                 enable_evaluation: bool = True,
                 enable_iteration: bool = False,
                 enable_cot: bool = True,
                 default_cot_mode: str = "zero_shot",
                 max_iterations: int = 2):
        """
        初始化引擎
        
        Args:
            project_name: 项目名称
            vector_db_path: 向量数据库路径
            enable_evaluation: 是否启用结果评估 (Self-RAG)
            enable_iteration: 是否启用迭代检索 (Adaptive-RAG)
            enable_cot: 是否启用思维链 (CoT)
            default_cot_mode: 默认 CoT 模式 (zero_shot, few_shot, self_consistency)
            max_iterations: 最大迭代次数
        """
        # 组件初始化
        self.query_router = QueryRouter()
        self.retrieval_decider = RetrievalDecider(
            project_name=project_name,
            vector_db_path=vector_db_path
        )
        self.result_evaluator = ResultEvaluator() if enable_evaluation else None
        
        # CoT 思维链推理器 (新增)
        self.enable_cot = enable_cot
        self.default_cot_mode = default_cot_mode
        self._cot_reasoner: Optional[CoTReasoner] = None
        
        # 配置
        self.enable_evaluation = enable_evaluation
        self.enable_iteration = enable_iteration
        self.max_iterations = max_iterations
        
        # 日志
        self.logger = logging.getLogger(__name__)
        
        # 统计
        self.stats = {
            "total_queries": 0,
            "avg_retrieval_time": 0,
            "avg_evaluation_time": 0,
            "avg_cot_time": 0,
            "cot_usage_count": 0,
            "retrieval_source_counts": {}
        }
    
    @property
    def cot_reasoner(self) -> CoTReasoner:
        """延迟加载 CoT 推理器"""
        if self._cot_reasoner is None:
            mode_map = {
                "zero_shot": CoTMode.ZERO_SHOT,
                "few_shot": CoTMode.FEW_SHOT,
                "self_consistency": CoTMode.SELF_CONSISTENCY,
                "direct": CoTMode.DIRECT
            }
            self._cot_reasoner = CoTReasoner(
                mode=mode_map.get(self.default_cot_mode, CoTMode.ZERO_SHOT)
            )
        return self._cot_reasoner
    
    def process(self, query: str, history: List[tuple] = None) -> RetrievalContext:
        """
        处理用户查询的完整流程
        
        Args:
            query: 用户查询
            history: 对话历史
            
        Returns:
            RetrievalContext: 检索上下文，包含所有中间结果
        """
        start_time = time.time()
        context = RetrievalContext(
            query=query,
            history=history or []
        )
        
        try:
            # 1. 阶段: 问题路由
            context.add_stage(ProcessStage.ROUTING)
            stage_start = time.time()
            context.retrieval_plan = self.query_router.route(query, history)
            context.add_stage(ProcessStage.ROUTING, time.time() - stage_start)
            
            self.logger.info(f"[路由] 问题类型: {context.retrieval_plan.question_type.value}, "
                           f"置信度: {context.retrieval_plan.confidence:.2f}")
            
            # 如果不需要检索，直接返回
            if not context.retrieval_plan.need_retrieval:
                context.use_retrieval = False
                context.final_action = ActionDecision.GENERATE_DIRECT
                context.add_stage(ProcessStage.COMPLETED, time.time() - start_time)
                context.total_time = time.time() - start_time
                return context
            
            # 2. 阶段: 多源检索
            context.add_stage(ProcessStage.RETRIEVING)
            stage_start = time.time()
            context.retrieval_result = self.retrieval_decider.retrieve(
                query, context.retrieval_plan
            )
            context.add_stage(ProcessStage.RETRIEVING, time.time() - stage_start)
            
            self.logger.info(f"[检索] 耗时: {context.retrieval_result.total_time:.2f}s, "
                           f"使用源: {context.retrieval_result.total_sources_used}")
            
            # 3. 阶段: 结果评估 (Self-RAG)
            if self.enable_evaluation and context.retrieval_result.total_sources_used > 0:
                context.add_stage(ProcessStage.EVALUATING)
                stage_start = time.time()
                context.evaluation_report = self.result_evaluator.evaluate(
                    query=query,
                    question_type=context.retrieval_plan.question_type.value,
                    triples=context.retrieval_result.triples,
                    documents=context.retrieval_result.documents,
                    wiki_summary=context.retrieval_result.wiki_summary
                )
                context.add_stage(ProcessStage.EVALUATING, time.time() - stage_start)
                
                self.logger.info(f"[评估] 整体相关性: {context.evaluation_report.overall_relevance.value}, "
                               f"置信度: {context.evaluation_report.overall_confidence:.2f}")
                
                # 基于评估结果决定行动
                context.final_action = context.evaluation_report.action
                
                # 检查是否需要迭代
                if self.enable_iteration and context.final_action == ActionDecision.ITERATE_RETRIEVAL:
                    context = self._iterate_retrieval(context)
            else:
                context.final_action = ActionDecision.USE_RETRIEVAL
            
            # 4. 阶段: CoT 思维链推理 (新增)
            if self.enable_cot and context.retrieval_plan.use_cot:
                context.add_stage(ProcessStage.REASONING)
                stage_start = time.time()
                self._apply_cot_reasoning(context)
                context.add_stage(ProcessStage.REASONING, time.time() - stage_start)
                
                self.logger.info(f"[CoT] 模式: {context.retrieval_plan.cot_mode}, "
                               f"推理链长度: {len(context.reasoning_chain.steps) if context.reasoning_chain else 0}")
            
            # 5. 阶段: Prompt 组装
            context.add_stage(ProcessStage.ASSEMBLING)
            stage_start = time.time()
            self._assemble_prompt(context)
            context.add_stage(ProcessStage.ASSEMBLING, time.time() - stage_start)
            
            # 完成
            context.add_stage(ProcessStage.COMPLETED)
            context.total_time = time.time() - start_time
            
            # 更新统计
            self._update_stats(context)
            
            return context
            
        except Exception as e:
            context.error = str(e)
            context.add_stage(ProcessStage.FAILED)
            context.total_time = time.time() - start_time
            self.logger.error(f"[错误] 处理查询时出错: {e}")
            return context
    
    def _apply_cot_reasoning(self, context: RetrievalContext):
        """应用 CoT 思维链推理"""
        if not context.retrieval_plan.use_cot:
            return
        
        # 根据路由决定 CoT 模式
        cot_mode_str = context.retrieval_plan.cot_mode
        
        # 创建对应模式的 CoT 推理器
        mode_map = {
            "zero_shot": CoTMode.ZERO_SHOT,
            "few_shot": CoTMode.FEW_SHOT,
            "self_consistency": CoTMode.SELF_CONSISTENCY,
            "direct": CoTMode.DIRECT
        }
        
        cot_mode = mode_map.get(cot_mode_str, CoTMode.ZERO_SHOT)
        reasoner = CoTReasoner(mode=cot_mode)
        
        # 执行推理
        context.reasoning_chain = reasoner.reason(
            query=context.query,
            knowledge_context=context.knowledge_context,
            depth=context.retrieval_plan.reasoning_depth
        )
        
        # 生成 CoT prompt
        context.cot_prompt = reasoner.build_cot_prompt(
            query=context.query,
            knowledge=context.knowledge_context
        )
    
    def _iterate_retrieval(self, context: RetrievalContext) -> RetrievalContext:
        """
        迭代检索 - Adaptive-RAG 的核心特性
        
        当评估结果显示需要更多信息时，重新制定检索策略并检索
        """
        iterations = 0
        
        while iterations < self.max_iterations and (
            context.final_action == ActionDecision.ITERATE_RETRIEVAL
        ):
            iterations += 1
            self.logger.info(f"[迭代 {iterations}] 重新制定检索策略...")
            
            # 基于评估报告的建议调整检索计划
            suggestions = context.evaluation_report.suggestions
            alternative_approaches = context.evaluation_report.alternative_approaches
            
            # 调整策略: 增加检索数量，尝试不同知识源
            original_plan = context.retrieval_plan
            new_plan = RetrievalPlan(
                need_retrieval=True,
                question_type=original_plan.question_type,
                knowledge_sources=original_plan.knowledge_sources.copy(),
                priority_sources=self._adjust_source_priority(
                    original_plan.priority_sources, 
                    suggestions
                ),
                max_triples=int(original_plan.max_triples * 1.5),
                max_docs=int(original_plan.max_docs * 1.5),
                reasoning_depth=min(original_plan.reasoning_depth + 1, 2),
                need_iterative=False,  # 避免无限循环
                confidence=original_plan.confidence,
                use_cot=original_plan.use_cot,
                cot_mode=original_plan.cot_mode
            )
            
            # 重新检索
            new_result = self.retrieval_decider.retrieve(context.query, new_plan)
            
            # 重新评估
            new_evaluation = self.result_evaluator.evaluate(
                query=context.query,
                question_type=new_plan.question_type.value,
                triples=new_result.triples,
                documents=new_result.documents,
                wiki_summary=new_result.wiki_summary
            )
            
            # 如果新结果更好，更新上下文
            if new_evaluation.overall_confidence > (
                context.evaluation_report.overall_confidence * 1.1
            ):
                self.logger.info(f"[迭代 {iterations}] 新结果更好，更新上下文")
                context.retrieval_plan = new_plan
                context.retrieval_result = new_result
                context.evaluation_report = new_evaluation
                context.final_action = new_evaluation.action
            else:
                self.logger.info(f"[迭代 {iterations}] 新结果无明显改善")
                context.final_action = ActionDecision.USE_RETRIEVAL
                break
        
        return context
    
    def _adjust_source_priority(self, original: List[str], 
                                suggestions: List[str]) -> List[str]:
        """根据评估建议调整知识源优先级"""
        new_priority = original.copy()
        
        for suggestion in suggestions:
            if '知识图谱' in suggestion or '图谱' in suggestion:
                if 'kg' in new_priority:
                    new_priority.remove('kg')
                    new_priority.insert(0, 'kg')
            elif 'Wiki' in suggestion or 'wikipedia' in suggestion.lower():
                if 'wiki' in new_priority:
                    new_priority.remove('wiki')
                    new_priority.insert(0, 'wiki')
            elif '文档' in suggestion or '向量' in suggestion:
                if 'vector' in new_priority:
                    new_priority.remove('vector')
                    new_priority.insert(0, 'vector')
        
        return new_priority
    
    def _assemble_prompt(self, context: RetrievalContext):
        """组装最终的 prompt (集成 CoT)"""
        if not context.use_retrieval or context.final_action == ActionDecision.GENERATE_DIRECT:
            # 无检索情况下的 prompt
            if self.enable_cot:
                # 使用 CoT 推理器生成 prompt
                context.assembled_prompt = context.cot_prompt or context.query
            else:
                context.assembled_prompt = context.query
            context.knowledge_context = ""
            return
        
        # 如果已经有 CoT prompt，使用它
        if context.cot_prompt:
            context.assembled_prompt = context.cot_prompt
            return
        
        # 获取检索结果
        triples = context.retrieval_result.triples
        documents = context.retrieval_result.documents
        wiki_summary = context.retrieval_result.wiki_summary
        
        # 如果启用了评估，过滤低质量结果
        if self.enable_evaluation and context.evaluation_report:
            filtered = self.result_evaluator.get_filtered_results(
                context.evaluation_report,
                min_relevance=RelevanceLevel.LOW
            )
            triples = filtered["triples"]
            documents = filtered["documents"]
            wiki_summary = filtered["wiki_summary"] or wiki_summary
        
        # 构建知识上下文
        context_parts = []
        
        # 1. 知识图谱三元组
        if triples:
            triples_str = "；".join([f"({t[0]} {t[1]} {t[2]})" for t in triples[:10]])
            context_parts.append(f"【知识图谱】{triples_str}")
        
        # 2. 文档片段
        if documents:
            docs_str = "；".join(documents[:3])
            context_parts.append(f"【相关文档】{docs_str}")
        
        # 3. Wikipedia 摘要
        if wiki_summary:
            context_parts.append(f"【Wikipedia】{wiki_summary[:500]}")
        
        # 组装知识上下文
        context.knowledge_context = "；".join(context_parts)
        
        # 检查是否启用 CoT
        if self.enable_cot and context.retrieval_plan and context.retrieval_plan.use_cot:
            # 使用 CoT 模式组装 prompt
            reasoner = self.cot_reasoner
            context.assembled_prompt = reasoner.build_cot_prompt(
                query=context.query,
                knowledge=context.knowledge_context
            )
        elif context.knowledge_context:
            # 普通模式 (无 CoT)
            context.assembled_prompt = (
                f"\n===参考资料===\n{context.knowledge_context}；\n\n"
                f"根据上面资料，用简洁且准确的话回答下面问题：\n{context.query}"
            )
        else:
            context.assembled_prompt = context.query
    
    def _update_stats(self, context: RetrievalContext):
        """更新统计信息"""
        self.stats["total_queries"] += 1
        
        if context.retrieval_result:
            # 平均检索时间
            current_avg = self.stats["avg_retrieval_time"]
            n = self.stats["total_queries"]
            self.stats["avg_retrieval_time"] = (
                (current_avg * (n - 1) + context.retrieval_result.total_time) / n
            )
            
            # 知识源使用统计
            for source, result in context.retrieval_result.results.items():
                if result.status == RetrievalStatus.COMPLETED:
                    self.stats["retrieval_source_counts"][source] = (
                        self.stats["retrieval_source_counts"].get(source, 0) + 1
                    )
        
        if context.evaluation_report:
            current_avg = self.stats["avg_evaluation_time"]
            n = self.stats["total_queries"]
            eval_time = context.stage_times.get(ProcessStage.EVALUATING.value, 0)
            self.stats["avg_evaluation_time"] = (
                (current_avg * (n - 1) + eval_time) / n
            )
        
        # CoT 统计
        if context.reasoning_chain:
            self.stats["cot_usage_count"] = self.stats.get("cot_usage_count", 0) + 1
            reason_time = context.stage_times.get(ProcessStage.REASONING.value, 0)
            current_avg = self.stats["avg_cot_time"]
            n = self.stats["total_queries"]
            self.stats["avg_cot_time"] = (
                (current_avg * (n - 1) + reason_time) / n
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats.copy()
    
    def reset_stats(self):
        """重置统计"""
        self.stats = {
            "total_queries": 0,
            "avg_retrieval_time": 0,
            "avg_evaluation_time": 0,
            "avg_cot_time": 0,
            "cot_usage_count": 0,
            "retrieval_source_counts": {}
        }


# ============ 便捷函数 ============

# 全局引擎实例 (延迟初始化)
_global_engine: Optional[AdaptiveRAGEngine] = None


def get_engine(project_name: str = "project_v1") -> AdaptiveRAGEngine:
    """获取全局引擎实例"""
    global _global_engine
    if _global_engine is None:
        _global_engine = AdaptiveRAGEngine(
            project_name=project_name,
            vector_db_path="./data/vector_db"
        )
    return _global_engine


def process_query(query: str, history: List[tuple] = None) -> RetrievalContext:
    """
    处理用户查询的便捷函数
    
    Args:
        query: 用户查询
        history: 对话历史
        
    Returns:
        RetrievalContext: 检索上下文
    """
    engine = get_engine()
    return engine.process(query, history)