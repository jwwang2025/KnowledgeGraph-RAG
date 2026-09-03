"""问题路由模块：基于 Adaptive-RAG 思想判断问题类型并决定检索策略。"""

import re
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional


class QuestionType(Enum):
    """问题类型枚举"""
    # 事实型问题 - 需要检索外部知识
    FACTUAL = "factual"           # 实体相关的事实问题 (谁、什么、在哪)
    DEFINITION = "definition"     # 定义类问题 (什么是、含义)
    COMPARISON = "comparison"     # 比较类问题 (A和B的区别)
    PROCEDURAL = "procedural"     # 过程类问题 (如何做、步骤)

    # 知识型问题 - 可能需要检索
    RELATION = "relation"         # 关系查询 (X和Y的关系)
    ATTRIBUTE = "attribute"       # 属性查询 (X的属性)

    # 解释型问题 - 需要深度检索
    EXPLANATION = "explanation"   # 解释为什么 (原因、原理)
    ANALYSIS = "analysis"         # 分析类问题

    # 非检索型问题 - 可以直接回答
    CHITCHAT = "chitchat"         # 闲聊
    OPINION = "opinion"           # 观点/主观问题
    MATH = "math"                 # 数学计算

    # 复合型问题 - 需要多源检索
    COMPLEX = "complex"          # 复杂问题


# 闲聊/问候类（不需要检索）
_CHITCHAT_PATTERNS = tuple(re.compile(p) for p in (
    r'^你好', r'^您好', r'^嗨', r'^哈喽', r'^hey', r'^hi',
    r'你是谁', r'你叫什么', r'你好吗', r'今天怎么样',
    r'^再见', r'^拜拜', r'^晚安',
))

# 纯主观/观点类（谨慎检索）
_OPINION_PATTERNS = tuple(re.compile(p) for p in (
    r'你觉得.*怎么样', r'你认为.*好吗', r'喜欢.*吗',
    r'觉得.*如何', r'推荐.*吗',
))

# 数学计算类（不需要外部检索）
_MATH_PATTERNS = tuple(re.compile(p) for p in (
    r'^\d+\s*[+\-*/]\s*\d+',  # 简单计算
    r'计算', r'等于多少', r'结果是',
))

# 各问题类型的分类关键词 (类型, 置信度, 关键词列表)
_CLASSIFY_RULES = (
    (QuestionType.DEFINITION, 0.85, ('什么是', '什么叫', '定义', '含义', '概念', '解释', '是什么')),
    (QuestionType.COMPARISON, 0.80, ('区别', '不同', '比较', '对比', '差异', '哪个好', '还是')),
    (QuestionType.PROCEDURAL, 0.80, ('如何', '怎么', '怎样', '步骤', '方法', '流程', '教程')),
    (QuestionType.EXPLANATION, 0.85, ('为什么', '为何', '原因', '原理', '怎么会')),
    (QuestionType.FACTUAL, 0.75, ('谁', '什么', '哪', '哪里', '几', '多少', '什么时候')),
)

# 属性查询关键词
_ATTRIBUTE_KEYWORDS = ('的属性', '的特点', '的特征', '的优缺点')

# 关系查询正则
_RELATION_PATTERNS = tuple(re.compile(p) for p in (
    r'.*和.*的关系', r'.*与.*的.*', r'.*属于.*',
    r'.*的.*是什么', r'.*和.*的区别',
))

# 不同问题类型对应的知识源配置
_SOURCE_CONFIG = {
    QuestionType.FACTUAL: {
        'sources': ['kg', 'vector', 'wiki', 'image'],
        'priority': ['kg', 'vector', 'wiki'],
    },
    QuestionType.DEFINITION: {
        'sources': ['wiki', 'vector', 'kg'],
        'priority': ['wiki', 'vector'],
    },
    QuestionType.COMPARISON: {
        'sources': ['kg', 'vector', 'wiki'],
        'priority': ['kg', 'vector'],
    },
    QuestionType.PROCEDURAL: {
        'sources': ['vector', 'wiki', 'kg'],
        'priority': ['vector', 'wiki'],
    },
    QuestionType.RELATION: {
        'sources': ['kg', 'vector', 'wiki'],
        'priority': ['kg', 'vector'],
    },
    QuestionType.ATTRIBUTE: {
        'sources': ['kg', 'wiki', 'vector'],
        'priority': ['kg', 'wiki'],
    },
    QuestionType.EXPLANATION: {
        'sources': ['wiki', 'vector', 'kg'],
        'priority': ['wiki', 'vector'],
    },
    QuestionType.ANALYSIS: {
        'sources': ['vector', 'wiki', 'kg'],
        'priority': ['vector', 'wiki'],
    },
    QuestionType.COMPLEX: {
        'sources': ['kg', 'vector', 'wiki', 'image'],
        'priority': ['kg', 'vector', 'wiki'],
    },
}

# 实体指示词：命中时优先使用知识图谱
_ENTITY_INDICATORS = ('是谁', '是什么', '在哪里', '的创始人', '的公司', '的国家')

# 各问题类型的基础推理深度
_REASONING_DEPTHS = {
    QuestionType.FACTUAL: 1,
    QuestionType.DEFINITION: 1,
    QuestionType.COMPARISON: 2,
    QuestionType.PROCEDURAL: 1,
    QuestionType.RELATION: 1,
    QuestionType.ATTRIBUTE: 1,
    QuestionType.EXPLANATION: 2,
    QuestionType.ANALYSIS: 2,
    QuestionType.COMPLEX: 2,
}

# 多实体分隔词
_MULTI_ENTITY_WORDS = ('和', '与', '还是', '或者', '以及')

# 需要 CoT 深度模式 / 迭代检索的问题类型
_DEEP_COT_TYPES = frozenset({QuestionType.COMPLEX, QuestionType.ANALYSIS, QuestionType.COMPARISON})

# 问题类型友好描述
_TYPE_DESCRIPTIONS = {
    QuestionType.FACTUAL: "事实型问题",
    QuestionType.DEFINITION: "定义型问题",
    QuestionType.COMPARISON: "比较型问题",
    QuestionType.PROCEDURAL: "过程型问题",
    QuestionType.RELATION: "关系查询问题",
    QuestionType.ATTRIBUTE: "属性查询问题",
    QuestionType.EXPLANATION: "解释型问题",
    QuestionType.ANALYSIS: "分析型问题",
    QuestionType.CHITCHAT: "闲聊对话",
    QuestionType.OPINION: "观点讨论",
    QuestionType.MATH: "数学计算",
    QuestionType.COMPLEX: "复杂问题",
}


@dataclass
class RetrievalPlan:
    """检索计划 - 描述需要执行的检索策略"""
    need_retrieval: bool                    # 是否需要检索
    question_type: QuestionType            # 问题类型
    knowledge_sources: List[str]            # 需要使用的知识源 (kg, vector, wiki, image)
    priority_sources: List[str]             # 优先级排序的知识源
    max_triples: int                       # 最大三元组数量
    max_docs: int                          # 最大文档数量
    reasoning_depth: int                   # 推理深度 (0=无需推理, 1=简单推理, 2=深度推理)
    need_iterative: bool                   # 是否需要迭代检索
    confidence: float                      # 分类置信度

    # CoT 思维链配置 (新增)
    use_cot: bool = False                 # 是否启用思维链
    cot_mode: str = "zero_shot"           # CoT 模式: zero_shot, few_shot, self_consistency


class QueryRouter:
    """问题路由器：基于规则和关键词匹配分析问题，决定检索策略与推理深度"""

    def __init__(self):
        """初始化路由器，加载配置"""
        self.max_triples = 10
        self.max_docs = 3

    def route(self, query: str, history: List[tuple] = None) -> RetrievalPlan:
        """分析问题并生成检索计划"""
        # 预处理
        query_clean = query.strip()
        query_lower = query_clean.lower()

        # 1. 首先判断是否需要检索
        need_retrieval, skip_reason = self._check_if_need_retrieval(query_clean, query_lower)

        if not need_retrieval:
            return RetrievalPlan(
                need_retrieval=False,
                question_type=QuestionType.CHITCHAT if skip_reason == "chitchat" else QuestionType.OPINION,
                knowledge_sources=[],
                priority_sources=[],
                max_triples=0,
                max_docs=0,
                reasoning_depth=0,
                need_iterative=False,
                confidence=0.95,
                use_cot=False,
                cot_mode="direct"
            )

        # 2-7. 分类 → 知识源 → 推理深度 → 迭代 → CoT → 数量限制
        question_type, confidence = self._classify_question(query_clean, query_lower)
        knowledge_sources, priority_sources = self._decide_knowledge_sources(
            question_type, query_clean, query_lower
        )
        reasoning_depth = self._decide_reasoning_depth(question_type, query_clean)
        need_iterative = self._check_iterative_need(question_type, query_clean)
        use_cot, cot_mode = self._decide_cot_mode(question_type, reasoning_depth, query_clean)
        max_triples = self._adjust_max_triples(question_type, self.max_triples)
        max_docs = self._adjust_max_docs(question_type, self.max_docs)

        return RetrievalPlan(
            need_retrieval=True,
            question_type=question_type,
            knowledge_sources=knowledge_sources,
            priority_sources=priority_sources,
            max_triples=max_triples,
            max_docs=max_docs,
            reasoning_depth=reasoning_depth,
            need_iterative=need_iterative,
            confidence=confidence,
            use_cot=use_cot,
            cot_mode=cot_mode
        )

    def _check_if_need_retrieval(self, query: str, query_lower: str) -> tuple:
        """判断是否需要检索，返回 (need_retrieval, skip_reason)"""
        # 闲聊/问候类 - 不需要检索
        for pattern in _CHITCHAT_PATTERNS:
            if pattern.search(query_lower):
                return False, "chitchat"

        # 纯主观/观点类 - 谨慎检索
        for pattern in _OPINION_PATTERNS:
            if pattern.search(query_lower):
                # 这类问题可以检索但不是必须
                if '推荐' in query or '觉得' in query:
                    return True, "opinion_search"  # 还是需要检索

        # 数学计算类 - 不需要外部检索
        for pattern in _MATH_PATTERNS:
            if pattern.search(query_lower):
                return False, "math"

        # 大多数问题都需要检索
        return True, "normal"

    def _classify_question(self, query: str, query_lower: str) -> tuple:
        """问题分类，返回 (question_type, confidence)"""
        for q_type, confidence, keywords in _CLASSIFY_RULES:
            for kw in keywords:
                if kw in query:
                    return q_type, confidence

        # 关系查询 - 常见于知识图谱
        for pattern in _RELATION_PATTERNS:
            if pattern.search(query_lower):
                return QuestionType.RELATION, 0.80

        # 属性查询
        for kw in _ATTRIBUTE_KEYWORDS:
            if kw in query:
                return QuestionType.ATTRIBUTE, 0.75

        # 默认归类为事实型（最常见）
        return QuestionType.FACTUAL, 0.60

    def _decide_knowledge_sources(self, question_type: QuestionType,
                                   query: str, query_lower: str) -> tuple:
        """根据问题类型决定知识源，返回 (knowledge_sources, priority_sources)"""
        config = _SOURCE_CONFIG.get(question_type, _SOURCE_CONFIG[QuestionType.FACTUAL])

        # 特殊检测：如果问题中包含实体名称，优先使用知识图谱
        priority = list(config['priority'])
        for ind in _ENTITY_INDICATORS:
            if ind in query:
                # 确保 kg 在最前面
                if 'kg' in config['sources']:
                    priority.remove('kg')
                    priority.insert(0, 'kg')
                break

        return list(config['sources']), priority

    def _decide_reasoning_depth(self, question_type: QuestionType, query: str) -> int:
        """决定推理深度 (0=无需推理, 1=简单推理, 2=深度推理)"""
        # 如果包含多实体，可能需要更深的推理
        entity_count = sum(1 for w in _MULTI_ENTITY_WORDS if w in query)

        base_depth = _REASONING_DEPTHS.get(question_type, 1)
        if entity_count >= 2:
            return min(base_depth + 1, 2)

        return base_depth

    def _decide_cot_mode(self, question_type: QuestionType, reasoning_depth: int,
                        query: str) -> tuple:
        """决定 CoT 思维链模式，返回 (use_cot, cot_mode)"""
        # 无需推理的问题不启用 CoT
        if reasoning_depth == 0:
            return False, "direct"

        # 复杂问题启用深度 CoT
        if question_type in _DEEP_COT_TYPES:
            return True, "self_consistency"

        # 解释类问题启用 Few-shot CoT (需要示例引导)
        if question_type == QuestionType.EXPLANATION:
            return True, "few_shot"

        # 其余情况启用 Zero-shot CoT
        return True, "zero_shot"

    def _check_iterative_need(self, question_type: QuestionType, query: str) -> bool:
        """检查是否需要迭代检索 (多条检索链)"""
        # 复杂问题或分析类问题可能需要迭代
        if question_type in _DEEP_COT_TYPES:
            return True

        # 多实体问题可能需要
        entity_separators = ('和', '与', '还是', '或者')
        return sum(1 for s in entity_separators if s in query) >= 2

    def _adjust_max_triples(self, question_type: QuestionType, base: int) -> int:
        """根据问题类型调整最大三元组数量"""
        if question_type == QuestionType.COMPLEX:
            return base * 1.5  # 复杂问题需要更多
        elif question_type in (QuestionType.COMPARISON, QuestionType.RELATION):
            return base * 1.2  # 比较和关系类稍多
        return base

    def _adjust_max_docs(self, question_type: QuestionType, base: int) -> int:
        """根据问题类型调整最大文档数量"""
        if question_type == QuestionType.EXPLANATION:
            return base + 2  # 解释类需要更多上下文
        return base

    def get_type_description(self, question_type: QuestionType) -> str:
        """获取问题类型的友好描述"""
        return _TYPE_DESCRIPTIONS.get(question_type, "未知类型")
