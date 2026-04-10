"""
问题路由模块 (Query Router)
基于 Adaptive-RAG 思想，根据问题特征判断问题类型并决定检索策略
"""

import re
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from config.settings import settings


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
    confidence: float                       # 分类置信度


class QueryRouter:
    """
    问题路由器 - 分析用户问题，决定检索策略
    
    基于规则和关键词匹配进行问题分类，适用于：
    - 快速判断问题类型
    - 决定需要使用的知识源
    - 控制检索深度
    """
    
    def __init__(self):
        """初始化路由器，加载配置"""
        self.max_triples = 10
        self.max_docs = 3
        
    def route(self, query: str, history: List[tuple] = None) -> RetrievalPlan:
        """
        分析问题并生成检索计划
        
        Args:
            query: 用户问题
            history: 对话历史 (可选)
            
        Returns:
            RetrievalPlan: 检索计划
        """
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
                confidence=0.95
            )
        
        # 2. 判断问题类型
        question_type, confidence = self._classify_question(query_clean, query_lower)
        
        # 3. 决定知识源
        knowledge_sources, priority_sources = self._decide_knowledge_sources(
            question_type, query_clean, query_lower
        )
        
        # 4. 决定推理深度
        reasoning_depth = self._decide_reasoning_depth(question_type, query_clean)
        
        # 5. 决定是否需要迭代检索
        need_iterative = self._check_iterative_need(question_type, query_clean)
        
        # 6. 调整检索数量限制
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
            confidence=confidence
        )
    
    def _check_if_need_retrieval(self, query: str, query_lower: str) -> tuple:
        """
        判断是否需要检索
        
        Returns:
            (need_retrieval: bool, skip_reason: str)
        """
        # 闲聊/问候类 - 不需要检索
        chitchat_patterns = [
            r'^你好', r'^您好', r'^嗨', r'^哈喽', r'^hey', r'^hi',
            r'你是谁', r'你叫什么', r'你好吗', r'今天怎么样',
            r'^再见', r'^拜拜', r'^晚安'
        ]
        for pattern in chitchat_patterns:
            if re.search(pattern, query_lower):
                return False, "chitchat"
        
        # 纯主观/观点类 - 谨慎检索
        opinion_patterns = [
            r'你觉得.*怎么样', r'你认为.*好吗', r'喜欢.*吗',
            r'觉得.*如何', r'推荐.*吗'
        ]
        for pattern in opinion_patterns:
            if re.search(pattern, query_lower):
                # 这类问题可以检索但不是必须
                if '推荐' in query or '觉得' in query:
                    return True, "opinion_search"  # 还是需要检索
        
        # 数学计算类 - 不需要外部检索
        math_patterns = [
            r'^\d+\s*[+\-*/]\s*\d+',  # 简单计算
            r'计算', r'等于多少', r'结果是'
        ]
        for pattern in math_patterns:
            if re.search(pattern, query_lower):
                return False, "math"
        
        # 大多数问题都需要检索
        return True, "normal"
    
    def _classify_question(self, query: str, query_lower: str) -> tuple:
        """
        问题分类
        
        Returns:
            (question_type: QuestionType, confidence: float)
        """
        # 定义类问题
        definition_keywords = ['什么是', '什么叫', '定义', '含义', '概念', '解释', '是什么']
        for kw in definition_keywords:
            if kw in query:
                return QuestionType.DEFINITION, 0.85
        
        # 比较类问题
        comparison_keywords = ['区别', '不同', '比较', '对比', '差异', '哪个好', '还是']
        for kw in comparison_keywords:
            if kw in query:
                return QuestionType.COMPARISON, 0.80
        
        # 过程/步骤类问题
        procedural_keywords = ['如何', '怎么', '怎样', '步骤', '方法', '流程', '教程']
        for kw in procedural_keywords:
            if kw in query:
                return QuestionType.PROCEDURAL, 0.80
        
        # 原因/解释类问题
        explanation_keywords = ['为什么', '为何', '原因', '原理', '怎么会', '怎么会']
        for kw in explanation_keywords:
            if kw in query:
                return QuestionType.EXPLANATION, 0.85
        
        # 谁/什么/哪里/时间等事实类
        factual_keywords = [
            ('谁', '何人', '是什么人'),      # 人物
            ('什么.*是谁', '什么是'),       # 实体
            ('在哪', '哪里', '地点'),        # 地点
            ('什么时候', '时间', '多久'),    # 时间
            ('多少', '数量', '价格'),        # 数量
        ]
        factual_indicators = ['谁', '什么', '哪', '哪里', '几', '多少', '什么时候']
        for ind in factual_indicators:
            if ind in query:
                return QuestionType.FACTUAL, 0.75
        
        # 关系查询 - 常见于知识图谱
        relation_patterns = [
            r'.*和.*的关系', r'.*与.*的.*', r'.*属于.*',
            r'.*的.*是什么', r'.*和.*的区别'
        ]
        for pattern in relation_patterns:
            if re.search(pattern, query_lower):
                return QuestionType.RELATION, 0.80
        
        # 属性查询
        attribute_keywords = ['的属性', '的特点', '的特征', '的优缺点']
        for kw in attribute_keywords:
            if kw in query:
                return QuestionType.ATTRIBUTE, 0.75
        
        # 默认归类为事实型（最常见）
        return QuestionType.FACTUAL, 0.60
    
    def _decide_knowledge_sources(self, question_type: QuestionType, 
                                   query: str, query_lower: str) -> tuple:
        """
        根据问题类型决定知识源
        
        Returns:
            (knowledge_sources: List[str], priority_sources: List[str])
        """
        # 不同问题类型对应的知识源配置
        source_config = {
            QuestionType.FACTUAL: {
                'sources': ['kg', 'vector', 'wiki', 'image'],
                'priority': ['kg', 'vector', 'wiki']
            },
            QuestionType.DEFINITION: {
                'sources': ['wiki', 'vector', 'kg'],
                'priority': ['wiki', 'vector']
            },
            QuestionType.COMPARISON: {
                'sources': ['kg', 'vector', 'wiki'],
                'priority': ['kg', 'vector']
            },
            QuestionType.PROCEDURAL: {
                'sources': ['vector', 'wiki', 'kg'],
                'priority': ['vector', 'wiki']
            },
            QuestionType.RELATION: {
                'sources': ['kg', 'vector', 'wiki'],
                'priority': ['kg', 'vector']
            },
            QuestionType.ATTRIBUTE: {
                'sources': ['kg', 'wiki', 'vector'],
                'priority': ['kg', 'wiki']
            },
            QuestionType.EXPLANATION: {
                'sources': ['wiki', 'vector', 'kg'],
                'priority': ['wiki', 'vector']
            },
            QuestionType.ANALYSIS: {
                'sources': ['vector', 'wiki', 'kg'],
                'priority': ['vector', 'wiki']
            },
            QuestionType.COMPLEX: {
                'sources': ['kg', 'vector', 'wiki', 'image'],
                'priority': ['kg', 'vector', 'wiki']
            },
        }
        
        config = source_config.get(question_type, source_config[QuestionType.FACTUAL])
        
        # 特殊检测：如果问题中包含实体名称，优先使用知识图谱
        entity_indicators = ['是谁', '是什么', '在哪里', '的创始人', '的公司', '的国家']
        for ind in entity_indicators:
            if ind in query:
                # 确保 kg 在最前面
                if 'kg' in config['sources']:
                    config['priority'].remove('kg')
                    config['priority'].insert(0, 'kg')
                break
        
        return config['sources'], config['priority']
    
    def _decide_reasoning_depth(self, question_type: QuestionType, query: str) -> int:
        """
        决定推理深度
        
        0: 无需推理，直接回答
        1: 简单推理 (如实体关系查询)
        2: 深度推理 (如多跳查询、比较分析)
        """
        depth_map = {
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
        
        # 如果包含多实体，可能需要更深的推理
        multi_entity_patterns = [r'和', r'与', r'还是', r'或者', r'以及']
        entity_count = sum(1 for p in multi_entity_patterns if p in query)
        
        base_depth = depth_map.get(question_type, 1)
        if entity_count >= 2:
            return min(base_depth + 1, 2)
        
        return base_depth
    
    def _check_iterative_need(self, question_type: QuestionType, query: str) -> bool:
        """
        检查是否需要迭代检索 (多条检索链)
        """
        # 复杂问题或分析类问题可能需要迭代
        if question_type in [QuestionType.COMPLEX, QuestionType.ANALYSIS, QuestionType.COMPARISON]:
            return True
        
        # 多实体问题可能需要
        entity_separators = ['和', '与', '还是', '或者']
        if sum(1 for s in entity_separators if s in query) >= 2:
            return True
        
        return False
    
    def _adjust_max_triples(self, question_type: QuestionType, base: int) -> int:
        """根据问题类型调整最大三元组数量"""
        if question_type == QuestionType.COMPLEX:
            return base * 1.5  # 复杂问题需要更多
        elif question_type in [QuestionType.COMPARISON, QuestionType.RELATION]:
            return base * 1.2  # 比较和关系类稍多
        return base
    
    def _adjust_max_docs(self, question_type: QuestionType, base: int) -> int:
        """根据问题类型调整最大文档数量"""
        if question_type == QuestionType.EXPLANATION:
            return base + 2  # 解释类需要更多上下文
        return base
    
    def get_type_description(self, question_type: QuestionType) -> str:
        """获取问题类型的友好描述"""
        descriptions = {
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
        return descriptions.get(question_type, "未知类型")