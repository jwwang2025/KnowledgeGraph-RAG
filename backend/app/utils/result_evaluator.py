"""
结果评估与反思模块 (Result Evaluator & Reflector)
基于 Self-RAG 思想，评估检索结果的相关性并决定下一步行动
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from config.settings import settings


class RelevanceLevel(Enum):
    """相关性等级枚举"""
    HIGH = "high"           # 高度相关
    MEDIUM = "medium"       # 中等相关
    LOW = "low"             # 低相关
    IRRELEVANT = "irrelevant"  # 不相关
    UNKNOWN = "unknown"     # 未知


class ActionDecision(Enum):
    """行动决策枚举 (Self-RAG 核心)"""
    # 使用检索结果
    USE_RETRIEVAL = "use_retrieval"           # 使用检索结果生成
    
    # 不使用检索结果
    USE_PARAMETRIC = "use_parametric"         # 使用模型参数知识
    USE_HISTORY = "use_history"               # 使用对话历史
    GENERATE_DIRECT = "generate_direct"       # 无需外部知识直接生成
    
    # 需要进一步行动
    ITERATE_RETRIEVAL = "iterate_retrieval"   # 迭代检索（重新检索）
    EXPAND_ENTITIES = "expand_entities"       # 扩展实体后再检索
    SWITCH_SOURCE = "switch_source"           # 切换知识源
    
    # 拒绝/无法回答
    REFUSE = "refuse"                         # 无法回答


@dataclass
class RelevanceScore:
    """单条检索结果的相关性评分"""
    source: str                    # 来源
    content: str                   # 内容摘要
    relevance: RelevanceLevel      # 相关性等级
    confidence: float              # 置信度 (0-1)
    reasons: List[str] = field(default_factory=list)  # 判定原因
    issues: List[str] = field(default_factory=list)    # 发现的问题
    
    # 详细评分维度
    semantic_match: float = 0.0    # 语义匹配度 (0-1)
    entity_match: float = 0.0     # 实体匹配度 (0-1)
    completeness: float = 0.0     # 完整性 (0-1)


@dataclass
class EvaluationReport:
    """评估报告"""
    query: str                               # 原始查询
    question_type: str                       # 问题类型
    
    # 检索结果评估
    triple_scores: List[RelevanceScore] = field(default_factory=list)
    document_scores: List[RelevanceScore] = field(default_factory=list)
    wiki_score: Optional[RelevanceScore] = None
    
    # 聚合评估
    overall_relevance: RelevanceLevel = RelevanceLevel.UNKNOWN
    overall_confidence: float = 0.0
    knowledge_sufficiency: float = 0.0       # 知识充足度 (0-1)
    
    # 决策
    action: ActionDecision = ActionDecision.USE_RETRIEVAL
    action_reasons: List[str] = field(default_factory=list)
    
    # 反思建议
    suggestions: List[str] = field(default_factory=list)
    alternative_approaches: List[str] = field(default_factory=list)
    
    # 质量指标
    total_results: int = 0
    high_relevant_count: int = 0
    medium_relevant_count: int = 0
    low_relevant_count: int = 0
    irrelevant_count: int = 0
    
    def get_summary(self) -> Dict[str, Any]:
        """获取评估摘要"""
        return {
            "query": self.query,
            "question_type": self.question_type,
            "overall_relevance": self.overall_relevance.value,
            "overall_confidence": f"{self.overall_confidence:.2f}",
            "knowledge_sufficiency": f"{self.knowledge_sufficiency:.2f}",
            "action": self.action.value,
            "total_results": self.total_results,
            "high_relevant": self.high_relevant_count,
            "relevant_total": self.high_relevant_count + self.medium_relevant_count
        }


class ResultEvaluator:
    """
    结果评估器 - 评估检索结果的质量和相关性
    
    基于 Self-RAG 的反思机制，对每条检索结果进行评估
    """
    
    def __init__(self, use_llm_evaluation: bool = False):
        """
        初始化评估器
        
        Args:
            use_llm_evaluation: 是否使用LLM进行深度评估 (暂未实现)
        """
        self.use_llm_evaluation = use_llm_evaluation
        
        # 相关性阈值配置
        self.thresholds = {
            'high': 0.7,      # >= 0.7 为高度相关
            'medium': 0.4,   # >= 0.4 为中等相关
            'low': 0.2,       # >= 0.2 为低相关
        }
    
    def evaluate(self, query: str, question_type: str,
                 triples: List[Tuple], documents: List[str],
                 wiki_summary: Optional[str] = None) -> EvaluationReport:
        """
        评估检索结果
        
        Args:
            query: 原始查询
            question_type: 问题类型
            triples: 知识图谱三元组
            documents: 检索到的文档
            wiki_summary: Wikipedia摘要
            
        Returns:
            EvaluationReport: 评估报告
        """
        report = EvaluationReport(
            query=query,
            question_type=question_type
        )
        
        # 1. 提取查询关键词和实体
        query_keywords = self._extract_keywords(query)
        query_entities = self._extract_entities(query)
        
        # 2. 评估三元组
        for triple in triples:
            score = self._evaluate_triple(triple, query, query_keywords, query_entities)
            report.triple_scores.append(score)
        
        # 3. 评估文档
        for doc in documents:
            score = self._evaluate_document(doc, query, query_keywords, query_entities)
            report.document_scores.append(score)
        
        # 4. 评估Wiki
        if wiki_summary:
            report.wiki_score = self._evaluate_wiki(
                wiki_summary, query, query_keywords, query_entities
            )
        
        # 5. 聚合评估
        self._aggregate_evaluation(report)
        
        # 6. 做出决策
        self._make_decision(report)
        
        return report
    
    def _extract_keywords(self, query: str) -> List[str]:
        """提取查询关键词"""
        # 移除停用词
        stopwords = ['的', '是', '在', '了', '和', '与', '或', '有', '吗', '呢', '吧', '啊']
        words = [w for w in query if len(w) >= 1 and w not in stopwords]
        
        # 保留2字及以上的词
        keywords = []
        for i in range(len(query)):
            for length in [4, 3, 2]:
                if i + length <= len(query):
                    word = query[i:i+length]
                    if word not in stopwords and not any(c in '，。！？、；：""''（）' for c in word):
                        keywords.append(word)
        
        return list(set(keywords))
    
    def _extract_entities(self, query: str) -> List[str]:
        """提取查询中的实体（简单基于规则）"""
        # 常见实体类型词缀
        entity_suffixes = {
            '人物': ['人', '者', '家', '师', '员', '生'],
            '组织': ['公司', '机构', '组织', '大学', '医院', '党', '会'],
            '地点': ['国', '省', '市', '县', '区', '镇', '村', '山', '河', '湖', '海'],
            '概念': ['主义', '论', '学', '理论', '方法', '技术'],
        }
        
        entities = []
        
        # 提取带词缀的实体
        for ent_type, suffixes in entity_suffixes.items():
            for suffix in suffixes:
                pattern = f'.*{suffix}'
                matches = re.findall(pattern, query)
                entities.extend(matches)
        
        # 提取引号内的内容
        quoted = re.findall(r'[""]([^""]+)[""]', query)
        entities.extend(quoted)
        
        # 提取「」内的内容
        brackets = re.findall(r'[「』]([^「」]+)[「」]', query)
        entities.extend(brackets)
        
        return list(set(entities))
    
    def _evaluate_triple(self, triple: Tuple, query: str,
                        keywords: List[str], entities: List[str]) -> RelevanceScore:
        """评估单个三元组的相关性"""
        subject, predicate, obj = triple
        
        # 构建三元组文本表示
        triple_text = f"{subject}{predicate}{obj}"
        triple_full = f"{subject} {predicate} {obj}"
        
        reasons = []
        issues = []
        
        # 1. 语义匹配度
        semantic_score = self._calculate_text_similarity(query, triple_text, keywords)
        
        # 2. 实体匹配度
        entity_score = 0.0
        entity_matches = []
        
        for entity in entities:
            if entity in triple_text:
                entity_score += 0.5
                entity_matches.append(entity)
        
        # 检查subject和object是否在查询中
        if subject in query:
            entity_score += 0.3
            reasons.append(f"主体匹配: {subject}")
        if obj in query:
            entity_score += 0.2
            reasons.append(f"客体匹配: {obj}")
        
        entity_score = min(entity_score, 1.0)
        
        # 3. 完整性评分
        completeness = 1.0
        if not subject or subject == "Unknown":
            completeness *= 0.5
            issues.append("主体缺失或未知")
        if not predicate or predicate == "Unknown":
            completeness *= 0.5
            issues.append("关系缺失或未知")
        if not obj or obj == "Unknown":
            completeness *= 0.5
            issues.append("客体缺失或未知")
        
        # 综合评分
        confidence = (semantic_score * 0.4 + entity_score * 0.4 + completeness * 0.2)
        
        # 确定相关性等级
        if confidence >= self.thresholds['high']:
            relevance = RelevanceLevel.HIGH
            reasons.append("高度相关")
        elif confidence >= self.thresholds['medium']:
            relevance = RelevanceLevel.MEDIUM
            reasons.append("中等相关")
        elif confidence >= self.thresholds['low']:
            relevance = RelevanceLevel.LOW
            reasons.append("低相关")
        else:
            relevance = RelevanceLevel.IRRELEVANT
            issues.append("不相关")
        
        return RelevanceScore(
            source="knowledge_graph",
            content=triple_full,
            relevance=relevance,
            confidence=confidence,
            reasons=reasons,
            issues=issues,
            semantic_match=semantic_score,
            entity_match=entity_score,
            completeness=completeness
        )
    
    def _evaluate_document(self, doc: str, query: str,
                          keywords: List[str], entities: List[str]) -> RelevanceScore:
        """评估文档的相关性"""
        reasons = []
        issues = []
        
        # 1. 语义匹配度
        semantic_score = self._calculate_text_similarity(query, doc, keywords)
        
        # 2. 实体匹配度
        entity_score = 0.0
        for entity in entities:
            if entity in doc:
                entity_score += 0.5
        
        # 检查关键词出现次数
        keyword_hits = 0
        for keyword in keywords:
            if len(keyword) >= 2 and keyword in doc:
                keyword_hits += 1
        
        entity_score = min(entity_score + keyword_hits * 0.1, 1.0)
        
        # 3. 完整性 - 文档长度适中为佳
        completeness = min(len(doc) / 500, 1.0) if len(doc) > 0 else 0.0
        
        # 综合评分
        confidence = semantic_score * 0.5 + entity_score * 0.3 + completeness * 0.2
        
        # 长度过短可能是噪音
        if len(doc) < 20:
            confidence *= 0.7
            issues.append("文档过短，可能不完整")
        
        # 确定相关性等级
        if confidence >= self.thresholds['high']:
            relevance = RelevanceLevel.HIGH
            reasons.append("文档高度相关")
        elif confidence >= self.thresholds['medium']:
            relevance = RelevanceLevel.MEDIUM
            reasons.append("文档中等相关")
        elif confidence >= self.thresholds['low']:
            relevance = RelevanceLevel.LOW
            reasons.append("文档低相关")
        else:
            relevance = RelevanceLevel.IRRELEVANT
            issues.append("文档不相关")
        
        return RelevanceScore(
            source="vector_database",
            content=doc[:200] + "..." if len(doc) > 200 else doc,
            relevance=relevance,
            confidence=confidence,
            reasons=reasons,
            issues=issues,
            semantic_match=semantic_score,
            entity_match=entity_score,
            completeness=completeness
        )
    
    def _evaluate_wiki(self, wiki_summary: str, query: str,
                      keywords: List[str], entities: List[str]) -> RelevanceScore:
        """评估Wikipedia摘要的相关性"""
        reasons = []
        issues = []
        
        # 语义匹配度
        semantic_score = self._calculate_text_similarity(query, wiki_summary, keywords)
        
        # 实体匹配度
        entity_score = 0.0
        for entity in entities:
            if entity in wiki_summary:
                entity_score += 0.5
        
        entity_score = min(entity_score, 1.0)
        
        # Wikipedia 通常比较权威，完整性较高
        completeness = 0.9
        
        confidence = semantic_score * 0.4 + entity_score * 0.4 + completeness * 0.2
        
        if confidence >= self.thresholds['high']:
            relevance = RelevanceLevel.HIGH
            reasons.append("Wikipedia内容高度相关")
        elif confidence >= self.thresholds['medium']:
            relevance = RelevanceLevel.MEDIUM
            reasons.append("Wikipedia内容中等相关")
        else:
            relevance = RelevanceLevel.LOW
            issues.append("Wikipedia内容相关性较低")
        
        return RelevanceScore(
            source="wikipedia",
            content=wiki_summary[:200] + "..." if len(wiki_summary) > 200 else wiki_summary,
            relevance=relevance,
            confidence=confidence,
            reasons=reasons,
            issues=issues,
            semantic_match=semantic_score,
            entity_match=entity_score,
            completeness=completeness
        )
    
    def _calculate_text_similarity(self, query: str, text: str, 
                                   keywords: List[str]) -> float:
        """计算文本相似度（基于关键词重叠）"""
        query_lower = query.lower()
        text_lower = text.lower()
        
        # 1. 字符级重叠
        query_chars = set(query_lower)
        text_chars = set(text_lower)
        char_overlap = len(query_chars & text_chars) / max(len(query_chars), 1)
        
        # 2. 关键词匹配
        keyword_matches = 0
        for keyword in keywords:
            if len(keyword) >= 2 and keyword in text_lower:
                keyword_matches += 1
        
        keyword_score = keyword_matches / max(len(keywords), 1) if keywords else 0
        
        # 3. N-gram 匹配 (bigram)
        query_bigrams = set(query_lower[i:i+2] for i in range(len(query_lower)-1))
        text_bigrams = set(text_lower[i:i+2] for i in range(len(text_lower)-1))
        bigram_overlap = len(query_bigrams & text_bigrams) / max(len(query_bigrams), 1) if query_bigrams else 0
        
        # 加权综合
        similarity = char_overlap * 0.2 + keyword_score * 0.5 + bigram_overlap * 0.3
        
        return min(similarity, 1.0)
    
    def _aggregate_evaluation(self, report: EvaluationReport):
        """聚合各条目的评估结果"""
        all_scores = (
            report.triple_scores + 
            report.document_scores + 
            ([report.wiki_score] if report.wiki_score else [])
        )
        
        report.total_results = len(all_scores)
        
        # 统计各等级数量
        for score in all_scores:
            if score.relevance == RelevanceLevel.HIGH:
                report.high_relevant_count += 1
            elif score.relevance == RelevanceLevel.MEDIUM:
                report.medium_relevant_count += 1
            elif score.relevance == RelevanceLevel.LOW:
                report.low_relevant_count += 1
            else:
                report.irrelevant_count += 1
        
        # 计算整体相关性和置信度
        if all_scores:
            avg_confidence = sum(s.confidence for s in all_scores) / len(all_scores)
            report.overall_confidence = avg_confidence
            
            # 整体相关性 = 加权平均，高相关的权重更高
            weights = {
                RelevanceLevel.HIGH: 1.0,
                RelevanceLevel.MEDIUM: 0.6,
                RelevanceLevel.LOW: 0.3,
                RelevanceLevel.IRRELEVANT: 0.0,
                RelevanceLevel.UNKNOWN: 0.5
            }
            
            weighted_sum = sum(
                weights.get(s.relevance, 0.5) * s.confidence 
                for s in all_scores
            ) / len(all_scores)
            
            if weighted_sum >= self.thresholds['high']:
                report.overall_relevance = RelevanceLevel.HIGH
            elif weighted_sum >= self.thresholds['medium']:
                report.overall_relevance = RelevanceLevel.MEDIUM
            elif weighted_sum >= self.thresholds['low']:
                report.overall_relevance = RelevanceLevel.LOW
            else:
                report.overall_relevance = RelevanceLevel.IRRELEVANT
            
            # 知识充足度
            relevant_count = report.high_relevant_count + report.medium_relevant_count
            report.knowledge_sufficiency = min(relevant_count / max(len(all_scores), 1), 1.0)
    
    def _make_decision(self, report: EvaluationReport):
        """基于评估结果做出行动决策"""
        high_count = report.high_relevant_count
        medium_count = report.medium_relevant_count
        relevant_total = high_count + medium_count
        
        # 无检索结果
        if report.total_results == 0:
            report.action = ActionDecision.GENERATE_DIRECT
            report.action_reasons.append("无检索结果，直接生成")
            report.suggestions.append("建议：扩展查询关键词或使用对话历史")
            return
        
        # 知识充足度判断
        if report.knowledge_sufficiency >= 0.5 and report.overall_confidence >= 0.5:
            report.action = ActionDecision.USE_RETRIEVAL
            report.action_reasons.append(
                f"检索质量良好 ({relevant_total} 条相关结果，置信度 {report.overall_confidence:.2f})"
            )
            return
        
        # 低相关性
        if report.low_relevant_count > relevant_total:
            report.action = ActionDecision.USE_RETRIEVAL  # 仍然使用，但会降低权重
            report.action_reasons.append(
                "检索结果相关性较低，但仍可作为参考"
            )
            report.suggestions.append(
                "建议：尝试其他表述或更换知识源"
            )
            return
        
        # 无高相关结果
        if high_count == 0 and medium_count == 0:
            report.action = ActionDecision.GENERATE_DIRECT
            report.action_reasons.append("无高相关检索结果，依赖模型自身知识")
            report.suggestions.append(
                "建议：调整查询表述，或启用迭代检索"
            )
            return
        
        # 默认使用检索结果
        report.action = ActionDecision.USE_RETRIEVAL
    
    def get_filtered_results(self, report: EvaluationReport,
                             min_relevance: RelevanceLevel = RelevanceLevel.LOW) -> Dict[str, Any]:
        """
        获取过滤后的结果（只保留相关性达标的结果）
        
        Args:
            report: 评估报告
            min_relevance: 最低相关性要求
            
        Returns:
            过滤后的结果字典
        """
        # 阈值映射
        threshold_map = {
            RelevanceLevel.HIGH: 1.0,
            RelevanceLevel.MEDIUM: 0.4,
            RelevanceLevel.LOW: 0.2,
            RelevanceLevel.IRRELEVANT: 0.0,
            RelevanceLevel.UNKNOWN: 0.0
        }
        
        min_confidence = threshold_map.get(min_relevance, 0.0)
        
        # 过滤三元组
        filtered_triples = [
            (s.content.split(' ', 2)[0], 
             s.content.split(' ', 2)[1] if len(s.content.split(' ', 2)) > 1 else '',
             s.content.split(' ', 2)[2] if len(s.content.split(' ', 2)) > 2 else '')
            for s in report.triple_scores
            if s.confidence >= min_confidence
        ]
        
        # 过滤文档
        filtered_docs = [
            s.content for s in report.document_scores
            if s.confidence >= min_confidence
        ]
        
        # Wiki 只在有结果时保留
        filtered_wiki = report.wiki_score.content if (
            report.wiki_score and report.wiki_score.confidence >= min_confidence
        ) else None
        
        return {
            "triples": filtered_triples,
            "documents": filtered_docs,
            "wiki_summary": filtered_wiki,
            "total_kept": len(filtered_triples) + len(filtered_docs) + (1 if filtered_wiki else 0)
        }