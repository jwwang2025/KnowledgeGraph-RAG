"""结果评估与反思模块：基于 Self-RAG 思想评估检索结果相关性并决定下一步行动。"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum


class RelevanceLevel(Enum):
    """相关性等级枚举"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    IRRELEVANT = "irrelevant"
    UNKNOWN = "unknown"


class ActionDecision(Enum):
    """行动决策枚举 (Self-RAG 核心)"""
    USE_RETRIEVAL = "use_retrieval"

    USE_PARAMETRIC = "use_parametric"
    USE_HISTORY = "use_history"
    GENERATE_DIRECT = "generate_direct"

    ITERATE_RETRIEVAL = "iterate_retrieval"
    EXPAND_ENTITIES = "expand_entities"
    SWITCH_SOURCE = "switch_source"

    REFUSE = "refuse"


_STOPWORDS = frozenset('的是在了和与或有吗呢吧啊')
_PUNCT_CHARS = frozenset('，。！？、；：""''（）')

_ENTITY_SUFFIXES = {
    '人物': ['人', '者', '家', '师', '员', '生'],
    '组织': ['公司', '机构', '组织', '大学', '医院', '党', '会'],
    '地点': ['国', '省', '市', '县', '区', '镇', '村', '山', '河', '湖', '海'],
    '概念': ['主义', '论', '学', '理论', '方法', '技术'],
}

_RELEVANCE_WEIGHTS = {
    RelevanceLevel.HIGH: 1.0,
    RelevanceLevel.MEDIUM: 0.6,
    RelevanceLevel.LOW: 0.3,
    RelevanceLevel.IRRELEVANT: 0.0,
    RelevanceLevel.UNKNOWN: 0.5,
}

_FILTER_THRESHOLDS = {
    RelevanceLevel.HIGH: 1.0,
    RelevanceLevel.MEDIUM: 0.4,
    RelevanceLevel.LOW: 0.2,
    RelevanceLevel.IRRELEVANT: 0.0,
    RelevanceLevel.UNKNOWN: 0.0,
}


def _truncate(text: str, limit: int = 200) -> str:
    """截断过长文本用于摘要展示"""
    return text[:limit] + "..." if len(text) > limit else text


@dataclass
class RelevanceScore:
    """单条检索结果的相关性评分"""
    source: str                    # 来源
    content: str                   # 内容摘要
    relevance: RelevanceLevel      # 相关性等级
    confidence: float              # 置信度 (0-1)
    reasons: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)

    semantic_match: float = 0.0
    entity_match: float = 0.0
    completeness: float = 0.0


@dataclass
class EvaluationReport:
    """评估报告"""
    query: str
    question_type: str

    triple_scores: List[RelevanceScore] = field(default_factory=list)
    document_scores: List[RelevanceScore] = field(default_factory=list)
    wiki_score: Optional[RelevanceScore] = None

    overall_relevance: RelevanceLevel = RelevanceLevel.UNKNOWN
    overall_confidence: float = 0.0
    knowledge_sufficiency: float = 0.0

    action: ActionDecision = ActionDecision.USE_RETRIEVAL
    action_reasons: List[str] = field(default_factory=list)

    suggestions: List[str] = field(default_factory=list)
    alternative_approaches: List[str] = field(default_factory=list)

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
    """结果评估器：基于 Self-RAG 反思机制评估检索结果的质量和相关性"""

    def __init__(self, use_llm_evaluation: bool = False):
        """use_llm_evaluation: 是否使用LLM进行深度评估 (暂未实现)"""
        self.use_llm_evaluation = use_llm_evaluation

        self.thresholds = {
            'high': 0.7,      # >= 0.7 为高度相关
            'medium': 0.4,   # >= 0.4 为中等相关
            'low': 0.2,       # >= 0.2 为低相关
        }

    def evaluate(self, query: str, question_type: str,
                 triples: List[Tuple], documents: List[str],
                 wiki_summary: Optional[str] = None) -> EvaluationReport:
        """评估检索结果，返回评估报告"""
        report = EvaluationReport(query=query, question_type=question_type)

        query_keywords = self._extract_keywords(query)
        query_entities = self._extract_entities(query)

        report.triple_scores.extend(
            self._evaluate_triple(t, query, query_keywords, query_entities)
            for t in triples
        )
        report.document_scores.extend(
            self._evaluate_document(d, query, query_keywords, query_entities)
            for d in documents
        )
        if wiki_summary:
            report.wiki_score = self._evaluate_wiki(
                wiki_summary, query, query_keywords, query_entities
            )

        self._aggregate_evaluation(report)
        self._make_decision(report)

        return report

    def _extract_keywords(self, query: str) -> List[str]:
        """提取查询关键词"""
        keywords = []
        for i in range(len(query)):
            for length in [4, 3, 2]:
                if i + length <= len(query):
                    word = query[i:i+length]
                    if word not in _STOPWORDS and not (_PUNCT_CHARS & set(word)):
                        keywords.append(word)

        return list(set(keywords))

    def _extract_entities(self, query: str) -> List[str]:
        """提取查询中的实体（简单基于规则）"""
        entities = []

        for suffixes in _ENTITY_SUFFIXES.values():
            for suffix in suffixes:
                entities.extend(re.findall(f'.*{suffix}', query))

        entities.extend(re.findall(r'[""]([^""]+)[""]', query))
        entities.extend(re.findall(r'[「』]([^「」]+)[「」]', query))

        return list(set(entities))

    def _grade_relevance(self, confidence: float, prefix: str = "",
                         irrelevant_issue: str = "不相关") -> Tuple[RelevanceLevel, str]:
        """根据置信度确定相关性等级，返回 (等级, 对应的理由/问题标注)"""
        if confidence >= self.thresholds['high']:
            return RelevanceLevel.HIGH, f"{prefix}高度相关"
        if confidence >= self.thresholds['medium']:
            return RelevanceLevel.MEDIUM, f"{prefix}中等相关"
        if confidence >= self.thresholds['low']:
            return RelevanceLevel.LOW, f"{prefix}低相关"
        return RelevanceLevel.IRRELEVANT, irrelevant_issue

    def _evaluate_triple(self, triple: Tuple, query: str,
                        keywords: List[str], entities: List[str]) -> RelevanceScore:
        """评估单个三元组的相关性"""
        subject, predicate, obj = triple
        triple_text = f"{subject}{predicate}{obj}"
        triple_full = f"{subject} {predicate} {obj}"

        reasons = []
        issues = []

        # 1. 语义匹配度
        semantic_score = self._calculate_text_similarity(query, triple_text, keywords)

        # 2. 实体匹配度
        entity_score = 0.0
        for entity in entities:
            if entity in triple_text:
                entity_score += 0.5

        if subject in query:
            entity_score += 0.3
            reasons.append(f"主体匹配: {subject}")
        if obj in query:
            entity_score += 0.2
            reasons.append(f"客体匹配: {obj}")

        entity_score = min(entity_score, 1.0)

        # 3. 完整性评分
        completeness = 1.0
        for part, label in ((subject, "主体"), (predicate, "关系"), (obj, "客体")):
            if not part or part == "Unknown":
                completeness *= 0.5
                issues.append(f"{label}缺失或未知")

        confidence = (semantic_score * 0.4 + entity_score * 0.4 + completeness * 0.2)

        relevance, note = self._grade_relevance(confidence)
        if relevance == RelevanceLevel.IRRELEVANT:
            issues.append(note)
        else:
            reasons.append(note)

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

        # 2. 实体匹配度 + 关键词命中
        entity_score = sum(0.5 for entity in entities if entity in doc)
        keyword_hits = sum(1 for kw in keywords if len(kw) >= 2 and kw in doc)
        entity_score = min(entity_score + keyword_hits * 0.1, 1.0)

        # 3. 完整性 - 文档长度适中为佳
        completeness = min(len(doc) / 500, 1.0) if len(doc) > 0 else 0.0

        confidence = semantic_score * 0.5 + entity_score * 0.3 + completeness * 0.2

        # 长度过短可能是噪音
        if len(doc) < 20:
            confidence *= 0.7
            issues.append("文档过短，可能不完整")

        relevance, note = self._grade_relevance(confidence, "文档")
        if relevance == RelevanceLevel.IRRELEVANT:
            issues.append(note)
        else:
            reasons.append(note)

        return RelevanceScore(
            source="vector_database",
            content=_truncate(doc),
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

        semantic_score = self._calculate_text_similarity(query, wiki_summary, keywords)
        entity_score = min(sum(0.5 for entity in entities if entity in wiki_summary), 1.0)

        # Wikipedia 通常比较权威，完整性较高
        completeness = 0.9

        confidence = semantic_score * 0.4 + entity_score * 0.4 + completeness * 0.2

        relevance, note = self._grade_relevance(confidence, "Wikipedia内容")
        # Wiki 低于中等即视为低相关，仅记录问题不记录理由
        if relevance == RelevanceLevel.IRRELEVANT:
            relevance = RelevanceLevel.LOW
        if relevance == RelevanceLevel.LOW:
            issues.append("Wikipedia内容相关性较低")
        else:
            reasons.append(note)

        return RelevanceScore(
            source="wikipedia",
            content=_truncate(wiki_summary),
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
        char_overlap = len(query_chars & set(text_lower)) / max(len(query_chars), 1)

        # 2. 关键词匹配
        if keywords:
            keyword_score = sum(1 for kw in keywords if len(kw) >= 2 and kw in text_lower) / len(keywords)
        else:
            keyword_score = 0

        # 3. N-gram 匹配 (bigram)
        query_bigrams = set(query_lower[i:i+2] for i in range(len(query_lower)-1))
        text_bigrams = set(text_lower[i:i+2] for i in range(len(text_lower)-1))
        bigram_overlap = len(query_bigrams & text_bigrams) / max(len(query_bigrams), 1) if query_bigrams else 0

        return min(char_overlap * 0.2 + keyword_score * 0.5 + bigram_overlap * 0.3, 1.0)

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
            report.overall_confidence = sum(s.confidence for s in all_scores) / len(all_scores)
            weighted_sum = sum(
                _RELEVANCE_WEIGHTS[s.relevance] * s.confidence
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
            report.knowledge_sufficiency = min(relevant_count / len(all_scores), 1.0)

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
            report.action_reasons.append("检索结果相关性较低，但仍可作为参考")
            report.suggestions.append("建议：尝试其他表述或更换知识源")
            return

        # 无高相关结果
        if high_count == 0 and medium_count == 0:
            report.action = ActionDecision.GENERATE_DIRECT
            report.action_reasons.append("无高相关检索结果，依赖模型自身知识")
            report.suggestions.append("建议：调整查询表述，或启用迭代检索")
            return

        # 默认使用检索结果
        report.action = ActionDecision.USE_RETRIEVAL

    def get_filtered_results(self, report: EvaluationReport,
                             min_relevance: RelevanceLevel = RelevanceLevel.LOW) -> Dict[str, Any]:
        """获取过滤后的结果（只保留相关性达标的结果）"""
        min_confidence = _FILTER_THRESHOLDS.get(min_relevance, 0.0)

        # 过滤三元组（content 为 "subject predicate obj" 格式）
        filtered_triples = []
        for s in report.triple_scores:
            if s.confidence >= min_confidence:
                parts = s.content.split(' ', 2)
                filtered_triples.append((
                    parts[0],
                    parts[1] if len(parts) > 1 else '',
                    parts[2] if len(parts) > 2 else ''
                ))

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
