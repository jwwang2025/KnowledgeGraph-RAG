"""Cohere 语义重排序模块：语义级精确重排序（Cohere API + 本地备选），用于多轮检索第二轮。"""

import os
import re
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum

from .fusion import FusionItem, ResultType


class RerankModel(Enum):
    """重排序模型枚举"""
    COHERE = "cohere"                  # Cohere API
    LOCAL_SIMILARITY = "local"         # 本地相似度计算
    HYBRID = "hybrid"                   # 混合模式


# 本地重排序的停用词
_LOCAL_STOPWORDS = frozenset('的是在了和与或有')

# 结果类型权重（三元组通常更精确）与排序优先级
_TYPE_WEIGHTS = {ResultType.TRIPLE: 1.2, ResultType.WIKI: 0.9}
_TYPE_PRIORITY = {ResultType.TRIPLE: 0, ResultType.WIKI: 1, ResultType.DOCUMENT: 2}

# 类型多样性配置
_DIVERSITY_TYPES = (ResultType.TRIPLE, ResultType.DOCUMENT, ResultType.WIKI)
_MAX_PER_TYPE = 5


@dataclass
class RerankResult:
    """单条重排序结果：重排序位置与语义相关度得分"""
    item: FusionItem                    # 原始 FusionItem
    rerank_score: float = 0.0         # 重排序得分 (0-1)
    rerank_position: int = 0           # 重排序后的位置 (从0开始)

    # 语义分析信息
    semantic_match: float = 0.0        # 语义匹配度
    keyword_match: float = 0.0        # 关键词匹配度
    context_relevance: float = 0.0    # 上下文相关度

    # 元数据
    reasoning: str = ""                 # 重排序理由
    confidence: float = 0.0            # 置信度

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "content": self.item.content,
            "type": self.item.result_type.value,
            "source": self.item.source,
            "rerank_score": self.rerank_score,
            "rerank_position": self.rerank_position,
            "semantic_match": self.semantic_match,
            "keyword_match": self.keyword_match,
            "context_relevance": self.context_relevance,
            "reasoning": self.reasoning,
            "confidence": self.confidence
        }


@dataclass
class RerankReport:
    """重排序报告：完整的重排序过程和结果"""
    query: str                          # 原始查询
    model_used: RerankModel             # 使用的重排序模型

    # 输入输出
    input_count: int = 0               # 输入候选数量
    output_count: int = 0              # 输出结果数量

    # 得分统计
    score_distribution: Dict[str, float] = field(default_factory=dict)
    avg_score: float = 0.0             # 平均得分
    max_score: float = 0.0             # 最高得分
    min_score: float = 0.0              # 最低得分

    # 位置变化统计
    position_changes: Dict[str, int] = field(default_factory=dict)  # 上升/下降数量

    # 结果
    results: List[RerankResult] = field(default_factory=list)

    # 性能
    rerank_time: float = 0.0
    api_calls: int = 0

    def get_summary(self) -> Dict[str, Any]:
        """获取摘要"""
        return {
            "query": self.query,
            "model_used": self.model_used.value,
            "input_count": self.input_count,
            "output_count": self.output_count,
            "avg_score": f"{self.avg_score:.3f}",
            "max_score": f"{self.max_score:.3f}",
            "rerank_time": f"{self.rerank_time:.3f}s",
            "api_calls": self.api_calls,
            "top_5_scores": [r.rerank_score for r in self.results[:5]]
        }


class CohereReranker:
    """Cohere 语义重排序器：语义理解能力强，与 RRF 融合结果配合使用"""

    def __init__(self, api_key: Optional[str] = None,
                 model: str = "rerank-multilingual-v3.0",
                 enable_local_fallback: bool = True):
        """api_key 默认从环境变量 COHERE_API_KEY 获取"""
        self.api_key = api_key or os.environ.get("COHERE_API_KEY")
        self.model = model
        self.enable_local_fallback = enable_local_fallback
        self._client = None

        # 配置
        self.truncation = True  # 是否截断过长文本

    @property
    def client(self):
        """延迟加载 Cohere 客户端"""
        if self._client is None and self.api_key:
            try:
                import cohere
                self._client = cohere.Client(self.api_key)
            except ImportError:
                print("[CohereReranker] 未安装 cohere 库，将使用本地备选方案")
                self._client = None
        return self._client

    def rerank(self, query: str, candidates: List[FusionItem],
               top_k: int = 10, return_documents: bool = True
               ) -> List[FusionItem]:
        """对候选结果进行语义重排序，返回重排序后的结果列表"""
        if not candidates:
            return []

        # 限制候选数量以节省 API 调用成本
        candidates = candidates[:min(len(candidates), 100)]

        # 尝试使用 Cohere API，不可用时回退本地方案
        if self.client and self.api_key:
            try:
                results = self._rerank_with_cohere(query, candidates, top_k, return_documents)
            except Exception as e:
                print(f"[CohereReranker] API 调用失败: {e}")
                results = self._rerank_locally(query, candidates, top_k)
        else:
            results = self._rerank_locally(query, candidates, top_k)

        # 更新结果位置
        for idx, item in enumerate(results):
            item.rerank_position = idx

        return results

    def _rerank_with_cohere(self, query: str, candidates: List[FusionItem],
                           top_k: int, return_documents: bool) -> List[FusionItem]:
        """使用 Cohere API 进行重排序"""
        documents = [item.content for item in candidates]

        response = self.client.rerank(
            query=query,
            documents=documents,
            model=self.model,
            top_n=top_k,
            return_documents=return_documents
        )

        # 使用 Cohere 的相关性得分创建副本（避免修改原列表）
        results = []
        for result in response.results:
            rerank_score = result.relevance_score
            new_item = deepcopy(candidates[result.index])
            new_item.rrf_score = rerank_score
            new_item.combined_score = rerank_score
            results.append(new_item)

        # 按得分降序排列
        results.sort(key=lambda x: x.rrf_score, reverse=True)

        return results[:top_k]

    def _rerank_locally(self, query: str, candidates: List[FusionItem],
                       top_k: int) -> List[FusionItem]:
        """本地轻量级重排序：关键词匹配 + 实体相似度，作为 API 不可用时的备选"""
        scored_items = []

        for idx, item in enumerate(candidates):
            # 1. 关键词匹配得分
            keyword_score = self._calculate_keyword_match(query, item.content)

            # 2. 实体匹配得分
            entity_score = self._calculate_entity_match(query, item.content, item.entities)

            # 3. 类型权重 + 4. 综合得分
            type_weight = _TYPE_WEIGHTS.get(item.result_type, 1.0)
            local_score = keyword_score * 0.4 + entity_score * 0.4 + type_weight * 0.2

            # 结合原始 RRF 得分
            combined_score = item.rrf_score * 0.3 + local_score * 0.7

            # 创建副本并更新得分
            new_item = deepcopy(item)
            new_item.rrf_score = combined_score
            new_item.combined_score = combined_score

            scored_items.append((combined_score, idx, new_item))

        # 按得分降序排列，返回前 k 条
        scored_items.sort(key=lambda x: x[0], reverse=True)

        return [item for _, _, item in scored_items[:top_k]]

    def _calculate_keyword_match(self, query: str, content: str) -> float:
        """计算关键词匹配得分（基于查询 n-gram 在内容中的出现情况）"""
        # 提取查询关键词 (2-4字词)
        query_keywords = []
        for i in range(len(query)):
            for length in [4, 3, 2]:
                if i + length <= len(query):
                    word = query[i:i+length]
                    if word not in _LOCAL_STOPWORDS:
                        query_keywords.append(word)

        query_keywords = list(set(query_keywords))
        if not query_keywords:
            return 0.5

        content_lower = content.lower()
        matches = sum(1 for kw in query_keywords if kw in content_lower)

        return min(matches / len(query_keywords), 1.0)

    def _calculate_entity_match(self, query: str, content: str,
                               entities: List[str]) -> float:
        """计算实体匹配得分：检查内容中是否包含查询中的实体"""
        content_lower = content.lower()

        if entities:
            entity_score = sum(1 for e in entities if e in content_lower) / len(entities)
        else:
            entity_score = 0.5  # 无实体信息时的默认分

        # 检查查询中的专有名词（连续汉字词）
        proper_nouns = re.findall(r'[\u4e00-\u9fff]{2,}', query)
        if proper_nouns:
            proper_noun_score = sum(1 for pn in proper_nouns if pn in content_lower) / len(proper_nouns)
        else:
            proper_noun_score = 0.5

        return (entity_score + proper_noun_score) / 2

    def rerank_with_report(self, query: str, candidates: List[FusionItem],
                          top_k: int = 10) -> tuple[List[FusionItem], RerankReport]:
        """重排序并返回 (重排序结果, 详细报告)"""
        if not candidates:
            return [], RerankReport(query=query, model_used=RerankModel.LOCAL_SIMILARITY)

        start_time = time.time()

        # 执行重排序
        results = self.rerank(query, candidates, top_k)

        # 构建报告
        report = RerankReport(
            query=query,
            model_used=RerankModel.COHERE if self.client else RerankModel.LOCAL_SIMILARITY,
            input_count=len(candidates),
            output_count=len(results),
            rerank_time=time.time() - start_time
        )

        # 计算统计信息
        if results:
            scores = [item.rrf_score for item in results]
            report.avg_score = sum(scores) / len(scores)
            report.max_score = max(scores)
            report.min_score = min(scores)

            # 得分分布
            report.score_distribution = {
                "high (>0.7)": sum(1 for s in scores if s > 0.7),
                "medium (0.4-0.7)": sum(1 for s in scores if 0.4 <= s <= 0.7),
                "low (<0.4)": sum(1 for s in scores if s < 0.4)
            }

        # 位置变化统计：按原始 RRF 得分降序得到原始排名，用稳定的 item_id 关联
        ranked = sorted(range(len(candidates)),
                        key=lambda i: candidates[i].rrf_score, reverse=True)
        original_rank = {candidates[i].item_id: rank for rank, i in enumerate(ranked)}

        promoted = 0
        demoted = 0
        for idx, item in enumerate(results):
            original_pos = original_rank.get(item.item_id, -1)
            if original_pos > idx:
                promoted += 1
            elif original_pos < idx:
                demoted += 1

        report.position_changes = {"promoted": promoted, "demoted": demoted}

        return results, report


class SelfRAGRefiner:
    """Self-RAG 结果精炼器：过滤低相关结果、保证类型多样性、优化排序"""

    def __init__(self, relevance_threshold: float = 0.3):
        """relevance_threshold: 相关性阈值，低于此阈值的结果将被过滤"""
        self.relevance_threshold = relevance_threshold

    def refine(self, query: str, items: List[FusionItem],
              min_count: int = 3, max_count: int = 10
              ) -> List[FusionItem]:
        """精炼结果"""
        if not items:
            return []

        # 1. 过滤低相关性（过滤后太少则保留阈值以下结果）
        filtered = [item for item in items if item.rrf_score >= self.relevance_threshold]
        if len(filtered) < min_count:
            filtered = items[:max(min_count, len(filtered))]
        else:
            filtered = filtered[:max_count]

        # 2. 确保类型多样性 + 3. 调整顺序
        refined = self._ensure_diversity(filtered)
        refined = self._optimize_order(refined)

        return refined

    def _ensure_diversity(self, items: List[FusionItem]) -> List[FusionItem]:
        """确保结果类型多样性，避免单一类型垄断结果列表"""
        by_type: Dict[ResultType, List[FusionItem]] = {rt: [] for rt in _DIVERSITY_TYPES}
        for item in items:
            by_type[item.result_type].append(item)

        # 每个类型至少保留一个（如果有），最多 _MAX_PER_TYPE 条
        result = []
        for type_items in by_type.values():
            if type_items:
                result.extend(type_items[:_MAX_PER_TYPE])

        result.sort(key=lambda x: x.rrf_score, reverse=True)
        return result

    def _optimize_order(self, items: List[FusionItem]) -> List[FusionItem]:
        """优化结果顺序：高分三元组优先，不同类型按优先级排列"""
        if len(items) <= 2:
            return items

        return sorted(items, key=lambda x: (_TYPE_PRIORITY.get(x.result_type, 3), -x.rrf_score))


# ==================== 便捷函数 ====================

def create_reranker(api_key: Optional[str] = None,
                   model: str = "rerank-multilingual-v3.0",
                   enable_local: bool = True) -> CohereReranker:
    """创建重排序器实例"""
    return CohereReranker(
        api_key=api_key,
        model=model,
        enable_local_fallback=enable_local
    )


def quick_rerank(query: str, candidates: List[FusionItem],
                api_key: Optional[str] = None,
                top_k: int = 10) -> List[FusionItem]:
    """快速重排序便捷函数"""
    reranker = create_reranker(api_key=api_key)
    return reranker.rerank(query, candidates, top_k)
