"""RRF 融合算法：多源异构检索结果的融合排序与去重，为重排序提供候选集。"""

import time
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from collections import defaultdict

_SCORE_RANGES = (
    (0.8, 1.0, "0.8-1.0"),
    (0.6, 0.8, "0.6-0.8"),
    (0.4, 0.6, "0.4-0.6"),
    (0.2, 0.4, "0.2-0.4"),
    (0.0, 0.2, "0.0-0.2"),
)


def _truncate(text: str, limit: int = 100) -> str:
    """截断文本，超长时追加省略号"""
    return text[:limit] + "..." if len(text) > limit else text


def _jaccard(set1: set, set2: set) -> float:
    """Jaccard 相似度"""
    union = len(set1 | set2)
    return len(set1 & set2) / union if union else 0.0


class ResultType(Enum):
    """结果类型枚举"""
    TRIPLE = "triple"
    DOCUMENT = "document"
    WIKI = "wiki"


@dataclass
class FusionItem:
    """融合后的单条结果项：内容、来源、排名与融合得分"""
    item_id: str                      # 内容哈希ID，用于去重
    result_type: ResultType           # 结果类型
    content: str                      # 原始内容文本
    summary: str = ""                 # 摘要（去重后合并使用）
    source: str = ""                  # 主要来源标识 (kg/vector/wiki)
    source_details: Dict[str, Any] = field(default_factory=dict)
    original_ranks: Dict[str, int] = field(default_factory=dict)   # {source: rank}
    original_scores: Dict[str, float] = field(default_factory=dict)  # {source: score}
    rrf_score: float = 0.0            # RRF 融合得分
    combined_score: float = 0.0       # 综合得分 (RRF + 语义相似度)
    entities: List[str] = field(default_factory=list)
    relations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.item_id)

    def merge_with(self, other: 'FusionItem'):
        """合并另一条同类型结果（去重时调用）"""
        # 保留内容更长的摘要
        if len(other.content) > len(self.content):
            self.content = other.content
            self.summary = other.summary

        # 合并来源排名与得分
        for src, rank in other.original_ranks.items():
            if src not in self.original_ranks or rank < self.original_ranks[src]:
                self.original_ranks[src] = rank

        for src, score in other.original_scores.items():
            self.original_scores[src] = max(self.original_scores.get(src, score), score)

        # 合并实体、关系和元数据
        self.entities = list(set(self.entities + other.entities))
        self.relations = list(set(self.relations + other.relations))
        self.metadata.update(other.metadata)


@dataclass
class FusionResult:
    """融合结果聚合：结果列表与统计信息"""
    items: List[FusionItem] = field(default_factory=list)
    total_input: int = 0               # 输入结果总数
    total_output: int = 0              # 输出结果总数
    duplicates_removed: int = 0        # 去重数量
    source_contributions: Dict[str, int] = field(default_factory=dict)
    score_distribution: Dict[str, int] = field(default_factory=dict)
    query: str = ""
    fusion_time: float = 0.0

    def get_top_k(self, k: int) -> List[FusionItem]:
        """获取前 k 条结果"""
        return self.items[:k]

    def get_by_type(self, result_type: ResultType) -> List[FusionItem]:
        """获取指定类型的结果"""
        return [item for item in self.items if item.result_type == result_type]

    def get_summary(self) -> Dict[str, Any]:
        """获取摘要"""
        return {
            "total_input": self.total_input,
            "total_output": self.total_output,
            "duplicates_removed": self.duplicates_removed,
            "source_contributions": self.source_contributions,
            "score_distribution": self.score_distribution,
            "top_5_scores": [item.rrf_score for item in self.items[:5]] if self.items else []
        }


class RRFusion:
    """
    RRF 融合器 - Reciprocal Rank Fusion

    RRF 公式: RRF(d) = Σ 1/(k + rank(d))，k 为平滑因子（通常60）。
    """

    def __init__(self, k: float = 60.0, enable_deduplication: bool = True,
                 dedup_threshold: float = 0.85):
        self.k = k
        self.enable_deduplication = enable_deduplication
        self.dedup_threshold = dedup_threshold

        self.source_weights = {
            'kg': 1.0,
            'vector': 1.0,
            'wiki': 1.0,
        }

    def fuse(self, query: str, source_results: Dict[str, List[Any]],
             source_scores: Optional[Dict[str, List[float]]] = None) -> FusionResult:
        """
        融合多源检索结果

        Args:
            query: 原始查询
            source_results: {'kg': [triple, ...], 'vector': [doc, ...], 'wiki': summary}
            source_scores: 各来源的得分（可选）
        """
        start_time = time.time()

        result = FusionResult(query=query)
        result.total_input = sum(len(v) if isinstance(v, list) else 1
                                 for v in source_results.values())

        all_items: List[FusionItem] = []
        for source, items in source_results.items():
            if not items:
                continue

            # Wikipedia 为单条结果（非列表时包装）
            if source == 'wiki' and items:
                items_list = [items]
            elif isinstance(items, list):
                items_list = items
            else:
                items_list = [items]

            scores = source_scores.get(source, []) if source_scores else []
            for idx, item in enumerate(items_list):
                all_items.append(self._create_fusion_item(
                    item, source, idx, scores[idx] if idx < len(scores) else 0.5
                ))

        if self.enable_deduplication:
            all_items, dup_count = self._deduplicate(all_items)
            result.duplicates_removed = dup_count

        fused_items = self._apply_rrf(all_items)

        for item in fused_items:
            item.combined_score = item.rrf_score
        fused_items.sort(key=lambda x: x.combined_score, reverse=True)

        result.items = fused_items
        result.total_output = len(fused_items)

        result.source_contributions = self._count_contributions(fused_items)
        result.score_distribution = self._calculate_score_distribution(fused_items)
        result.fusion_time = time.time() - start_time

        return result

    def _create_fusion_item(self, item: Any, source: str,
                            rank: int, score: float) -> FusionItem:
        """根据不同来源创建 FusionItem"""
        entities: List[str] = []
        relations: List[str] = []

        if source == 'kg' and isinstance(item, tuple):
            subject, predicate, obj = item
            content = f"{subject} {predicate} {obj}"
            summary = f"({subject}, {predicate}, {obj})"
            entities = [subject, obj]
            relations = [predicate]
            result_type = ResultType.TRIPLE
        elif source == 'vector' and isinstance(item, str):
            content = item
            summary = _truncate(item)
            result_type = ResultType.DOCUMENT
        elif source == 'wiki':
            content = item.get('summary', str(item)) if isinstance(item, dict) else str(item)
            summary = _truncate(content)
            result_type = ResultType.WIKI
        else:
            content = str(item)
            summary = _truncate(content)
            result_type = ResultType.DOCUMENT

        return FusionItem(
            item_id=self._generate_id(content),
            result_type=result_type,
            content=content,
            summary=summary,
            source=source,
            original_ranks={source: rank + 1},  # 排名从1开始
            original_scores={source: score},
            entities=entities,
            relations=relations,
            metadata={"original_rank": rank}
        )

    def _generate_id(self, content: str) -> str:
        """生成内容唯一ID（内容哈希）"""
        return hashlib.md5(content.lower().strip().encode('utf-8')).hexdigest()[:16]

    def _deduplicate(self, items: List[FusionItem]) -> Tuple[List[FusionItem], int]:
        """基于内容ID与相似度去重，返回 (去重后列表, 去重数量)"""
        if not items:
            return [], 0

        # 按类型分组
        type_groups: Dict[ResultType, List[FusionItem]] = defaultdict(list)
        for item in items:
            type_groups[item.result_type].append(item)

        result = []
        total_removed = 0

        for group in type_groups.values():
            seen: List[FusionItem] = []
            seen_char_sets: List[set] = []  # 与 seen 一一对应的字符集合

            for item in group:
                item_char_set = set(item.content.lower())
                is_duplicate = False

                for existing, existing_set in zip(seen, seen_char_sets):
                    if item.item_id == existing.item_id:
                        is_duplicate = True
                    elif self.enable_deduplication and \
                            _jaccard(item_char_set, existing_set) >= self.dedup_threshold:
                        is_duplicate = True

                    if is_duplicate:
                        existing.merge_with(item)
                        total_removed += 1
                        break

                if not is_duplicate:
                    seen.append(item)
                    seen_char_sets.append(item_char_set)

            result.extend(seen)

        return result, total_removed

    def _apply_rrf(self, items: List[FusionItem]) -> List[FusionItem]:
        """应用 RRF 算法: RRF(d) = Σ weight_source / (k + rank_source(d))"""
        item_groups: Dict[str, List[FusionItem]] = defaultdict(list)
        for item in items:
            item_groups[item.item_id].append(item)

        fused_map: Dict[str, FusionItem] = {}

        for item_id, group in item_groups.items():
            rrf_score = 0.0
            for item in group:
                weight = self.source_weights.get(item.source, 1.0)
                for source, rank in item.original_ranks.items():
                    source_weight = self.source_weights.get(source, 1.0)
                    rrf_score += source_weight / (self.k + rank) * weight

            representative = group[0]
            representative.rrf_score = rrf_score
            fused_map[item_id] = representative

        return list(fused_map.values())

    def _count_contributions(self, items: List[FusionItem]) -> Dict[str, int]:
        """统计各来源的贡献数量"""
        contributions = defaultdict(int)
        for item in items:
            for source in item.original_ranks.keys():
                contributions[source] += 1
        return dict(contributions)

    def _calculate_score_distribution(self, items: List[FusionItem]) -> Dict[str, int]:
        """计算得分分布"""
        if not items:
            return {}

        distribution = {label: 0 for _, _, label in _SCORE_RANGES}
        for item in items:
            for min_val, max_val, label in _SCORE_RANGES:
                if min_val <= item.rrf_score < max_val:
                    distribution[label] += 1
                    break
        return distribution

    def set_source_weight(self, source: str, weight: float):
        """设置来源权重 (kg/vector/wiki)"""
        if weight > 0:
            self.source_weights[source] = weight


class MultiRoundRetrieval:
    """
    多轮检索策略控制器 (Self-RAG)：

    1. 首轮粗召回：多源并行检索
    2. RRF 融合：整合多源结果
    3. 二轮精排：Cohere 语义重排序
    """

    def __init__(self, k: float = 60.0, enable_dedup: bool = True):
        self.fusion_engine = RRFusion(k=k, enable_deduplication=enable_dedup)
        self.reranker = None
        self._enable_rerank = True

    def set_reranker(self, reranker):
        """设置重排序器"""
        self.reranker = reranker

    def enable_reranking(self, enabled: bool):
        """启用/禁用重排序"""
        self._enable_rerank = enabled

    def execute_round1_coarse(self, query: str,
                              kg_results: List[Tuple],
                              vector_results: List[str],
                              wiki_result: Optional[str] = None,
                              vector_scores: Optional[List[float]] = None
                              ) -> FusionResult:
        """第一轮：粗召回与 RRF 融合"""
        source_results = {
            'kg': kg_results,
            'vector': vector_results,
            'wiki': wiki_result if wiki_result else ""
        }
        source_scores = {'vector': vector_scores} if vector_scores else {}
        return self.fusion_engine.fuse(query, source_results, source_scores)

    def execute_round2_rerank(self, query: str,
                              candidates: List[FusionItem],
                              top_k: int = 10) -> List[FusionItem]:
        """第二轮：语义重排序"""
        if not self._enable_rerank or not self.reranker or not candidates:
            return candidates[:top_k]

        try:
            return self.reranker.rerank(query, candidates, top_k=top_k)
        except Exception as e:
            print(f"[多轮检索] 重排序失败: {e}")
            return candidates[:top_k]

    def execute_full_pipeline(self, query: str,
                              kg_results: List[Tuple],
                              vector_results: List[str],
                              wiki_result: Optional[str] = None,
                              vector_scores: Optional[List[float]] = None,
                              final_top_k: int = 10) -> Dict[str, Any]:
        """执行完整的多轮检索流程，返回融合结果与最终结果"""
        fusion_result = self.execute_round1_coarse(
            query=query,
            kg_results=kg_results,
            vector_results=vector_results,
            wiki_result=wiki_result,
            vector_scores=vector_scores
        )
        candidates = fusion_result.items

        if self._enable_rerank and self.reranker:
            final_items = self.execute_round2_rerank(
                query, candidates, top_k=final_top_k
            )
        else:
            final_items = candidates[:final_top_k]

        return {
            "fusion_result": fusion_result,
            "final_results": final_items,
            "total_candidates": len(candidates),
            "final_count": len(final_items),
            "round1_time": fusion_result.fusion_time,
            "round2_enabled": self._enable_rerank and self.reranker is not None
        }


def fuse_results(query: str, source_results: Dict[str, List],
                 source_scores: Optional[Dict[str, List[float]]] = None,
                 k: float = 60.0) -> FusionResult:
    """便捷函数：执行 RRF 融合"""
    return RRFusion(k=k).fuse(query, source_results, source_scores)


def deduplicate_items(items: List[FusionItem],
                      threshold: float = 0.85) -> List[FusionItem]:
    """便捷函数：对结果列表去重"""
    fusion = RRFusion(k=60.0, enable_deduplication=True, dedup_threshold=threshold)
    deduped, _ = fusion._deduplicate(items)
    return deduped
