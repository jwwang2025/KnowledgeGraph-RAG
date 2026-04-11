"""
RRF 融合算法模块 (Reciprocal Rank Fusion)
基于 Self-RAG 的多源异构检索结果融合与去重

核心功能：
1. RRF 算法对多源检索结果进行融合排序
2. 去重处理，消除重复内容
3. 支持多种相关性信号的综合评估
4. 为后续 Cohere 重排序提供高质量候选集
"""

import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
from enum import Enum
from collections import defaultdict


class ResultType(Enum):
    """结果类型枚举"""
    TRIPLE = "triple"           # 知识图谱三元组
    DOCUMENT = "document"       # 文档片段
    WIKI = "wiki"               # Wikipedia 内容


@dataclass
class FusionItem:
    """
    融合后的单条结果项
    
    包含原始内容、来源信息、融合得分等
    """
    # 唯一标识
    item_id: str                      # 内容哈希ID，用于去重
    result_type: ResultType           # 结果类型
    
    # 内容信息
    content: str                      # 原始内容文本
    summary: str = ""                  # 摘要（去重后合并使用）
    
    # 来源信息
    source: str = ""                   # 主要来源标识 (kg/vector/wiki)
    source_details: Dict[str, Any] = field(default_factory=dict)  # 详细来源信息
    
    # 排名信息 (各源中的原始排名)
    original_ranks: Dict[str, int] = field(default_factory=dict)  # {source: rank}
    original_scores: Dict[str, float] = field(default_factory=dict)  # {source: score}
    
    # 融合得分
    rrf_score: float = 0.0            # RRF 融合得分
    combined_score: float = 0.0       # 综合得分 (RRF + 语义相似度)
    
    # 关联信息
    entities: List[str] = field(default_factory=list)  # 关联的实体
    relations: List[str] = field(default_factory=list)  # 关联的关系
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        """支持基于 item_id 的哈希"""
        return hash(self.item_id)
    
    def merge_with(self, other: 'FusionItem'):
        """
        合并另一条同类型结果（去重时调用）
        
        合并策略：
        1. 保留得分最高的原始排名
        2. 累加各来源的得分
        3. 合并元数据
        """
        # 保留得分更高的内容摘要
        if len(other.content) > len(self.content):
            self.content = other.content
            self.summary = other.summary
        
        # 合并来源信息
        for src, rank in other.original_ranks.items():
            if src not in self.original_ranks or rank < self.original_ranks[src]:
                self.original_ranks[src] = rank
        
        for src, score in other.original_scores.items():
            if src not in self.original_scores:
                self.original_scores[src] = score
            else:
                # 累加得分
                self.original_scores[src] = max(self.original_scores[src], score)
        
        # 合并实体和关系
        self.entities = list(set(self.entities + other.entities))
        self.relations = list(set(self.relations + other.relations))
        
        # 合并元数据
        self.metadata.update(other.metadata)


@dataclass
class FusionResult:
    """
    融合结果聚合
    
    包含融合后的结果列表、统计信息等
    """
    # 融合后的结果列表 (按综合得分降序)
    items: List[FusionItem] = field(default_factory=list)
    
    # 统计信息
    total_input: int = 0               # 输入结果总数
    total_output: int = 0             # 输出结果总数
    duplicates_removed: int = 0        # 去重数量
    
    # 各来源贡献统计
    source_contributions: Dict[str, int] = field(default_factory=dict)
    
    # 得分分布
    score_distribution: Dict[str, int] = field(default_factory=dict)  # {range: count}
    
    # 元数据
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
    RRF 融合器 - 实现 Reciprocal Rank Fusion 算法
    
    RRF 公式: RRF(d) = Σ 1/(k + rank(d))
    其中:
    - d: 文档
    - rank(d): 文档在某个排名列表中的排名 (从1开始)
    - k: 平滑因子 (通常设为60)
    
    特点:
    1. 无需学习权重参数
    2. 对各来源平等对待
    3. 排名越靠前的结果权重越高
    """
    
    def __init__(self, k: float = 60.0, enable_deduplication: bool = True,
                 dedup_threshold: float = 0.85):
        """
        初始化 RRF 融合器
        
        Args:
            k: RRF 平滑因子，默认60。值越大，各排名的差异越小
            enable_deduplication: 是否启用去重
            dedup_threshold: 去重相似度阈值 (0-1)，值越大去重越严格
        """
        self.k = k
        self.enable_deduplication = enable_deduplication
        self.dedup_threshold = dedup_threshold
        
        # 融合权重 (可选，用于调整不同来源的重要性)
        self.source_weights = {
            'kg': 1.0,      # 知识图谱
            'vector': 1.0,  # 向量检索
            'wiki': 1.0,    # Wikipedia
        }
    
    def fuse(self, query: str, source_results: Dict[str, List[Any]],
             source_scores: Optional[Dict[str, List[float]]] = None) -> FusionResult:
        """
        融合多源检索结果
        
        Args:
            query: 原始查询
            source_results: 各来源的检索结果
                格式: {
                    'kg': [triple1, triple2, ...],
                    'vector': [doc1, doc2, ...],
                    'wiki': wiki_summary
                }
            source_scores: 各来源的得分 (可选)
                格式: {
                    'kg': [0.9, 0.8, ...],
                    'vector': [0.85, 0.7, ...]
                }
        
        Returns:
            FusionResult: 融合结果
        """
        import time
        start_time = time.time()
        
        result = FusionResult(query=query)
        result.total_input = sum(len(v) if isinstance(v, list) else 1 
                                for v in source_results.values())
        
        # 1. 将各来源结果转换为 FusionItem
        all_items: List[FusionItem] = []
        
        for source, items in source_results.items():
            if not items:
                continue
            
            # 处理不同类型的结果
            if source == 'kg' and isinstance(items, list):
                # 知识图谱三元组
                items_list = items
            elif source == 'vector' and isinstance(items, list):
                # 向量文档
                items_list = items
            elif source == 'wiki' and items:
                # Wikipedia 单条结果
                items_list = [items]
            else:
                items_list = items if isinstance(items, list) else [items]
            
            # 获取得分列表
            scores = source_scores.get(source, []) if source_scores else []
            
            # 转换为 FusionItem
            for idx, item in enumerate(items_list):
                fusion_item = self._create_fusion_item(
                    item, source, idx, scores[idx] if idx < len(scores) else 0.5
                )
                all_items.append(fusion_item)
        
        # 2. 去重处理
        if self.enable_deduplication:
            all_items, dup_count = self._deduplicate(all_items)
            result.duplicates_removed = dup_count
        
        # 3. RRF 融合
        fused_items = self._apply_rrf(all_items)
        
        # 4. 综合得分计算
        for item in fused_items:
            item.combined_score = item.rrf_score
        
        # 5. 按综合得分排序
        fused_items.sort(key=lambda x: x.combined_score, reverse=True)
        
        result.items = fused_items
        result.total_output = len(fused_items)
        
        # 6. 统计信息
        result.source_contributions = self._count_contributions(fused_items)
        result.score_distribution = self._calculate_score_distribution(fused_items)
        result.fusion_time = time.time() - start_time
        
        return result
    
    def _create_fusion_item(self, item: Any, source: str, 
                           rank: int, score: float) -> FusionItem:
        """
        根据不同来源创建 FusionItem
        
        Args:
            item: 原始结果项
            source: 来源标识
            rank: 在该来源中的排名
            score: 相关性得分
        
        Returns:
            FusionItem: 融合项
        """
        # 根据来源解析内容
        if source == 'kg' and isinstance(item, tuple):
            # 三元组 (subject, predicate, object)
            subject, predicate, obj = item
            content = f"{subject} {predicate} {obj}"
            summary = f"({subject}, {predicate}, {obj})"
            entities = [subject, obj]
            relations = [predicate]
            item_id = self._generate_id(content)
            result_type = ResultType.TRIPLE
        elif source == 'vector' and isinstance(item, str):
            # 文档片段
            content = item
            summary = item[:100] + "..." if len(item) > 100 else item
            entities = []
            relations = []
            item_id = self._generate_id(content)
            result_type = ResultType.DOCUMENT
        elif source == 'wiki':
            # Wikipedia 结果
            if isinstance(item, dict):
                content = item.get('summary', str(item))
                summary = content[:100] + "..." if len(content) > 100 else content
            else:
                content = str(item)
                summary = content[:100] + "..." if len(content) > 100 else content
            entities = []
            relations = []
            item_id = self._generate_id(content)
            result_type = ResultType.WIKI
        else:
            # 默认处理
            content = str(item)
            summary = content[:100] + "..." if len(content) > 100 else content
            entities = []
            relations = []
            item_id = self._generate_id(content)
            result_type = ResultType.DOCUMENT
        
        return FusionItem(
            item_id=item_id,
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
        """生成内容唯一ID"""
        # 使用内容哈希作为ID
        content_normalized = content.lower().strip()
        return hashlib.md5(content_normalized.encode('utf-8')).hexdigest()[:16]
    
    def _calculate_similarity(self, content1: str, content2: str) -> float:
        """
        计算两段内容的相似度
        
        使用字符级 Jaccard 相似度
        """
        if not content1 or not content2:
            return 0.0
        
        # 转换为字符集合
        set1 = set(content1.lower())
        set2 = set(content2.lower())
        
        # Jaccard 相似度
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _deduplicate(self, items: List[FusionItem]) -> Tuple[List[FusionItem], int]:
        """
        去重处理
        
        基于内容相似度和类型进行去重
        返回去重后的列表和去重数量
        """
        if not items:
            return [], 0
        
        # 按类型分组
        type_groups: Dict[ResultType, List[FusionItem]] = defaultdict(list)
        for item in items:
            type_groups[item.result_type].append(item)
        
        result = []
        total_removed = 0
        
        for result_type, group in type_groups.items():
            # 在同类型结果中查找重复
            seen: List[FusionItem] = []
            
            for item in group:
                is_duplicate = False
                
                for existing in seen:
                    # 检查是否重复
                    if item.item_id == existing.item_id:
                        # 完全相同，合并
                        existing.merge_with(item)
                        is_duplicate = True
                        total_removed += 1
                        break
                    elif self.enable_deduplication:
                        # 检查相似度
                        similarity = self._calculate_similarity(
                            item.content, existing.content
                        )
                        if similarity >= self.dedup_threshold:
                            # 高度相似，合并
                            existing.merge_with(item)
                            is_duplicate = True
                            total_removed += 1
                            break
                
                if not is_duplicate:
                    seen.append(item)
            
            result.extend(seen)
        
        return result, total_removed
    
    def _apply_rrf(self, items: List[FusionItem]) -> List[FusionItem]:
        """
        应用 RRF 算法计算融合得分
        
        公式: RRF(d) = Σ weight_source / (k + rank_source(d))
        """
        # 按 item_id 分组
        item_groups: Dict[str, List[FusionItem]] = defaultdict(list)
        for item in items:
            item_groups[item.item_id].append(item)
        
        # 计算每个唯一项的 RRF 得分
        fused_map: Dict[str, FusionItem] = {}
        
        for item_id, group in item_groups.items():
            rrf_score = 0.0
            
            # 累加各来源的 RRF 贡献
            for item in group:
                weight = self.source_weights.get(item.source, 1.0)
                for source, rank in item.original_ranks.items():
                    source_weight = self.source_weights.get(source, 1.0)
                    # RRF 公式: 1 / (k + rank)
                    rrf_contribution = source_weight / (self.k + rank)
                    rrf_score += rrf_contribution * weight
            
            # 取第一个作为代表，并赋值 RRF 得分
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
        
        # 定义分数区间
        ranges = [
            (0.8, 1.0, "0.8-1.0"),
            (0.6, 0.8, "0.6-0.8"),
            (0.4, 0.6, "0.4-0.6"),
            (0.2, 0.4, "0.2-0.4"),
            (0.0, 0.2, "0.0-0.2")
        ]
        
        distribution = {r[2]: 0 for r in ranges}
        
        for item in items:
            score = item.rrf_score
            for min_val, max_val, label in ranges:
                if min_val <= score < max_val:
                    distribution[label] += 1
                    break
        
        return distribution
    
    def set_source_weight(self, source: str, weight: float):
        """
        设置来源权重
        
        Args:
            source: 来源标识 (kg/vector/wiki)
            weight: 权重值 (> 0)
        """
        if weight > 0:
            self.source_weights[source] = weight


class MultiRoundRetrieval:
    """
    多轮检索策略控制器
    
    基于 Self-RAG 思想，实现：
    1. 首轮粗召回：多源并行检索
    2. RRF 融合：整合多源结果
    3. 二轮精排：Cohere 语义重排序
    """
    
    def __init__(self, k: float = 60.0, enable_dedup: bool = True):
        """
        初始化多轮检索控制器
        
        Args:
            k: RRF 平滑因子
            enable_dedup: 是否启用去重
        """
        self.fusion_engine = RRFusion(k=k, enable_deduplication=enable_dedup)
        self.reranker = None  # 延迟加载
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
        """
        第一轮：粗召回与融合
        
        Args:
            query: 查询
            kg_results: 知识图谱三元组
            vector_results: 向量检索文档
            wiki_result: Wikipedia 结果
            vector_scores: 向量检索得分
        
        Returns:
            FusionResult: 融合后的候选集
        """
        # 准备输入
        source_results = {
            'kg': kg_results,
            'vector': vector_results,
            'wiki': wiki_result if wiki_result else ""
        }
        
        # 准备得分
        source_scores = {}
        if vector_scores:
            source_scores['vector'] = vector_scores
        # 可根据需要添加 kg 和 wiki 的得分
        
        # 执行 RRF 融合
        return self.fusion_engine.fuse(query, source_results, source_scores)
    
    def execute_round2_rerank(self, query: str, 
                              candidates: List[FusionItem],
                              top_k: int = 10) -> List[FusionItem]:
        """
        第二轮：语义重排序
        
        Args:
            query: 查询
            candidates: 第一轮候选结果
            top_k: 返回前 k 条
        
        Returns:
            List[FusionItem]: 重排序后的结果
        """
        if not self._enable_rerank or not self.reranker or not candidates:
            # 不启用重排序，直接返回
            return candidates[:top_k]
        
        try:
            # 调用 Cohere 重排序
            reranked = self.reranker.rerank(query, candidates, top_k=top_k)
            return reranked
        except Exception as e:
            print(f"[多轮检索] 重排序失败: {e}")
            # 回退到原始排序
            return candidates[:top_k]
    
    def execute_full_pipeline(self, query: str,
                              kg_results: List[Tuple],
                              vector_results: List[str],
                              wiki_result: Optional[str] = None,
                              vector_scores: Optional[List[float]] = None,
                              final_top_k: int = 10) -> Dict[str, Any]:
        """
        执行完整的多轮检索流程
        
        Args:
            query: 查询
            kg_results: 知识图谱三元组
            vector_results: 向量检索文档
            wiki_result: Wikipedia 结果
            vector_scores: 向量检索得分
            final_top_k: 最终返回数量
        
        Returns:
            Dict: 包含融合结果和最终结果
        """
        # 第一轮：粗召回与融合
        fusion_result = self.execute_round1_coarse(
            query=query,
            kg_results=kg_results,
            vector_results=vector_results,
            wiki_result=wiki_result,
            vector_scores=vector_scores
        )
        
        # 获取候选集
        candidates = fusion_result.items
        
        # 第二轮：语义重排序
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


# ==================== 便捷函数 ====================

def fuse_results(query: str, source_results: Dict[str, List],
                source_scores: Optional[Dict[str, List[float]]] = None,
                k: float = 60.0) -> FusionResult:
    """
    便捷函数：执行 RRF 融合
    
    Args:
        query: 查询
        source_results: 各来源结果
        source_scores: 各来源得分
        k: RRF 平滑因子
    
    Returns:
        FusionResult: 融合结果
    """
    fusion = RRFusion(k=k)
    return fusion.fuse(query, source_results, source_scores)


def deduplicate_items(items: List[FusionItem], 
                     threshold: float = 0.85) -> List[FusionItem]:
    """
    便捷函数：对结果列表去重
    
    Args:
        items: 待去重列表
        threshold: 相似度阈值
    
    Returns:
        List[FusionItem]: 去重后的列表
    """
    fusion = RRFusion(k=60.0, enable_deduplication=True, 
                     dedup_threshold=threshold)
    deduped, _ = fusion._deduplicate(items)
    return deduped
