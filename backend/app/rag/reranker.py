"""
Cohere 语义重排序模块
基于语义感知的多轮检索精细化重排序

核心功能：
1. 调用 Cohere Rerank API 进行语义级别的精确重排序
2. 支持本地轻量级重排序作为备选方案
3. 集成到多轮检索流程的第二轮
4. 与 Self-RAG 评估机制协同工作
"""

import os
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from enum import Enum

from .fusion import FusionItem, FusionResult, ResultType


class RerankModel(Enum):
    """重排序模型枚举"""
    COHERE = "cohere"                  # Cohere API
    LOCAL_SIMILARITY = "local"         # 本地相似度计算
    HYBRID = "hybrid"                   # 混合模式


@dataclass
class RerankResult:
    """
    单条重排序结果
    
    包含重排序后的位置、语义相关度得分等
    """
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
    """
    重排序报告
    
    包含完整的重排序过程和结果
    """
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
    """
    Cohere 语义重排序器
    
    使用 Cohere Rerank API 进行语义级别的精确重排序
    
    特点：
    1. 语义理解能力强，能捕捉深层语义关系
    2. 支持中英文语义匹配
    3. 响应速度快，适合实时场景
    4. 与 RRF 融合结果配合使用效果最佳
    """
    
    def __init__(self, api_key: Optional[str] = None,
                 model: str = "rerank-multilingual-v3.0",
                 enable_local_fallback: bool = True):
        """
        初始化 Cohere 重排序器
        
        Args:
            api_key: Cohere API 密钥 (默认从环境变量获取)
            model: 使用的重排序模型
            enable_local_fallback: 当 API 不可用时是否使用本地备选
        """
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
        """
        对候选结果进行语义重排序
        
        Args:
            query: 查询文本
            candidates: 候选结果列表 (FusionItem)
            top_k: 返回前 k 条结果
            return_documents: 是否返回文档内容
        
        Returns:
            List[FusionItem]: 重排序后的结果列表
        """
        if not candidates:
            return []
        
        # 限制候选数量以节省 API 调用成本
        max_candidates = min(len(candidates), 100)
        candidates = candidates[:max_candidates]
        
        # 构建重排序报告
        report = RerankReport(
            query=query,
            model_used=RerankModel.LOCAL_SIMILARITY,
            input_count=len(candidates)
        )
        
        start_time = time.time()
        
        # 尝试使用 Cohere API
        if self.client and self.api_key:
            try:
                results = self._rerank_with_cohere(query, candidates, top_k)
                report.model_used = RerankModel.COHERE
            except Exception as e:
                print(f"[CohereReranker] API 调用失败: {e}")
                results = self._rerank_locally(query, candidates, top_k)
        else:
            # 使用本地备选方案
            results = self._rerank_locally(query, candidates, top_k)
        
        report.rerank_time = time.time() - start_time
        report.output_count = len(results)
        
        # 更新结果位置
        for idx, item in enumerate(results):
            item.rerank_position = idx
        
        return results
    
    def _rerank_with_cohere(self, query: str, candidates: List[FusionItem],
                           top_k: int) -> List[FusionItem]:
        """
        使用 Cohere API 进行重排序
        """
        # 准备文档列表
        documents = [item.content for item in candidates]
        
        # 调用 Cohere Rerank API
        response = self.client.rerank(
            query=query,
            documents=documents,
            model=self.model,
            top_n=top_k,
            return_documents=return_documents
        )
        
        # 构建结果映射
        results = []
        for idx, result in enumerate(response.results):
            original_idx = result.index
            rerank_score = result.relevance_score
            
            # 更新原始项的得分
            item = candidates[original_idx]
            
            # 使用 Cohere 的相关性得分
            item.rrf_score = rerank_score
            item.combined_score = rerank_score
            
            # 创建新的 FusionItem 以避免修改原列表
            from copy import deepcopy
            new_item = deepcopy(item)
            new_item.rrf_score = rerank_score
            new_item.combined_score = rerank_score
            
            results.append(new_item)
        
        # 按得分降序排列
        results.sort(key=lambda x: x.rrf_score, reverse=True)
        
        return results[:top_k]
    
    def _rerank_locally(self, query: str, candidates: List[FusionItem],
                       top_k: int) -> List[FusionItem]:
        """
        使用本地轻量级重排序
        
        基于关键词匹配和语义相似度计算
        作为 API 不可用时的备选方案
        """
        from copy import deepcopy
        
        # 计算每个候选的本地得分
        scored_items = []
        
        for idx, item in enumerate(candidates):
            # 1. 关键词匹配得分
            keyword_score = self._calculate_keyword_match(query, item.content)
            
            # 2. 实体匹配得分
            entity_score = self._calculate_entity_match(query, item.content, item.entities)
            
            # 3. 类型权重 (三元组通常更精确)
            type_weight = 1.0
            if item.result_type == ResultType.TRIPLE:
                type_weight = 1.2  # 三元组权重稍高
            elif item.result_type == ResultType.WIKI:
                type_weight = 0.9  # Wikipedia 权重稍低
            
            # 4. 综合得分
            local_score = (
                keyword_score * 0.4 +
                entity_score * 0.4 +
                type_weight * 0.2
            )
            
            # 结合原始 RRF 得分
            combined_score = (
                item.rrf_score * 0.3 +
                local_score * 0.7
            )
            
            # 创建副本并更新得分
            new_item = deepcopy(item)
            new_item.rrf_score = combined_score
            new_item.combined_score = combined_score
            
            scored_items.append((combined_score, idx, new_item))
        
        # 按得分降序排列
        scored_items.sort(key=lambda x: x[0], reverse=True)
        
        # 返回前 k 条
        return [item for _, _, item in scored_items[:top_k]]
    
    def _calculate_keyword_match(self, query: str, content: str) -> float:
        """
        计算关键词匹配得分
        
        基于查询词在内容中的出现情况
        """
        # 提取查询关键词 (2-4字词)
        query_keywords = []
        for i in range(len(query)):
            for length in [4, 3, 2]:
                if i + length <= len(query):
                    word = query[i:i+length]
                    if word not in ['的', '是', '在', '了', '和', '与', '或', '有']:
                        query_keywords.append(word)
        
        # 去重
        query_keywords = list(set(query_keywords))
        
        if not query_keywords:
            return 0.5
        
        # 统计匹配
        content_lower = content.lower()
        matches = sum(1 for kw in query_keywords if kw in content_lower)
        
        # 归一化
        return min(matches / len(query_keywords), 1.0)
    
    def _calculate_entity_match(self, query: str, content: str,
                               entities: List[str]) -> float:
        """
        计算实体匹配得分
        
        检查内容中是否包含查询中的实体
        """
        query_lower = query.lower()
        content_lower = content.lower()
        
        # 检查实体匹配
        if entities:
            entity_matches = sum(1 for e in entities if e in content_lower)
            entity_score = entity_matches / len(entities)
        else:
            entity_score = 0.5  # 无实体信息时的默认分
        
        # 检查查询中的专有名词（2字以上且包含特定字符的词）
        import re
        proper_nouns = re.findall(r'[\u4e00-\u9fff]{2,}', query)
        proper_noun_matches = sum(1 for pn in proper_nouns if pn in content_lower)
        
        if proper_nouns:
            proper_noun_score = proper_noun_matches / len(proper_nouns)
        else:
            proper_noun_score = 0.5
        
        return (entity_score + proper_noun_score) / 2
    
    def rerank_with_report(self, query: str, candidates: List[FusionItem],
                          top_k: int = 10) -> tuple[List[FusionItem], RerankReport]:
        """
        重排序并返回详细报告
        
        Args:
            query: 查询
            candidates: 候选列表
            top_k: 返回数量
        
        Returns:
            (重排序结果, 报告)
        """
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
        
        # 位置变化统计
        original_scores = [(item.rrf_score, idx) for idx, item in enumerate(candidates)]
        original_scores.sort(key=lambda x: x[0], reverse=True)
        
        # 统计位置变化
        promoted = 0
        demoted = 0
        for idx, item in enumerate(results):
            # 找到原始位置
            original_pos = next((i for i, (_, orig_idx) in enumerate(original_scores) 
                               if orig_idx == candidates.index(item)), -1)
            if original_pos > idx:
                promoted += 1
            elif original_pos < idx:
                demoted += 1
        
        report.position_changes = {"promoted": promoted, "demoted": demoted}
        
        return results, report


class SelfRAGRefiner:
    """
    Self-RAG 结果精炼器
    
    结合 Self-RAG 的反思机制，对重排序后的结果进行进一步优化：
    1. 过滤低相关性结果
    2. 调整内容片段长度
    3. 优化上下文连贯性
    """
    
    def __init__(self, relevance_threshold: float = 0.3):
        """
        初始化精炼器
        
        Args:
            relevance_threshold: 相关性阈值，低于此阈值的结果将被过滤
        """
        self.relevance_threshold = relevance_threshold
    
    def refine(self, query: str, items: List[FusionItem],
              min_count: int = 3, max_count: int = 10
              ) -> List[FusionItem]:
        """
        精炼结果
        
        Args:
            query: 查询
            items: 重排序后的结果
            min_count: 最少保留数量
            max_count: 最多保留数量
        
        Returns:
            List[FusionItem]: 精炼后的结果
        """
        if not items:
            return []
        
        # 1. 过滤低相关性
        filtered = [
            item for item in items
            if item.rrf_score >= self.relevance_threshold
        ]
        
        # 如果过滤后太少，保留阈值以下的结果
        if len(filtered) < min_count:
            filtered = items[:max(min_count, len(filtered))]
        else:
            filtered = filtered[:max_count]
        
        # 2. 确保类型多样性
        refined = self._ensure_diversity(filtered)
        
        # 3. 调整顺序 - 优先返回不同类型的高质量结果
        refined = self._optimize_order(refined)
        
        return refined
    
    def _ensure_diversity(self, items: List[FusionItem]) -> List[FusionItem]:
        """
        确保结果类型多样性
        
        避免单一类型垄断结果列表
        """
        # 按类型分组
        by_type: Dict[ResultType, List[FusionItem]] = {
            ResultType.TRIPLE: [],
            ResultType.DOCUMENT: [],
            ResultType.WIKI: []
        }
        
        for item in items:
            by_type[item.result_type].append(item)
        
        # 每个类型至少保留一个（如果有）
        result = []
        max_per_type = 5  # 每种类型最多5条
        
        for result_type, type_items in by_type.items():
            if type_items:
                result.extend(type_items[:max_per_type])
        
        # 按得分排序
        result.sort(key=lambda x: x.rrf_score, reverse=True)
        
        return result
    
    def _optimize_order(self, items: List[FusionItem]) -> List[FusionItem]:
        """
        优化结果顺序
        
        策略：
        1. 高分三元组优先
        2. 不同类型交叉排列
        3. 保持语义连贯性
        """
        if len(items) <= 2:
            return items
        
        # 简单策略：按类型优先级排序
        type_priority = {
            ResultType.TRIPLE: 0,    # 优先三元组
            ResultType.WIKI: 1,      # 其次 Wikipedia
            ResultType.DOCUMENT: 2   # 最后文档
        }
        
        return sorted(items, key=lambda x: (type_priority.get(x.result_type, 3), -x.rrf_score))


# ==================== 便捷函数 ====================

def create_reranker(api_key: Optional[str] = None,
                   model: str = "rerank-multilingual-v3.0",
                   enable_local: bool = True) -> CohereReranker:
    """
    创建重排序器实例
    
    Args:
        api_key: Cohere API 密钥
        model: 使用的模型
        enable_local: 启用本地备选
    
    Returns:
        CohereReranker: 重排序器实例
    """
    return CohereReranker(
        api_key=api_key,
        model=model,
        enable_local_fallback=enable_local
    )


def quick_rerank(query: str, candidates: List[FusionItem],
                api_key: Optional[str] = None,
                top_k: int = 10) -> List[FusionItem]:
    """
    快速重排序便捷函数
    
    Args:
        query: 查询
        candidates: 候选结果
        api_key: API 密钥
        top_k: 返回数量
    
    Returns:
        List[FusionItem]: 重排序后的结果
    """
    reranker = create_reranker(api_key=api_key)
    return reranker.rerank(query, candidates, top_k)
