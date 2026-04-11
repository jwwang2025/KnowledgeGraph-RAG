"""
检索决策器模块 (Retrieval Decider)
基于 Adaptive-RAG 思想，根据检索计划执行多源自适应检索
支持引用溯源机制
支持 Self-RAG 多轮检索策略（RRF 融合 + Cohere 重排序）
"""

import json
import time
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Union
from enum import Enum

from .query_router import QueryRouter, RetrievalPlan, QuestionType
from .citation import Citation, CitationContext, CitationGenerator, CitationSource
from .fusion import RRFusion, FusionResult, FusionItem, ResultType, MultiRoundRetrieval
from .reranker import CohereReranker, RerankReport, SelfRAGRefiner
from app.search import VectorSearcher, WikiSearcher, ImageSearcher
from app.kg import search_node_item, convert_graph_to_triples
from app.nlp import Ner


class RetrievalStatus(Enum):
    """检索状态枚举"""
    PENDING = "pending"           # 待检索
    IN_PROGRESS = "in_progress"  # 检索中
    COMPLETED = "completed"       # 已完成
    FAILED = "failed"             # 失败
    SKIPPED = "skipped"           # 跳过


@dataclass
class RetrievalResult:
    """单条检索结果"""
    source: str                          # 知识源标识
    status: RetrievalStatus              # 检索状态
    data: Any                            # 检索到的数据
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据
    error: Optional[str] = None          # 错误信息
    elapsed_time: float = 0.0           # 检索耗时(秒)


@dataclass  
class MultiSourceRetrievalResult:
    """多源检索结果聚合"""
    query: str                           # 原始查询
    plan: RetrievalPlan                  # 检索计划
    results: Dict[str, RetrievalResult]  # 各知识源结果
    total_time: float = 0.0             # 总耗时
    
    # 聚合后的数据
    triples: List[tuple] = field(default_factory=list)       # 知识图谱三元组
    documents: List[str] = field(default_factory=list)       # 文档片段
    wiki_summary: Optional[str] = None                       # Wikipedia摘要
    image_url: Optional[str] = None                          # 图片URL
    
    # 质量指标
    total_sources_used: int = 0                               # 使用的知识源数量
    has_structured_knowledge: bool = False                   # 是否有结构化知识
    has_unstructured_knowledge: bool = False                # 是否有非结构化知识
    
    # 引用溯源 (新增)
    citation_context: CitationContext = field(default_factory=CitationContext)  # 引用上下文
    relevance_scores: Dict[str, List[float]] = field(default_factory=dict)      # 各源的相关性得分
    
    # ========== 多轮检索相关字段 (Self-RAG) ==========
    # 第一轮：RRF 融合结果
    fusion_result: Optional[FusionResult] = None              # RRF 融合结果
    
    # 第二轮：Cohere 重排序结果
    reranked_items: List[FusionItem] = field(default_factory=list)  # 重排序后的结果
    rerank_report: Optional[RerankReport] = None              # 重排序报告
    
    # 多轮检索配置
    enable_multi_round: bool = False                          # 是否启用多轮检索
    multi_round_config: Dict[str, Any] = field(default_factory=dict)  # 配置信息
    
    def get_combined_context(self) -> str:
        """生成合并后的上下文字符串"""
        parts = []
        
        # 添加三元组
        if self.triples:
            triples_str = "；".join([f"({t[0]} {t[1]} {t[2]})" for t in self.triples])
            parts.append(f"知识图谱信息：{triples_str}")
        
        # 添加文档
        if self.documents:
            docs_str = "；".join(self.documents[:3])
            parts.append(f"相关文档：{docs_str}")
        
        # 添加Wiki
        if self.wiki_summary:
            parts.append(f"Wikipedia：{self.wiki_summary[:500]}")
        
        return "；".join(parts)
    
    def get_retrieval_summary(self) -> Dict[str, Any]:
        """获取检索摘要"""
        return {
            "query": self.query,
            "question_type": self.plan.question_type.value,
            "sources_used": self.total_sources_used,
            "triples_count": len(self.triples),
            "docs_count": len(self.documents),
            "has_wiki": self.wiki_summary is not None,
            "has_image": self.image_url is not None,
            "has_structured": self.has_structured_knowledge,
            "has_unstructured": self.has_unstructured_knowledge,
            "total_time": f"{self.total_time:.2f}s",
            "citations_count": len(self.citation_context.merge_all().citations)
        }


class RetrievalDecider:
    """
    检索决策器 - 自适应执行多源检索
    
    核心功能：
    1. 根据检索计划决定执行顺序
    2. 并行/串行执行多源检索
    3. 实时监控检索状态
    4. 聚合检索结果
    5. 支持 Self-RAG 多轮检索策略:
       - 第一轮：RRF 融合多源结果
       - 第二轮：Cohere 语义重排序
    """
    
    def __init__(self, project_name: str = "project_v1", 
                 vector_db_path: str = "./data/vector_db",
                 max_workers: int = 3,
                 enable_multi_round: bool = True,
                 cohere_api_key: Optional[str] = None,
                 cohere_model: str = "rerank-multilingual-v3.0"):
        """
        初始化检索决策器
        
        Args:
            project_name: 项目名称
            vector_db_path: 向量数据库路径
            max_workers: 最大并行检索数 (暂未使用，预留)
            enable_multi_round: 是否启用多轮检索 (RRF + Cohere)
            cohere_api_key: Cohere API 密钥
            cohere_model: Cohere 重排序模型
        """
        self.project_name = project_name
        self.vector_db_path = vector_db_path
        
        # 多轮检索配置 (Self-RAG)
        self.enable_multi_round = enable_multi_round
        self.multi_round_config = {
            "rrf_k": 60.0,              # RRF 平滑因子
            "dedup_threshold": 0.85,       # 去重相似度阈值
            "cohere_model": cohere_model,
            "cohere_api_key": cohere_api_key,
            "final_top_k": 10,           # 最终返回数量
            "fusion_candidates": 50       # 融合候选数量
        }
        
        # 延迟初始化的组件
        self._vector_searcher = None
        self._ner = None
        self._wiki_searcher = None
        self._image_searcher = None
        
        # 多轮检索组件 (延迟初始化)
        self._fusion_engine = None
        self._reranker = None
        self._refiner = None
        
    # ========== 延迟加载组件 ==========
    
    @property
    def vector_searcher(self) -> VectorSearcher:
        """延迟加载向量搜索器"""
        if self._vector_searcher is None:
            self._vector_searcher = VectorSearcher(
                collection_name=f"{self.project_name}_docs",
                persist_dir=self.vector_db_path
            )
        return self._vector_searcher
    
    @property
    def ner(self) -> Ner:
        """延迟加载NER组件"""
        if self._ner is None:
            self._ner = Ner()
        return self._ner
    
    @property
    def wiki_searcher(self) -> WikiSearcher:
        """延迟加载Wikipedia搜索器"""
        if self._wiki_searcher is None:
            self._wiki_searcher = WikiSearcher()
        return self._wiki_searcher
    
    @property
    def image_searcher(self) -> ImageSearcher:
        """延迟加载图像搜索器"""
        if self._image_searcher is None:
            self._image_searcher = ImageSearcher()
        return self._image_searcher
    
    # ========== 多轮检索组件 (延迟初始化) ==========
    
    @property
    def fusion_engine(self) -> RRFusion:
        """延迟加载 RRF 融合引擎"""
        if self._fusion_engine is None:
            self._fusion_engine = RRFusion(
                k=self.multi_round_config.get("rrf_k", 60.0),
                enable_deduplication=True,
                dedup_threshold=self.multi_round_config.get("dedup_threshold", 0.85)
            )
        return self._fusion_engine
    
    @property
    def reranker(self) -> Optional[CohereReranker]:
        """延迟加载 Cohere 重排序器"""
        if self._reranker is None and self.enable_multi_round:
            api_key = self.multi_round_config.get("cohere_api_key")
            if api_key is None:
                import os
                api_key = os.environ.get("COHERE_API_KEY")
            
            self._reranker = CohereReranker(
                api_key=api_key,
                model=self.multi_round_config.get("cohere_model", "rerank-multilingual-v3.0"),
                enable_local_fallback=True
            )
        return self._reranker
    
    @property
    def refiner(self) -> SelfRAGRefiner:
        """延迟加载 Self-RAG 精炼器"""
        if self._refiner is None:
            self._refiner = SelfRAGRefiner(relevance_threshold=0.3)
        return self._refiner
    
    # ========== 核心检索方法 ==========
    
    def retrieve(self, query: str, plan: RetrievalPlan) -> MultiSourceRetrievalResult:
        """
        执行多源自适应检索
        
        Args:
            query: 用户查询
            plan: 检索计划
            
        Returns:
            MultiSourceRetrievalResult: 多源检索结果
        """
        start_time = time.time()
        
        # 初始化结果对象
        result = MultiSourceRetrievalResult(
            query=query,
            plan=plan,
            results={},
            enable_multi_round=self.enable_multi_round,
            multi_round_config=self.multi_round_config
        )
        
        # 如果不需要检索，直接返回
        if not plan.need_retrieval:
            return result
        
        # 按优先级执行各知识源检索
        for source in plan.priority_sources:
            retrieval_result = self._retrieve_from_source(query, source, plan)
            result.results[source] = retrieval_result
            
            # 更新聚合数据
            if retrieval_result.status == RetrievalStatus.COMPLETED:
                self._aggregate_result(result, source, retrieval_result)
        
        # ========== 多轮检索后处理 (Self-RAG) ==========
        if self.enable_multi_round:
            result = self._apply_multi_round_processing(query, result)
        
        # 计算总耗时
        result.total_time = time.time() - start_time
        
        return result
    
    def _apply_multi_round_processing(self, query: str, 
                                     result: MultiSourceRetrievalResult
                                     ) -> MultiSourceRetrievalResult:
        """
        应用多轮检索后处理流程
        
        第一轮：RRF 融合多源结果
        第二轮：Cohere 语义重排序
        
        Args:
            query: 查询
            result: 多源检索结果
        
        Returns:
            更新后的结果
        """
        round_start = time.time()
        
        # 准备各来源数据
        kg_results = result.triples
        vector_results = result.documents
        wiki_result = {"summary": result.wiki_summary} if result.wiki_summary else None
        vector_scores = result.relevance_scores.get("vector", [])
        
        # 第一轮：RRF 融合
        fusion_result = self.fusion_engine.fuse(
            query=query,
            source_results={
                'kg': kg_results,
                'vector': vector_results,
                'wiki': wiki_result if wiki_result else ""
            },
            source_scores={'vector': vector_scores} if vector_scores else None
        )
        
        result.fusion_result = fusion_result
        
        print(f"[多轮检索] 第一轮 RRF 融合: 输入 {fusion_result.total_input} 条, "
              f"输出 {fusion_result.total_output} 条, "
              f"去重 {fusion_result.duplicates_removed} 条")
        
        # 获取融合后的候选集
        candidates = fusion_result.items
        
        # 如果候选数量为0，直接返回
        if not candidates:
            return result
        
        # 第二轮：语义重排序
        if self.reranker:
            top_k = self.multi_round_config.get("final_top_k", 10)
            
            try:
                reranked_items, rerank_report = self.reranker.rerank_with_report(
                    query=query,
                    candidates=candidates,
                    top_k=top_k
                )
                
                result.reranked_items = reranked_items
                result.rerank_report = rerank_report
                
                print(f"[多轮检索] 第二轮 Cohere 重排序: 输入 {rerank_report.input_count} 条, "
                      f"输出 {rerank_report.output_count} 条, "
                      f"模型: {rerank_report.model_used.value}, "
                      f"耗时: {rerank_report.rerank_time:.3f}s")
                
                # Self-RAG 精炼
                refined_items = self.refiner.refine(
                    query=query,
                    items=reranked_items,
                    min_count=3,
                    max_count=top_k
                )
                
                # 更新结果
                result.reranked_items = refined_items
                
                # 提取精炼后的内容用于后续处理
                result.triples, result.documents, result.wiki_summary = \
                    self._extract_refined_content(refined_items, kg_results, vector_results)
                
            except Exception as e:
                print(f"[多轮检索] 重排序失败: {e}")
                # 使用 RRF 融合结果作为后备
                result.reranked_items = candidates[:self.multi_round_config.get("final_top_k", 10)]
        
        print(f"[多轮检索] 总耗时: {time.time() - round_start:.3f}s")
        
        return result
    
    def _extract_refined_content(self, items: List[FusionItem],
                                original_triples: List[tuple],
                                original_docs: List[str]
                                ) -> tuple[List[tuple], List[str], Optional[str]]:
        """
        从精炼后的结果中提取内容
        
        Args:
            items: 精炼后的结果列表
            original_triples: 原始三元组列表
            original_docs: 原始文档列表
        
        Returns:
            (triples, documents, wiki_summary)
        """
        triples = []
        documents = []
        wiki_summary = None
        
        for item in items:
            if item.result_type == ResultType.TRIPLE:
                # 从原始三元组中找到匹配的内容
                for triple in original_triples:
                    triple_str = f"{triple[0]} {triple[1]} {triple[2]}"
                    if triple_str in item.content or item.content in triple_str:
                        triples.append(triple)
                        break
            elif item.result_type == ResultType.DOCUMENT:
                documents.append(item.content)
            elif item.result_type == ResultType.WIKI:
                wiki_summary = item.content
        
        # 限制数量
        triples = triples[:10]
        documents = documents[:5]
        
        return triples, documents, wiki_summary
    
    def _retrieve_from_source(self, query: str, source: str, 
                             plan: RetrievalPlan) -> RetrievalResult:
        """
        从单个知识源检索
        
        Args:
            query: 查询
            source: 知识源标识
            plan: 检索计划
            
        Returns:
            RetrievalResult: 检索结果
        """
        start_time = time.time()
        
        try:
            if source == "kg":
                return self._retrieve_from_kg(query, plan)
            elif source == "vector":
                return self._retrieve_from_vector(query, plan)
            elif source == "wiki":
                return self._retrieve_from_wiki(query, plan)
            elif source == "image":
                return self._retrieve_from_image(query, plan)
            else:
                return RetrievalResult(
                    source=source,
                    status=RetrievalStatus.SKIPPED,
                    data=None,
                    error=f"Unknown source: {source}"
                )
        except Exception as e:
            return RetrievalResult(
                source=source,
                status=RetrievalStatus.FAILED,
                data=None,
                error=str(e),
                elapsed_time=time.time() - start_time
            )
    
    def _retrieve_from_kg(self, query: str, plan: RetrievalPlan) -> RetrievalResult:
        """
        从知识图谱检索三元组
        
        核心逻辑：
        1. 使用NER识别查询中的实体
        2. 在知识图谱中搜索实体相关的三元组
        3. 返回聚合的三元组列表
        """
        try:
            # 1. 识别实体
            entities = self.ner.get_entities(
                query, 
                etypes=["物体类", "人物类", "地点类", "组织机构类", "事件类", "世界地区类", "术语类"]
            )
            
            if not entities:
                return RetrievalResult(
                    source="kg",
                    status=RetrievalStatus.COMPLETED,
                    data=[],
                    metadata={"entities_found": 0}
                )
            
            # 2. 搜索图谱
            lite_graph = {'nodes': [], 'links': [], 'sents': []}
            
            for entity in entities[:5]:  # 限制实体数量
                lite_graph = search_node_item(entity, lite_graph if lite_graph['nodes'] else None)
            
            # 3. 转换为三元组
            triples = []
            for entity in entities[:3]:  # 主要针对前几个实体
                triples += convert_graph_to_triples(lite_graph, entity)
            
            # 4. 限制数量
            triples = triples[:plan.max_triples]
            
            return RetrievalResult(
                source="kg",
                status=RetrievalStatus.COMPLETED,
                data=triples,
                metadata={
                    "entities_found": len(entities),
                    "entities_used": min(len(entities), 5),
                    "triples_count": len(triples)
                },
                elapsed_time=0.0  # 忽略不计
            )
            
        except Exception as e:
            return RetrievalResult(
                source="kg",
                status=RetrievalStatus.FAILED,
                data=None,
                error=f"知识图谱检索失败: {str(e)}"
            )
    
    def _retrieve_from_vector(self, query: str, plan: RetrievalPlan) -> RetrievalResult:
        """
        从向量数据库检索文档
        
        核心逻辑：
        1. 使用语义检索找到相关文档
        2. 根据相似度过滤低质量结果
        3. 返回文档列表
        """
        try:
            # 语义检索
            search_results = self.vector_searcher.search(
                query, 
                top_k=plan.max_docs,
                threshold=1.5  # ChromaDB的距离阈值，越小越相似
            )
            
            documents = []
            if search_results and search_results.get('documents'):
                docs = search_results['documents']
                if docs and len(docs) > 0:
                    # ChromaDB 返回格式: [[doc1, doc2, ...]]
                    documents = [d for d in docs[0] if d]
            
            # 限制数量
            documents = documents[:plan.max_docs]
            
            return RetrievalResult(
                source="vector",
                status=RetrievalStatus.COMPLETED,
                data=documents,
                metadata={
                    "docs_found": len(documents),
                    "similarity_scores": search_results.get('distances', [[]])[0][:len(documents)]
                }
            )
            
        except Exception as e:
            return RetrievalResult(
                source="vector",
                status=RetrievalStatus.FAILED,
                data=None,
                error=f"向量检索失败: {str(e)}"
            )
    
    def _retrieve_from_wiki(self, query: str, plan: RetrievalPlan) -> RetrievalResult:
        """
        从Wikipedia检索补充知识
        
        核心逻辑：
        1. 先尝试用NER识别的实体搜索
        2. 如果没有实体，用原始查询搜索
        3. 返回摘要信息
        """
        try:
            # 尝试实体搜索
            entities = self.ner.get_entities(
                query,
                etypes=["物体类", "人物类", "地点类", "组织机构类"]
            )
            
            wiki = None
            search_targets = entities + [query] if entities else [query]
            
            for target in search_targets:
                wiki = self.wiki_searcher.search(target)
                if wiki is not None:
                    break
            
            if wiki is None:
                return RetrievalResult(
                    source="wiki",
                    status=RetrievalStatus.COMPLETED,
                    data=None,
                    metadata={"found": False}
                )
            
            # 简繁转换并截取摘要
            from opencc import OpenCC
            cc = OpenCC('t2s')
            summary = cc.convert(wiki.summary)[:500]
            
            return RetrievalResult(
                source="wiki",
                status=RetrievalStatus.COMPLETED,
                data={
                    "title": cc.convert(wiki.title),
                    "summary": summary
                },
                metadata={
                    "found": True,
                    "title": wiki.title
                }
            )
            
        except Exception as e:
            return RetrievalResult(
                source="wiki",
                status=RetrievalStatus.FAILED,
                data=None,
                error=f"Wikipedia检索失败: {str(e)}"
            )
    
    def _retrieve_from_image(self, query: str, plan: RetrievalPlan) -> RetrievalResult:
        """
        检索相关图片
        
        目前图像搜索不区分问题类型，总是执行
        """
        try:
            image_url = self.image_searcher.search(query)
            
            return RetrievalResult(
                source="image",
                status=RetrievalStatus.COMPLETED,
                data=image_url,
                metadata={"found": image_url is not None}
            )
            
        except Exception as e:
            return RetrievalResult(
                source="image",
                status=RetrievalStatus.FAILED,
                data=None,
                error=f"图像检索失败: {str(e)}"
            )
    
    def _aggregate_result(self, result: MultiSourceRetrievalResult, 
                          source: str, retrieval: RetrievalResult):
        """聚合单条检索结果到总结果中，并生成引用"""
        if retrieval.status != RetrievalStatus.COMPLETED:
            return
        
        result.total_sources_used += 1
        
        if source == "kg":
            result.triples = retrieval.data or []
            result.has_structured_knowledge = len(result.triples) > 0
            # 生成知识图谱引用
            if result.triples:
                relevance = retrieval.metadata.get("relevance_scores", [0.8] * len(result.triples))
                citations = CitationGenerator.generate_triple_citations(
                    result.triples, result.query, relevance
                )
                result.citation_context.triples_citations.extend(citations)
                result.citation_context.raw_triples = result.triples
                result.relevance_scores["kg"] = relevance
            
        elif source == "vector":
            result.documents = retrieval.data or []
            result.has_unstructured_knowledge = len(result.documents) > 0
            # 生成文档引用
            if result.documents:
                distances = retrieval.metadata.get("similarity_scores", [])
                relevance_scores = [max(0, 1 - d) for d in distances] if distances else [0.7] * len(result.documents)
                citations = CitationGenerator.generate_document_citations(
                    result.documents, query=result.query, relevance_scores=relevance_scores
                )
                result.citation_context.document_citations.extend(citations)
                result.citation_context.raw_documents = result.documents
                result.relevance_scores["vector"] = relevance_scores
            
        elif source == "wiki":
            if retrieval.data:
                result.wiki_summary = retrieval.data.get("summary")
                result.has_unstructured_knowledge = True
                # 生成 Wikipedia 引用
                wiki_citation = CitationGenerator.generate_wiki_citation(
                    retrieval.data, result.query
                )
                if wiki_citation:
                    result.citation_context.wiki_citations.append(wiki_citation)
                    result.citation_context.raw_wiki = retrieval.data
                    
        elif source == "image":
            result.image_url = retrieval.data
            # 生成图像引用
            if result.image_url:
                img_citation = CitationGenerator.generate_image_citation(
                    result.image_url, result.query
                )
                if img_citation:
                    result.citation_context.image_citations.append(img_citation)
    
    def get_source_status(self, result: MultiSourceRetrievalResult) -> Dict[str, str]:
        """获取各知识源的状态摘要"""
        status = {}
        for source, retrieval in result.results.items():
            status[source] = {
                RetrievalStatus.COMPLETED: "✅ 成功",
                RetrievalStatus.FAILED: "❌ 失败",
                RetrievalStatus.SKIPPED: "⏭️ 跳过",
                RetrievalStatus.PENDING: "⏳ 待执行",
                RetrievalStatus.IN_PROGRESS: "🔄 进行中"
            }.get(retrieval.status, "❓ 未知")
        return status
    
    # ========== 多轮检索控制方法 ==========
    
    def enable_multi_round_retrieval(self, enabled: bool = True):
        """启用/禁用多轮检索"""
        self.enable_multi_round = enabled
        print(f"[RetrievalDecider] 多轮检索: {'启用' if enabled else '禁用'}")
    
    def set_multi_round_config(self, config: Dict[str, Any]):
        """设置多轮检索配置"""
        valid_keys = ["rrf_k", "dedup_threshold", "cohere_model", "cohere_api_key", 
                      "final_top_k", "fusion_candidates"]
        for key, value in config.items():
            if key in valid_keys:
                self.multi_round_config[key] = value
        print(f"[RetrievalDecider] 多轮检索配置已更新: {self.multi_round_config}")
    
    def get_multi_round_stats(self) -> Dict[str, Any]:
        """获取多轮检索统计信息"""
        return {
            "enabled": self.enable_multi_round,
            "config": self.multi_round_config.copy(),
            "components_loaded": {
                "fusion_engine": self._fusion_engine is not None,
                "reranker": self._reranker is not None,
                "refiner": self._refiner is not None
            }
        }