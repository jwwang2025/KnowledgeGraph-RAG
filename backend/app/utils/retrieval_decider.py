"""
检索决策器模块 (Retrieval Decider)
基于 Adaptive-RAG 思想，根据检索计划执行多源自适应检索
"""

import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum

from .query_router import QueryRouter, RetrievalPlan, QuestionType
from .vector_searcher import VectorSearcher
from .graph_utils import search_node_item, convert_graph_to_triples
from .ner import Ner
from .query_wiki import WikiSearcher
from .image_searcher import ImageSearcher


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
            "total_time": f"{self.total_time:.2f}s"
        }


class RetrievalDecider:
    """
    检索决策器 - 自适应执行多源检索
    
    核心功能：
    1. 根据检索计划决定执行顺序
    2. 并行/串行执行多源检索
    3. 实时监控检索状态
    4. 聚合检索结果
    """
    
    def __init__(self, project_name: str = "project_v1", 
                 vector_db_path: str = "./data/vector_db",
                 max_workers: int = 3):
        """
        初始化检索决策器
        
        Args:
            project_name: 项目名称
            vector_db_path: 向量数据库路径
            max_workers: 最大并行检索数 (暂未使用，预留)
        """
        self.project_name = project_name
        self.vector_db_path = vector_db_path
        
        # 延迟初始化的组件
        self._vector_searcher = None
        self._ner = None
        self._wiki_searcher = None
        self._image_searcher = None
        
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
            results={}
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
        
        # 计算总耗时
        result.total_time = time.time() - start_time
        
        return result
    
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
        """聚合单条检索结果到总结果中"""
        if retrieval.status != RetrievalStatus.COMPLETED:
            return
        
        result.total_sources_used += 1
        
        if source == "kg":
            result.triples = retrieval.data or []
            result.has_structured_knowledge = len(result.triples) > 0
            
        elif source == "vector":
            result.documents = retrieval.data or []
            result.has_unstructured_knowledge = len(result.documents) > 0
            
        elif source == "wiki":
            if retrieval.data:
                result.wiki_summary = retrieval.data.get("summary")
                result.has_unstructured_knowledge = True
                
        elif source == "image":
            result.image_url = retrieval.data
    
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