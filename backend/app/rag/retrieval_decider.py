"""检索决策器：基于 Adaptive-RAG 思想执行多源自适应检索，支持引用溯源与 Self-RAG 多轮检索。"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum

from .query_router import RetrievalPlan
from .citation import CitationContext, CitationGenerator
from .fusion import RRFusion, FusionResult, FusionItem, ResultType
from .reranker import CohereReranker, RerankReport, SelfRAGRefiner
from app.search import VectorSearcher, WikiSearcher, ImageSearcher
from app.kg import search_node_item, convert_graph_to_triples
from app.nlp import Ner

_KG_ENTITY_TYPES = ["物体类", "人物类", "地点类", "组织机构类", "事件类", "世界地区类", "术语类"]
_WIKI_ENTITY_TYPES = ["物体类", "人物类", "地点类", "组织机构类"]

# 各知识源检索失败时的错误信息前缀（与原实现保持一致）
_SOURCE_ERROR_LABELS = {
    "kg": "知识图谱",
    "vector": "向量",
    "wiki": "Wikipedia",
    "image": "图像",
}

_opencc_converter = None


def _get_opencc():
    global _opencc_converter
    if _opencc_converter is None:
        from opencc import OpenCC
        _opencc_converter = OpenCC('t2s')
    return _opencc_converter


def _build_wiki_wrapper(top_k: int = 3):
    """构建 Wikipedia API Wrapper"""
    from langchain_community.utilities import WikipediaAPIWrapper
    return WikipediaAPIWrapper(top_k_results=top_k)


class RetrievalStatus(Enum):
    """检索状态枚举"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class RetrievalResult:
    """单条检索结果"""
    source: str
    status: RetrievalStatus
    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    elapsed_time: float = 0.0


@dataclass
class MultiSourceRetrievalResult:
    """多源检索结果聚合"""
    query: str
    plan: RetrievalPlan
    results: Dict[str, RetrievalResult]
    total_time: float = 0.0

    triples: List[tuple] = field(default_factory=list)
    documents: List[str] = field(default_factory=list)
    wiki_summary: Optional[str] = None
    image_url: Optional[str] = None

    total_sources_used: int = 0
    has_structured_knowledge: bool = False
    has_unstructured_knowledge: bool = False

    citation_context: CitationContext = field(default_factory=CitationContext)
    relevance_scores: Dict[str, List[float]] = field(default_factory=dict)

    # ========== 多轮检索相关字段 (Self-RAG) ==========
    fusion_result: Optional[FusionResult] = None
    reranked_items: List[FusionItem] = field(default_factory=list)
    rerank_report: Optional[RerankReport] = None

    enable_multi_round: bool = False
    multi_round_config: Dict[str, Any] = field(default_factory=dict)

    def get_combined_context(self) -> str:
        """生成合并后的上下文字符串"""
        parts = []
        if self.triples:
            triples_str = "；".join([f"({t[0]} {t[1]} {t[2]})" for t in self.triples])
            parts.append(f"知识图谱信息：{triples_str}")
        if self.documents:
            parts.append(f"相关文档：{'；'.join(self.documents[:3])}")
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

    根据检索计划执行各知识源检索并聚合结果；
    多轮检索策略：第一轮 RRF 融合，第二轮 Cohere 语义重排序。
    """

    def __init__(self, project_name: str = "project_v1",
                 vector_db_path: str = "./data/vector_db",
                 max_workers: int = 3,
                 enable_multi_round: bool = True,
                 cohere_api_key: Optional[str] = None,
                 cohere_model: str = "rerank-multilingual-v3.0"):
        self.project_name = project_name
        self.vector_db_path = vector_db_path

        self.enable_multi_round = enable_multi_round
        self.multi_round_config = {
            "rrf_k": 60.0,
            "dedup_threshold": 0.85,
            "cohere_model": cohere_model,
            "cohere_api_key": cohere_api_key,
            "final_top_k": 10,
            "fusion_candidates": 50
        }

        self._vector_searcher = None
        self._ner = None
        self._wiki_searcher = None
        self._image_searcher = None

        self._fusion_engine = None
        self._reranker = None
        self._refiner = None

    @property
    def vector_searcher(self) -> VectorSearcher:
        if self._vector_searcher is None:
            self._vector_searcher = VectorSearcher(
                collection_name=f"{self.project_name}_docs",
                persist_dir=self.vector_db_path
            )
        return self._vector_searcher

    @property
    def ner(self) -> Ner:
        if self._ner is None:
            self._ner = Ner()
        return self._ner

    @property
    def wiki_searcher(self) -> WikiSearcher:
        if self._wiki_searcher is None:
            self._wiki_searcher = WikiSearcher()
        return self._wiki_searcher

    @property
    def image_searcher(self) -> ImageSearcher:
        if self._image_searcher is None:
            self._image_searcher = ImageSearcher()
        return self._image_searcher

    @property
    def fusion_engine(self) -> RRFusion:
        if self._fusion_engine is None:
            self._fusion_engine = RRFusion(
                k=self.multi_round_config.get("rrf_k", 60.0),
                enable_deduplication=True,
                dedup_threshold=self.multi_round_config.get("dedup_threshold", 0.85)
            )
        return self._fusion_engine

    @property
    def reranker(self) -> Optional[CohereReranker]:
        if self._reranker is None and self.enable_multi_round:
            import os
            api_key = self.multi_round_config.get("cohere_api_key")
            if api_key is None:
                api_key = os.environ.get("COHERE_API_KEY")

            self._reranker = CohereReranker(
                api_key=api_key,
                model=self.multi_round_config.get("cohere_model", "rerank-multilingual-v3.0"),
                enable_local_fallback=True
            )
        return self._reranker

    @property
    def refiner(self) -> SelfRAGRefiner:
        if self._refiner is None:
            self._refiner = SelfRAGRefiner(relevance_threshold=0.3)
        return self._refiner

    def retrieve(self, query: str, plan: RetrievalPlan) -> MultiSourceRetrievalResult:
        """执行多源自适应检索"""
        start_time = time.time()

        result = MultiSourceRetrievalResult(
            query=query,
            plan=plan,
            results={},
            enable_multi_round=self.enable_multi_round,
            multi_round_config=self.multi_round_config
        )

        if not plan.need_retrieval:
            return result

        for source in plan.priority_sources:
            retrieval_result = self._retrieve_from_source(query, source, plan)
            result.results[source] = retrieval_result

            if retrieval_result.status == RetrievalStatus.COMPLETED:
                self._aggregate_result(result, source, retrieval_result)

        if self.enable_multi_round:
            result = self._apply_multi_round_processing(query, result)

        result.total_time = time.time() - start_time
        return result

    def _apply_multi_round_processing(self, query: str,
                                      result: MultiSourceRetrievalResult
                                      ) -> MultiSourceRetrievalResult:
        """
        多轮检索后处理：第一轮 RRF 融合，第二轮 Cohere 语义重排序
        """
        round_start = time.time()

        kg_results = result.triples
        vector_results = result.documents
        wiki_result = {"summary": result.wiki_summary} if result.wiki_summary else None
        vector_scores = result.relevance_scores.get("vector", [])
        top_k = self.multi_round_config.get("final_top_k", 10)

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

        candidates = fusion_result.items
        if not candidates:
            return result

        if self.reranker:
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

                refined_items = self.refiner.refine(
                    query=query,
                    items=reranked_items,
                    min_count=3,
                    max_count=top_k
                )

                result.reranked_items = refined_items
                result.triples, result.documents, result.wiki_summary = \
                    self._extract_refined_content(refined_items, kg_results, vector_results)

            except Exception as e:
                print(f"[多轮检索] 重排序失败: {e}")
                # 使用 RRF 融合结果作为后备
                result.reranked_items = candidates[:top_k]

        print(f"[多轮检索] 总耗时: {time.time() - round_start:.3f}s")

        return result

    def _extract_refined_content(self, items: List[FusionItem],
                                 original_triples: List[tuple],
                                 original_docs: List[str]
                                 ) -> tuple[List[tuple], List[str], Optional[str]]:
        """从精炼后的结果中提取 (triples, documents, wiki_summary)"""
        triples = []
        documents = []
        wiki_summary = None

        for item in items:
            if item.result_type == ResultType.TRIPLE:
                for triple in original_triples:
                    triple_str = f"{triple[0]} {triple[1]} {triple[2]}"
                    if triple_str in item.content or item.content in triple_str:
                        triples.append(triple)
                        break
            elif item.result_type == ResultType.DOCUMENT:
                documents.append(item.content)
            elif item.result_type == ResultType.WIKI:
                wiki_summary = item.content

        return triples[:10], documents[:5], wiki_summary

    def _retrieve_from_source(self, query: str, source: str,
                              plan: RetrievalPlan) -> RetrievalResult:
        """从单个知识源检索，统一包装异常处理"""
        start_time = time.time()

        handler = getattr(self, f"_retrieve_from_{source}", None)
        if handler is None:
            return RetrievalResult(
                source=source,
                status=RetrievalStatus.SKIPPED,
                data=None,
                error=f"Unknown source: {source}"
            )

        try:
            return handler(query, plan)
        except Exception as e:
            label = _SOURCE_ERROR_LABELS.get(source, source)
            return RetrievalResult(
                source=source,
                status=RetrievalStatus.FAILED,
                data=None,
                error=f"{label}检索失败: {str(e)}",
                elapsed_time=time.time() - start_time
            )

    def _retrieve_from_kg(self, query: str, plan: RetrievalPlan) -> RetrievalResult:
        """从知识图谱检索三元组：NER 识别实体 -> 图谱搜索 -> 转换为三元组"""
        entities = self.ner.get_entities(query, etypes=_KG_ENTITY_TYPES)

        if not entities:
            return RetrievalResult(
                source="kg",
                status=RetrievalStatus.COMPLETED,
                data=[],
                metadata={"entities_found": 0}
            )

        lite_graph = {'nodes': [], 'links': [], 'sents': []}
        for entity in entities[:5]:  # 限制实体数量
            lite_graph = search_node_item(entity, lite_graph if lite_graph['nodes'] else None)

        triples = []
        for entity in entities[:3]:
            triples += convert_graph_to_triples(lite_graph, entity)
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

    def _retrieve_from_vector(self, query: str, plan: RetrievalPlan) -> RetrievalResult:
        """从向量数据库检索文档：语义检索并过滤低质量结果"""
        search_results = self.vector_searcher.search(
            query,
            top_k=plan.max_docs,
            where_document=None  # 暂不使用文档过滤
        )

        documents = []
        if search_results and search_results.get('documents'):
            docs = search_results['documents']
            if docs:
                # ChromaDB 返回格式: [[doc1, doc2, ...]]
                documents = [d for d in docs[0] if d]

        documents = documents[:plan.max_docs]

        distances = search_results.get('distances', [[]])[0][:len(documents)] if documents else []

        return RetrievalResult(
            source="vector",
            status=RetrievalStatus.COMPLETED,
            data=documents,
            metadata={
                "docs_found": len(documents),
                "similarity_scores": distances
            }
        )

    def _retrieve_from_wiki(self, query: str, plan: RetrievalPlan) -> RetrievalResult:
        """从Wikipedia检索补充知识：优先用NER实体搜索，否则用原始查询"""
        entities = self.ner.get_entities(query, etypes=_WIKI_ENTITY_TYPES)

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

        cc = _get_opencc()
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

    def _retrieve_from_image(self, query: str, plan: RetrievalPlan) -> RetrievalResult:
        """检索相关图片（图像搜索不区分问题类型，总是执行）"""
        image_url = self.image_searcher.search(query)

        return RetrievalResult(
            source="image",
            status=RetrievalStatus.COMPLETED,
            data=image_url,
            metadata={"found": image_url is not None}
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
                wiki_citation = CitationGenerator.generate_wiki_citation(
                    retrieval.data, result.query
                )
                if wiki_citation:
                    result.citation_context.wiki_citations.append(wiki_citation)
                    result.citation_context.raw_wiki = retrieval.data

        elif source == "image":
            result.image_url = retrieval.data
            if result.image_url:
                img_citation = CitationGenerator.generate_image_citation(
                    result.image_url, result.query
                )
                if img_citation:
                    result.citation_context.image_citations.append(img_citation)

    _STATUS_LABELS = {
        RetrievalStatus.COMPLETED: "✅ 成功",
        RetrievalStatus.FAILED: "❌ 失败",
        RetrievalStatus.SKIPPED: "⏭️ 跳过",
        RetrievalStatus.PENDING: "⏳ 待执行",
        RetrievalStatus.IN_PROGRESS: "🔄 进行中",
    }

    def get_source_status(self, result: MultiSourceRetrievalResult) -> Dict[str, str]:
        """获取各知识源的状态摘要"""
        return {
            source: self._STATUS_LABELS.get(retrieval.status, "❓ 未知")
            for source, retrieval in result.results.items()
        }

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

    def get_langchain_retriever(self, source: str = "vector", **kwargs):
        """获取 LangChain Retriever（source: "vector" / "wiki"）"""
        if source == "vector":
            return self.vector_searcher.as_retriever(**kwargs)
        elif source == "wiki":
            from langchain_community.tools import WikipediaQueryRun
            return WikipediaQueryRun(api_wrapper=_build_wiki_wrapper(kwargs.get("top_k", 3)))
        else:
            raise ValueError(f"不支持的知识源类型: {source}")

    def get_multi_source_retriever(self, sources: List[str] = None, **kwargs):
        """获取多源检索 Retriever"""
        from app.rag.langchain_components import MultiSourceRetriever

        if sources is None:
            sources = ["vector", "wiki"]

        vector_retriever = None
        wiki_wrapper = None
        if "vector" in sources:
            vector_retriever = self.vector_searcher.as_retriever(**kwargs)
        if "wiki" in sources:
            wiki_wrapper = _build_wiki_wrapper(kwargs.get("top_k", 3))

        return MultiSourceRetriever(
            vector_retriever=vector_retriever,
            wiki_wrapper=wiki_wrapper,
            k=kwargs.get("k", 4),
            source_weights=kwargs.get("source_weights", {"vector": 0.4, "kg": 0.4, "wiki": 0.2})
        )
