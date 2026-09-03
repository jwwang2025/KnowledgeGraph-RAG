"""引用溯源（Citations）模块：确保回答中关键细节与检索来源可追溯验证。"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum


class CitationSource(Enum):
    """引用来源枚举"""
    KNOWLEDGE_GRAPH = "knowledge_graph"    # 知识图谱三元组
    VECTOR_DOCUMENT = "vector_document"    # 向量数据库文档
    WIKIPEDIA = "wikipedia"               # Wikipedia 百科
    IMAGE = "image"                        # 图像搜索
    USER_HISTORY = "user_history"         # 用户历史对话
    SYSTEM_CONTEXT = "system_context"      # 系统上下文


class CitationType(Enum):
    """引用类型枚举"""
    DIRECT_QUOTE = "direct_quote"          # 直接引用
    INDIRECT_REFERENCE = "indirect_ref"    # 间接引用
    SEMANTIC_MATCH = "semantic_match"      # 语义匹配
    INFERRED = "inferred"                  # 推断生成


# 各来源的显示图标
_SOURCE_ICONS = {
    CitationSource.KNOWLEDGE_GRAPH: "📊",
    CitationSource.VECTOR_DOCUMENT: "📄",
    CitationSource.WIKIPEDIA: "🌐",
    CitationSource.IMAGE: "🖼️",
}


@dataclass
class Citation:
    """单条引用记录：追踪回答中每个关键细节的来源"""
    # 基础信息
    source: CitationSource                  # 来源类型
    source_id: str                          # 来源唯一标识（如文档ID、三元组哈希等）
    source_name: str                        # 来源名称（如文档名、Wikipedia词条名）

    # 内容信息
    original_text: str                      # 原文内容
    excerpt: str                            # 截取的引用片段
    position: Tuple[int, int] = (0, 0)     # 在原文中的位置 (start, end)

    # 关联信息
    related_entities: List[str] = field(default_factory=list)  # 关联的实体列表
    related_triples: List[tuple] = field(default_factory=list)  # 关联的三元组

    # 质量指标
    relevance_score: float = 0.0           # 相关性得分 (0-1)
    confidence_score: float = 0.0           # 置信度得分 (0-1)
    citation_type: CitationType = CitationType.DIRECT_QUOTE

    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)   # 附加元数据
    retrieved_at: float = field(default_factory=time.time)      # 检索时间

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "source": self.source.value,
            "source_id": self.source_id,
            "source_name": self.source_name,
            "excerpt": self.excerpt,
            "relevance_score": round(self.relevance_score, 3),
            "confidence_score": round(self.confidence_score, 3),
            "citation_type": self.citation_type.value,
            "related_entities": self.related_entities,
            "metadata": self.metadata
        }

    def get_markdown_ref(self) -> str:
        """获取 Markdown 格式的引用标记"""
        return f"[{self.source_name}](ref:{self.source_id})"

    def get_inline_mark(self) -> str:
        """获取行内引用标记"""
        return f"【{self.source.value}:{self.source_name}】"


def _sorted_by_relevance(citations: List[Citation]) -> List[Citation]:
    """按相关性降序排序"""
    return sorted(citations, key=lambda x: x.relevance_score, reverse=True)


def _truncate(text: str, limit: int) -> str:
    """截断过长文本并附加省略号"""
    return text[:limit] + ('...' if len(text) > limit else '')


@dataclass
class CitationSet:
    """引用集合：支持按来源类型分组、排序、去重等操作"""
    citations: List[Citation] = field(default_factory=list)
    query: str = ""                        # 关联的查询
    generated_at: float = field(default_factory=time.time)

    def add(self, citation: Citation):
        """添加引用（按 source_id + excerpt 去重）"""
        for existing in self.citations:
            if existing.source_id == citation.source_id and existing.excerpt == citation.excerpt:
                return  # 已存在，跳过
        self.citations.append(citation)

    def add_batch(self, citations: List[Citation]):
        """批量添加引用"""
        for c in citations:
            self.add(c)

    def get_by_source(self, source: CitationSource) -> List[Citation]:
        """按来源类型获取引用"""
        return [c for c in self.citations if c.source == source]

    def get_sorted_by_relevance(self) -> List[Citation]:
        """按相关性排序"""
        return _sorted_by_relevance(self.citations)

    def get_sorted_by_confidence(self) -> List[Citation]:
        """按置信度排序"""
        return sorted(self.citations, key=lambda x: x.confidence_score, reverse=True)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "query": self.query,
            "total_citations": len(self.citations),
            "citations": [c.to_dict() for c in self.citations],
            "sources_breakdown": self.get_sources_summary(),
            "generated_at": self.generated_at
        }

    def get_sources_summary(self) -> Dict[str, int]:
        """获取各来源类型的引用数量"""
        summary = {}
        for citation in self.citations:
            source_name = citation.source.value
            summary[source_name] = summary.get(source_name, 0) + 1
        return summary

    def format_for_display(self) -> str:
        """格式化输出用于显示"""
        if not self.citations:
            return ""

        lines = ["\n--- 参考来源 ---"]
        for i, c in enumerate(self.get_sorted_by_relevance(), 1):
            lines.append(f"[{i}] {c.source_name} ({c.source.value})")
            lines.append(f"    {_truncate(c.excerpt, 100)}")
            lines.append(f"    相关度: {c.relevance_score:.2f} | 置信度: {c.confidence_score:.2f}")
            lines.append("")

        return "\n".join(lines)


@dataclass
class CitationContext:
    """引用上下文：在 RAG 流程中收集和聚合引用"""
    # 检索阶段收集的原始引用
    triples_citations: List[Citation] = field(default_factory=list)
    document_citations: List[Citation] = field(default_factory=list)
    wiki_citations: List[Citation] = field(default_factory=list)
    image_citations: List[Citation] = field(default_factory=list)

    # 生成阶段嵌入的引用
    embedded_citations: List[Citation] = field(default_factory=list)

    # 原始检索结果（用于回溯）
    raw_triples: List[tuple] = field(default_factory=list)
    raw_documents: List[str] = field(default_factory=list)
    raw_wiki: Optional[Dict[str, str]] = None

    def merge_all(self) -> CitationSet:
        """合并所有引用"""
        citation_set = CitationSet()
        citation_set.citations = (
            self.triples_citations +
            self.document_citations +
            self.wiki_citations +
            self.image_citations +
            self.embedded_citations
        )
        return citation_set

    def get_all_source_ids(self) -> List[str]:
        """获取所有引用的来源ID"""
        return [c.source_id for c in self.merge_all().citations]


class CitationGenerator:
    """引用生成器：从知识图谱三元组 / 文档片段 / Wikipedia / 图像结果生成引用记录"""

    @staticmethod
    def generate_triple_citations(triples: List[tuple],
                                   query: str,
                                   relevance_scores: List[float] = None) -> List[Citation]:
        """从知识图谱三元组生成引用"""
        citations = []

        for i, triple in enumerate(triples):
            s, p, o = triple[0], triple[1], triple[2]
            triple_str = f"{s}-{p}-{o}"

            # 生成引用片段
            excerpt = f"({s} {p} {o})"

            # 计算相关性得分
            relevance = relevance_scores[i] if relevance_scores and i < len(relevance_scores) else 0.8

            citations.append(Citation(
                source=CitationSource.KNOWLEDGE_GRAPH,
                source_id=f"kg_{hash(triple_str) % 100000:05d}",
                source_name=f"知识图谱:{s}",
                original_text=excerpt,
                excerpt=excerpt,
                position=(0, len(excerpt)),
                related_entities=[s, o],
                related_triples=[triple],
                relevance_score=relevance,
                confidence_score=0.95,  # 知识图谱数据置信度较高
                citation_type=CitationType.DIRECT_QUOTE,
                metadata={"subject": s, "predicate": p, "object": o}
            ))

        return citations

    @staticmethod
    def generate_document_citations(documents: List[str],
                                      doc_ids: List[str] = None,
                                      query: str = "",
                                      relevance_scores: List[float] = None) -> List[Citation]:
        """从文档片段生成引用"""
        citations = []

        for i, doc in enumerate(documents):
            source_id = doc_ids[i] if doc_ids and i < len(doc_ids) else f"doc_{i:03d}"

            # 截取引用片段（取前100字符）
            excerpt = doc[:100] if len(doc) > 100 else doc

            relevance = relevance_scores[i] if relevance_scores and i < len(relevance_scores) else 0.7

            citations.append(Citation(
                source=CitationSource.VECTOR_DOCUMENT,
                source_id=source_id,
                source_name=f"文档片段 #{i+1}",
                original_text=doc,
                excerpt=excerpt,
                position=(0, min(100, len(doc))),
                relevance_score=relevance,
                confidence_score=0.8,
                citation_type=CitationType.SEMANTIC_MATCH,
                metadata={
                    "full_text_length": len(doc),
                    "query_context": query[:50] if query else ""
                }
            ))

        return citations

    @staticmethod
    def generate_wiki_citation(wiki_result: Dict[str, str],
                                 query: str = "") -> Optional[Citation]:
        """从 Wikipedia 结果生成引用"""
        if not wiki_result or not wiki_result.get("title"):
            return None

        title = wiki_result["title"]
        summary = wiki_result.get("summary", "")

        # 截取引用片段
        excerpt = summary[:150] if len(summary) > 150 else summary

        return Citation(
            source=CitationSource.WIKIPEDIA,
            source_id=f"wiki_{hash(title) % 100000:05d}",
            source_name=title,
            original_text=summary,
            excerpt=excerpt,
            position=(0, len(excerpt)),
            related_entities=[title],
            relevance_score=0.9,
            confidence_score=0.95,  # Wikipedia 来自权威来源
            citation_type=CitationType.DIRECT_QUOTE,
            metadata={
                "wiki_title": title,
                "full_summary_length": len(summary)
            }
        )

    @staticmethod
    def generate_image_citation(image_url: str,
                                query: str = "") -> Optional[Citation]:
        """从图像搜索结果生成引用"""
        if not image_url:
            return None

        return Citation(
            source=CitationSource.IMAGE,
            source_id=f"img_{hash(image_url) % 100000:05d}",
            source_name="相关图片",
            original_text=image_url,
            excerpt="[相关图片]",
            relevance_score=0.7,
            confidence_score=0.6,
            citation_type=CitationType.INDIRECT_REFERENCE,
            metadata={"image_url": image_url, "query": query}
        )


class CitationEmbedder:
    """引用嵌入器：支持上标数字 [1]、脚注、行内标记三种嵌入格式"""

    def __init__(self, format_type: str = "superscript"):
        """format_type: "superscript" / "footnote" / "inline" """
        self.format_type = format_type

    def embed_citations(self, text: str,
                        citations: List[Citation],
                        matched_segments: List[Tuple[str, int]] = None) -> Tuple[str, List[Citation]]:
        """将引用标记嵌入文本，返回 (嵌入后的文本, 实际使用的引用列表)"""
        if not citations or not matched_segments:
            return text, []

        # 按相关性排序并分配编号
        sorted_citations = _sorted_by_relevance(citations)
        citation_map = {c.source_id: idx + 1 for idx, c in enumerate(sorted_citations)}

        # 替换匹配片段
        result_text = text
        used_indices = set()

        for segment, citation_idx in matched_segments:
            if citation_idx < len(sorted_citations):
                citation = sorted_citations[citation_idx]
                ref_num = citation_map.get(citation.source_id)
                if ref_num:
                    if self.format_type == "inline":
                        replacement = f"{segment}{citation.get_inline_mark()}"
                    else:
                        replacement = f"{segment}[{ref_num}]"

                    result_text = result_text.replace(segment, replacement, 1)
                    used_indices.add(ref_num - 1)

        return result_text, [sorted_citations[i] for i in sorted(used_indices)]

    def add_footnote(self, text: str, citations: List[Citation]) -> str:
        """为文本添加脚注引用"""
        if not citations:
            return text

        footnotes = ["\n\n--- 参考来源 ---\n"]
        for i, citation in enumerate(_sorted_by_relevance(citations), 1):
            source_icon = _SOURCE_ICONS.get(citation.source, "📌")
            footnotes.append(
                f"[{i}] {source_icon} {citation.source_name}\n"
                f"    {_truncate(citation.excerpt, 80)}\n"
            )

        return text + "\n".join(footnotes)

    def format_citations_for_api(self, citations: List[Citation]) -> List[Dict[str, Any]]:
        """格式化引用用于 API 返回"""
        if not citations:
            return []

        return [
            {
                "id": idx + 1,
                "source": c.source.value,
                "source_name": c.source_name,
                "excerpt": c.excerpt,
                "relevance_score": round(c.relevance_score, 3),
                "confidence": round(c.confidence_score, 3),
                "metadata": c.metadata
            }
            for idx, c in enumerate(_sorted_by_relevance(citations))
        ]
