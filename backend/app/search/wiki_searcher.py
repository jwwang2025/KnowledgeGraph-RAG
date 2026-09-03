"""Wikipedia 搜索器：原生 API + LangChain 工具/Loader 封装，含简繁转换。"""
from typing import List, Dict, Any, Optional, Union

from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain_core.documents import Document

from opencc import OpenCC

cc_s2t = OpenCC('s2t')  # 简体 -> 繁体
cc_t2s = OpenCC('t2s')  # 繁体 -> 简体

# 模块级客户端缓存：同一 (user_agent, language) 复用原生 Wikipedia 客户端
_WIKI_API_CACHE: Dict[tuple, Any] = {}
# 同一 (top_k, language) 复用 LangChain Wrapper
_WIKI_WRAPPER_CACHE: Dict[tuple, WikipediaAPIWrapper] = {}


class WikiSearcher:
    """Wikipedia 搜索器：支持传统接口、LangChain 工具接口与 DocumentLoader 接口。"""

    def __init__(
        self,
        user_agent: str = "KnowledgeGraph-RAG/1.0",
        language: str = "zh",
        top_k_results: int = 3
    ):
        self.user_agent = user_agent
        self.language = language
        self.top_k_results = top_k_results

        self._wikipedia_wrapper = None
        self._wiki_api = None

    @property
    def wikipedia_api(self):
        """原生 WikipediaAPI（模块级缓存复用）"""
        if self._wiki_api is None:
            key = (self.user_agent, self.language)
            if key not in _WIKI_API_CACHE:
                import wikipediaapi
                _WIKI_API_CACHE[key] = wikipediaapi.Wikipedia(
                    user_agent=self.user_agent,
                    language=self.language
                )
            self._wiki_api = _WIKI_API_CACHE[key]
        return self._wiki_api

    @property
    def langchain_wrapper(self) -> WikipediaAPIWrapper:
        """LangChain Wikipedia API Wrapper（模块级缓存复用）"""
        if self._wikipedia_wrapper is None:
            key = (self.top_k_results, self.language)
            if key not in _WIKI_WRAPPER_CACHE:
                _WIKI_WRAPPER_CACHE[key] = WikipediaAPIWrapper(
                    top_k_results=self.top_k_results,
                    language=self.language
                )
            self._wikipedia_wrapper = _WIKI_WRAPPER_CACHE[key]
        return self._wikipedia_wrapper

    @property
    def langchain_tool(self) -> WikipediaQueryRun:
        """LangChain Wikipedia 工具"""
        return WikipediaQueryRun(api_wrapper=self.langchain_wrapper)

    def _to_simplified(self, text: str) -> str:
        """zh 语言下繁体转简体，其他语言原样返回"""
        return cc_t2s.convert(text) if self.language == "zh" else text

    def search(self, query: str) -> Optional[Any]:
        """搜索 Wikipedia 页面（zh 下自动尝试简->繁转换查询），返回页面对象或 None"""
        try:
            page = self.wikipedia_api.page(query)

            if not page.exists():
                converted_query = cc_s2t.convert(query) if self.language == "zh" else query
                if converted_query != query:
                    page = self.wikipedia_api.page(converted_query)

            if page.exists():
                return page
        except Exception as e:
            print(f"[WikiSearcher] 搜索失败: {e}")

        return None

    def search_langchain(self, query: str) -> str:
        """LangChain 风格搜索，返回摘要文本"""
        return self.langchain_wrapper.run(query)

    def load_document(self, query: str) -> List[Document]:
        """加载 Wikipedia 页面为 LangChain Document"""
        page = self.search(query)
        if page is None:
            return []

        title = self._to_simplified(page.title)
        summary = self._to_simplified(page.summary)[:1000]

        return [Document(
            page_content=f"标题: {title}\n\n摘要: {summary}",
            metadata={
                "source": "wikipedia",
                "title": title,
                "url": page.fullurl,
                "language": self.language,
                "query": query
            }
        )]

    def get_full_content(self, query: str, max_length: int = 5000) -> Optional[Dict[str, Any]]:
        """获取 Wikipedia 完整内容（title/content 转简体，summary 保留原文）"""
        page = self.search(query)
        if page is None:
            return None

        return {
            "title": self._to_simplified(page.title),
            "content": self._to_simplified(page.text)[:max_length],
            "url": page.fullurl,
            "summary": page.summary[:500]
        }

    def search_multi(
        self,
        queries: List[str],
        combine: bool = True
    ) -> Union[List[Dict[str, Any]], str]:
        """多关键词搜索，combine=True 时合并为文本，否则返回字典列表"""
        results = []
        for query in queries:
            page = self.search(query)
            if page:
                results.append({
                    "title": self._to_simplified(page.title),
                    "summary": self._to_simplified(page.summary)[:500],
                    "url": page.fullurl
                })

        if not results:
            return [] if not combine else ""

        if combine:
            return "\n\n".join(f"【{r['title']}】{r['summary']}" for r in results)

        return results


class WikipediaDocumentLoader:
    """LangChain 兼容的 Wikipedia Document Loader"""

    def __init__(
        self,
        query: str,
        load_max_docs: int = 1,
        language: str = "zh"
    ):
        self.query = query
        self.load_max_docs = load_max_docs
        self.language = language

    def load(self) -> List[Document]:
        return WikiSearcher(language=self.language).load_document(self.query)

    def lazy_load(self):
        """懒加载文档迭代器"""
        yield from self.load()


def create_wikipedia_retriever(
    top_k_results: int = 3,
    language: str = "zh"
) -> WikipediaQueryRun:
    """创建 LangChain Wikipedia 检索工具"""
    wrapper = WikipediaAPIWrapper(top_k_results=top_k_results, language=language)
    return WikipediaQueryRun(api_wrapper=wrapper)


def search_wikipedia(query: str, language: str = "zh") -> Optional[Dict[str, Any]]:
    """便捷的 Wikipedia 搜索函数"""
    page = WikiSearcher(language=language).search(query)
    if page is None:
        return None

    return {
        "title": cc_t2s.convert(page.title),
        "summary": cc_t2s.convert(page.summary)[:500],
        "url": page.fullurl
    }
