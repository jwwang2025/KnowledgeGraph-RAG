"""
Wikipedia 搜索器
基于 LangChain 的 WikipediaAPIWrapper 封装

支持 LangChain 工具接口，同时保留原有功能
"""

import re
from typing import List, Dict, Any, Optional, Union

# LangChain Wikipedia
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain_core.documents import Document

from opencc import OpenCC

cc_s2t = OpenCC('s2t')  # 简体中文转繁体中文
cc_t2s = OpenCC('t2s')  # 繁体中文转简体中文


class WikiSearcher:
    """
    Wikipedia 搜索器
    
    支持:
    - 传统搜索接口
    - LangChain 工具接口
    - LangChain DocumentLoader 接口
    """
    
    def __init__(
        self,
        user_agent: str = "KnowledgeGraph-RAG/1.0",
        language: str = "zh",
        top_k_results: int = 3
    ):
        """
        初始化 WikiSearcher
        
        Args:
            user_agent: 用户代理
            language: 语言代码 (zh/en)
            top_k_results: 返回结果数量
        """
        self.user_agent = user_agent
        self.language = language
        self.top_k_results = top_k_results
        
        # 初始化 LangChain Wikipedia API Wrapper
        self._wikipedia_wrapper = None
        
        # 初始化原生 WikipediaAPI
        self._wiki_api = None
    
    @property
    def wikipedia_api(self):
        """获取原生 WikipediaAPI"""
        if self._wiki_api is None:
            import wikipediaapi
            self._wiki_api = wikipediaapi.Wikipedia(
                user_agent=self.user_agent,
                language=self.language
            )
        return self._wiki_api
    
    @property
    def langchain_wrapper(self) -> WikipediaAPIWrapper:
        """获取 LangChain Wikipedia API Wrapper"""
        if self._wikipedia_wrapper is None:
            self._wikipedia_wrapper = WikipediaAPIWrapper(
                top_k_results=self.top_k_results,
                language=self.language
            )
        return self._wikipedia_wrapper
    
    @property
    def langchain_tool(self) -> WikipediaQueryRun:
        """获取 LangChain Wikipedia 工具"""
        return WikipediaQueryRun(api_wrapper=self.langchain_wrapper)
    
    def search(self, query: str) -> Optional[Any]:
        """
        搜索 Wikipedia 页面 (传统接口)
        
        Args:
            query: 搜索关键词
        
        Returns:
            Wikipedia 页面对象 或 None
        """
        result = None
        
        try:
            # 尝试原始查询
            page = self.wikipedia_api.page(query)
            
            if not page.exists():
                # 尝试简繁转换
                converted_query = cc_s2t.convert(query) if self.language == "zh" else query
                if converted_query != query:
                    page = self.wikipedia_api.page(converted_query)
            
            if page.exists():
                result = page
                
        except Exception as e:
            print(f"[WikiSearcher] 搜索失败: {e}")
        
        return result
    
    def search_langchain(self, query: str) -> str:
        """
        LangChain 风格的搜索
        
        Args:
            query: 搜索关键词
        
        Returns:
            Wikipedia 摘要文本
        """
        return self.langchain_wrapper.run(query)
    
    def load_document(self, query: str) -> List[Document]:
        """
        加载 Wikipedia 文档为 LangChain Document
        
        Args:
            query: 搜索关键词
        
        Returns:
            Document 列表
        """
        page = self.search(query)
        
        if page is None:
            return []
        
        # 转换为简体中文
        title = cc_t2s.convert(page.title) if self.language == "zh" else page.title
        summary = cc_t2s.convert(page.summary)[:1000] if self.language == "zh" else page.summary[:1000]
        
        # 创建 Document
        doc = Document(
            page_content=f"标题: {title}\n\n摘要: {summary}",
            metadata={
                "source": "wikipedia",
                "title": title,
                "url": page.fullurl,
                "language": self.language,
                "query": query
            }
        )
        
        return [doc]
    
    def get_full_content(self, query: str, max_length: int = 5000) -> Optional[Dict[str, Any]]:
        """
        获取 Wikipedia 完整内容
        
        Args:
            query: 搜索关键词
            max_length: 最大内容长度
        
        Returns:
            包含 title, content, url 的字典
        """
        page = self.search(query)
        
        if page is None:
            return None
        
        title = cc_t2s.convert(page.title) if self.language == "zh" else page.title
        content = cc_t2s.convert(page.text)[:max_length] if self.language == "zh" else page.text[:max_length]
        
        return {
            "title": title,
            "content": content,
            "url": page.fullurl,
            "summary": page.summary[:500]
        }
    
    def search_multi(
        self,
        queries: List[str],
        combine: bool = True
    ) -> Union[List[Dict[str, Any]], str]:
        """
        多关键词搜索
        
        Args:
            queries: 关键词列表
            combine: 是否合并结果
        
        Returns:
            如果 combine=True 返回合并文本，否则返回字典列表
        """
        results = []
        
        for query in queries:
            page = self.search(query)
            if page:
                title = cc_t2s.convert(page.title) if self.language == "zh" else page.title
                summary = cc_t2s.convert(page.summary)[:500] if self.language == "zh" else page.summary[:500]
                
                results.append({
                    "title": title,
                    "summary": summary,
                    "url": page.fullurl
                })
        
        if not results:
            return [] if not combine else ""
        
        if combine:
            combined = "\n\n".join([
                f"【{r['title']}】{r['summary']}"
                for r in results
            ])
            return combined
        
        return results


# =============================================================================
# LangChain DocumentLoader
# =============================================================================

class WikipediaDocumentLoader:
    """
    LangChain 兼容的 Wikipedia Document Loader
    
    用于将 Wikipedia 页面加载为 LangChain Document 对象
    """
    
    def __init__(
        self,
        query: str,
        load_max_docs: int = 1,
        language: str = "zh"
    ):
        """
        初始化 Wikipedia Document Loader
        
        Args:
            query: 搜索查询
            load_max_docs: 最大加载文档数
            language: 语言代码
        """
        self.query = query
        self.load_max_docs = load_max_docs
        self.language = language
    
    def load(self) -> List[Document]:
        """加载文档"""
        searcher = WikiSearcher(language=self.language)
        return searcher.load_document(self.query)
    
    def lazy_load(self):
        """懒加载文档迭代器"""
        for doc in self.load():
            yield doc


# =============================================================================
# 便捷函数
# =============================================================================

def create_wikipedia_retriever(
    top_k_results: int = 3,
    language: str = "zh"
) -> WikipediaQueryRun:
    """
    创建 LangChain Wikipedia 检索工具
    
    Args:
        top_k_results: 返回结果数量
        language: 语言代码
    
    Returns:
        WikipediaQueryRun 实例
    """
    wrapper = WikipediaAPIWrapper(top_k_results=top_k_results, language=language)
    return WikipediaQueryRun(api_wrapper=wrapper)


def search_wikipedia(query: str, language: str = "zh") -> Optional[Dict[str, Any]]:
    """
    便捷的 Wikipedia 搜索函数
    
    Args:
        query: 搜索关键词
        language: 语言代码
    
    Returns:
        搜索结果字典
    """
    searcher = WikiSearcher(language=language)
    page = searcher.search(query)
    
    if page is None:
        return None
    
    return {
        "title": cc_t2s.convert(page.title),
        "summary": cc_t2s.convert(page.summary)[:500],
        "url": page.fullurl
    }
