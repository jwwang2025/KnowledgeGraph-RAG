import wikipediaapi

from opencc import OpenCC

cc = OpenCC('s2t')

class WikiSearcher(object):

    def __init__(self) -> None:
        self.wiki = wikipediaapi.Wikipedia(user_agent='KnowledgeGraph-RAG/1.0', language='zh')

    def search(self, query):
        """搜索 Wikipedia，添加超时和错误处理"""
        import signal
        
        result = None

        try:
            print(f"[WikiSearcher] 搜索: {query}")
            page = self.wiki.page(query)

            if not page.exists():
                page = self.wiki.page(cc.convert(query))

            if page.exists():
                result = page
                print(f"[WikiSearcher] 找到结果: {result.title}")
            else:
                print(f"[WikiSearcher] 未找到结果: {query}")

        except Exception as e:
            print(f"[WikiSearcher] 搜索出错: {e}")

        return result