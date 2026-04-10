import os
import json
import argparse
import sys

sys.path.append('modules')
sys.path.append('backend/app')


class VectorIndexer:
    """向量数据库索引构建器"""
    
    def __init__(self, project_name="project_v1", persist_dir="./data/vector_db"):
        self.project_name = project_name
        self.project_dir = f"data/{project_name}"
        self.persist_dir = persist_dir
        self.searcher = None
        
    def _init_searcher(self):
        """延迟初始化搜索器"""
        if self.searcher is None:
            from app.utils.vector_searcher import VectorSearcher
            self.searcher = VectorSearcher(
                collection_name=f"{self.project_name}_docs",
                persist_dir=self.persist_dir
            )
        return self.searcher
        
    def index_raw_documents(self):
        """将原始文档向量化入库"""
        raw_file = os.path.join(self.project_dir, "raw_data", "raw_data.txt")
        
        if not os.path.exists(raw_file):
            print(f"[警告] 原始数据文件不存在: {raw_file}")
            return 0
            
        with open(raw_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        if not lines:
            print(f"[警告] 原始数据文件为空")
            return 0
            
        searcher = self._init_searcher()
        searcher.reset()
        
        texts = []
        ids = []
        metadatas = []
        
        for i, line in enumerate(lines):
            texts.append(line)
            ids.append(f"doc_{i}")
            metadatas.append({"source": "raw_data", "line_num": i})
            
        searcher.add_documents(texts, ids, metadatas)
        print(f"[完成] 已索引 {len(texts)} 条原始文档")
        return len(texts)
        
    def index_graph_sentences(self):
        """将知识图谱中的句子向量化入库"""
        graph_file = os.path.join(self.project_dir, "data.json")
        
        if not os.path.exists(graph_file):
            print(f"[警告] 知识图谱文件不存在: {graph_file}")
            return 0
            
        with open(graph_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        sents = data.get('sents', [])
        
        if not sents:
            print(f"[警告] 知识图谱中没有句子")
            return 0
            
        searcher = self._init_searcher()
        
        texts = []
        ids = []
        metadatas = []
        
        for i, sent in enumerate(sents):
            texts.append(sent)
            ids.append(f"sent_{i}")
            metadatas.append({"source": "graph_sents", "sent_num": i})
            
        searcher.add_documents(texts, ids, metadatas)
        print(f"[完成] 已索引 {len(texts)} 条图谱句子")
        return len(texts)
        
    def index_graph_triples(self):
        """将知识图谱中的三元组向量化入库"""
        graph_file = os.path.join(self.project_dir, "data.json")
        
        if not os.path.exists(graph_file):
            print(f"[警告] 知识图谱文件不存在: {graph_file}")
            return 0
            
        with open(graph_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        nodes = data.get('nodes', [])
        links = data.get('links', [])
        
        if not nodes or not links:
            print(f"[警告] 知识图谱为空")
            return 0
            
        node_map = {i: node['name'] for i, node in enumerate(nodes)}
        triples = []
        
        for link in links:
            source_idx = int(link['source'])
            target_idx = int(link['target'])
            if source_idx in node_map and target_idx in node_map:
                triple_str = f"({node_map[source_idx]} {link['name']} {node_map[target_idx]})"
                triples.append(triple_str)
                
        if not triples:
            print(f"[警告] 知识图谱中没有三元组")
            return 0
            
        searcher = self._init_searcher()
        
        texts = triples
        ids = [f"triple_{i}" for i in range(len(triples))]
        metadatas = [{"source": "graph_triples"} for _ in triples]
        
        searcher.add_documents(texts, ids, metadatas)
        print(f"[完成] 已索引 {len(triples)} 条三元组")
        return len(triples)
        
    def index_all(self):
        """构建所有索引"""
        print("=" * 50)
        print("开始构建向量索引...")
        print("=" * 50)
        
        total = 0
        total += self.index_raw_documents()
        total += self.index_graph_sentences()
        total += self.index_graph_triples()
        
        if self.searcher:
            info = self.searcher.get_collection_info()
            print("=" * 50)
            print(f"索引构建完成！总计 {info['count']} 条向量")
            print(f"Collection: {info['name']}")
            print("=" * 50)
            
        return total


def main():
    parser = argparse.ArgumentParser(description='构建向量数据库索引')
    parser.add_argument('--project', type=str, default='project_v1', 
                        help='项目名称 (default: project_v1)')
    parser.add_argument('--persist-dir', type=str, default='./data/vector_db',
                        help='向量数据库持久化路径')
    parser.add_argument('--source', type=str, choices=['raw', 'graph', 'triples', 'all'],
                        default='all', help='索引来源')
    args = parser.parse_args()
    
    indexer = VectorIndexer(
        project_name=args.project,
        persist_dir=args.persist_dir
    )
    
    if args.source == 'raw':
        indexer.index_raw_documents()
    elif args.source == 'graph':
        indexer.index_graph_sentences()
    elif args.source == 'triples':
        indexer.index_graph_triples()
    else:
        indexer.index_all()


if __name__ == '__main__':
    main()
