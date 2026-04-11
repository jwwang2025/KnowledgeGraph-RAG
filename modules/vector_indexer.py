import os
import json
import argparse
import sys
import glob

sys.path.append('modules')
sys.path.append('backend')


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
            from app.search.vector_searcher import VectorSearcher
            self.searcher = VectorSearcher(
                collection_name=f"{self.project_name}_docs",
                persist_dir=self.persist_dir
            )
        return self.searcher
        
    def index_raw_documents(self):
        """将原始文档向量化入库"""
        # 支持多个可能的路径
        possible_paths = [
            os.path.join(self.project_dir, "raw_data", "raw_data.txt"),
            os.path.join(self.project_dir, "..", "raw_data", "raw_data.txt"),
            "data/raw_data/raw_data.txt",
        ]
        
        raw_file = None
        for path in possible_paths:
            if os.path.exists(path):
                raw_file = path
                break
                
        if not raw_file:
            print(f"[警告] 原始数据文件不存在，搜索路径: {possible_paths}")
            return 0
            
        print(f"[信息] 使用原始数据文件: {raw_file}")
            
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
        
    def _load_jsonl(self, file_path):
        """加载 JSONL 格式文件（每行一个 JSON 对象）
        
        Args:
            file_path: JSONL 文件路径
            
        Returns:
            list: JSON 对象列表
        """
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"[警告] 第 {line_num} 行 JSON 解析失败: {e}")
                    continue
        return data
        
    def index_graph_sentences(self):
        """将知识图谱中的句子向量化入库"""
        graph_file = os.path.join(self.project_dir, "knowledge_graph", "knowledge_graph.json")
        graph_records = []
        
        # 尝试多个可能的路径
        if not os.path.exists(graph_file):
            # 尝试从最新迭代版本加载
            history_files = glob.glob(os.path.join(self.project_dir, "history", "*.json"))
            if history_files:
                latest_history = max(history_files, key=os.path.getmtime)
                print(f"[信息] 从历史文件加载: {latest_history}")
                with open(latest_history, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    
                # 获取最新迭代的知识图谱
                if config.get('kg_paths'):
                    latest_kg = config['kg_paths'][-1]
                    if os.path.exists(latest_kg):
                        graph_records = self._load_jsonl(latest_kg)
                        
        if not graph_records and os.path.exists(graph_file):
            # 尝试直接加载 data.json（可能是标准 JSON 格式）
            try:
                with open(graph_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 如果是包含 sents 字段的标准格式
                    if 'sents' in data:
                        return self._index_sentences_from_list(data['sents'])
            except json.JSONDecodeError:
                # 可能是 JSONL 格式
                graph_records = self._load_jsonl(graph_file)
                
        if not graph_records:
            print(f"[警告] 知识图谱文件不存在")
            return 0
            
        # 从 JSONL 记录中提取句子
        sents = [record.get('sentText', '') for record in graph_records if record.get('sentText')]
        
        if not sents:
            print(f"[警告] 知识图谱中没有句子")
            return 0
            
        searcher = self._init_searcher()
        
        texts = sents
        ids = [f"sent_{i}" for i in range(len(sents))]
        metadatas = [{"source": "graph_sents", "sent_num": i} for i in range(len(sents))]
        
        searcher.add_documents(texts, ids, metadatas)
        print(f"[完成] 已索引 {len(texts)} 条图谱句子")
        return len(texts)
        
    def _index_sentences_from_list(self, sents):
        """从句子列表创建索引（用于标准 JSON 格式）
        
        Args:
            sents: 句子列表
            
        Returns:
            int: 索引的句子数量
        """
        if not sents:
            print(f"[警告] 句子列表为空")
            return 0
            
        searcher = self._init_searcher()
        
        texts = sents
        ids = [f"sent_{i}" for i in range(len(sents))]
        metadatas = [{"source": "graph_sents", "sent_num": i} for i in range(len(sents))]
        
        searcher.add_documents(texts, ids, metadatas)
        print(f"[完成] 已索引 {len(texts)} 条图谱句子")
        return len(texts)
        
    def index_graph_triples(self):
        """将知识图谱中的三元组向量化入库"""
        graph_file = os.path.join(self.project_dir, "knowledge_graph", "knowledge_graph.json")
        graph_records = []
        
        # 尝试多个可能的路径
        if not os.path.exists(graph_file):
            # 尝试从最新迭代版本加载
            history_files = glob.glob(os.path.join(self.project_dir, "history", "*.json"))
            if history_files:
                latest_history = max(history_files, key=os.path.getmtime)
                print(f"[信息] 从历史文件加载: {latest_history}")
                with open(latest_history, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    
                # 获取最新迭代的知识图谱
                if config.get('kg_paths'):
                    latest_kg = config['kg_paths'][-1]
                    if os.path.exists(latest_kg):
                        graph_records = self._load_jsonl(latest_kg)
                        
        if not graph_records and os.path.exists(graph_file):
            # 尝试直接加载 data.json（可能是标准 JSON 格式）
            try:
                with open(graph_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 如果是包含 nodes 和 links 字段的标准格式
                    if 'nodes' in data and 'links' in data:
                        return self._index_triples_from_nodes_links(data)
            except json.JSONDecodeError:
                # 可能是 JSONL 格式
                graph_records = self._load_jsonl(graph_file)
                
        if not graph_records:
            print(f"[警告] 知识图谱文件不存在")
            return 0
            
        # 从 JSONL 记录中提取三元组
        triples = self._extract_triples_from_records(graph_records)
        
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
        
    def _extract_triples_from_records(self, records):
        """从 JSONL 记录中提取三元组
        
        Args:
            records: JSONL 解析后的记录列表
            
        Returns:
            list: 三元组字符串列表
        """
        triples = []
        for record in records:
            # 跳过没有关系提及的记录
            if 'relationMentions' not in record:
                continue
                
            sent_text = record.get('sentText', '')
            relation_mentions = record.get('relationMentions', [])
            
            for mention in relation_mentions:
                em1 = mention.get('em1Text', '')
                em2 = mention.get('em2Text', '')
                label = mention.get('label', '')
                
                if em1 and em2 and label:
                    triple_str = f"({em1} {label} {em2})"
                    triples.append(triple_str)
                    
        return triples
        
    def _index_triples_from_nodes_links(self, data):
        """从 nodes 和 links 字段创建三元组索引（用于标准 JSON 格式）
        
        Args:
            data: 包含 nodes 和 links 的字典
            
        Returns:
            int: 索引的三元组数量
        """
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
