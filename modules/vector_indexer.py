import os
import json
import argparse
import sys
import glob

sys.path.append('modules')
sys.path.append('backend')


class VectorIndexer:
    """向量数据库索引构建器"""
    
    def __init__(self, project_name="project_v1", persist_dir="./data/vector_db", encoder_type="default"):
        self.project_name = project_name
        self.project_dir = f"data/{project_name}"
        self.persist_dir = persist_dir
        self.encoder_type = encoder_type  # "default" 或 "qwen3"
        self.searcher = None
        self.hierarchical_indexer = None
        
    def _init_searcher(self):
        """延迟初始化搜索器"""
        if self.searcher is None:
            from app.search.vector_searcher import VectorSearcher
            self.searcher = VectorSearcher(
                collection_name=f"{self.project_name}_docs",
                persist_dir=self.persist_dir
            )
        return self.searcher
        
    def _init_hierarchical_indexer(self):
        """延迟初始化层级索引器 (Qwen3-Embedding)"""
        if self.hierarchical_indexer is None:
            from app.search.hierarchical_index import HierarchicalVectorIndex
            self.hierarchical_indexer = HierarchicalVectorIndex(
                project_name=self.project_name,
                persist_dir=self.persist_dir
            )
        return self.hierarchical_indexer
        
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
        
    def _get_data_files(self):
        """获取所有可用的数据文件"""
        files = {}
        
        possible_raw_paths = [
            os.path.join(self.project_dir, "raw_data", "raw_data.txt"),
            os.path.join(self.project_dir, "..", "raw_data", "raw_data.txt"),
            "data/raw_data/raw_data.txt",
        ]
        for path in possible_raw_paths:
            if os.path.exists(path):
                files['raw'] = path
                break
                
        graph_file = os.path.join(self.project_dir, "knowledge_graph", "knowledge_graph.json")
        if os.path.exists(graph_file):
            files['graph'] = graph_file
        else:
            history_files = glob.glob(os.path.join(self.project_dir, "history", "*.json"))
            if history_files:
                latest_history = max(history_files, key=os.path.getmtime)
                with open(latest_history, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                if config.get('kg_paths'):
                    latest_kg = config['kg_paths'][-1]
                    if os.path.exists(latest_kg):
                        files['graph'] = latest_kg
                        
        clean_data_paths = [
            os.path.join(self.project_dir, "base_filtered.json"),
            "data/clean_data_res_doc2_300epoch.json",
        ]
        for path in clean_data_paths:
            if os.path.exists(path):
                files['clean'] = path
                break
                
        return files
        
    def index_raw_documents(self):
        """将原始文档向量化入库"""
        files = self._get_data_files()
        
        if 'raw' not in files:
            print(f"[警告] 原始数据文件不存在")
            return 0
            
        raw_file = files['raw']
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
        
    def index_graph_sentences(self):
        """将知识图谱中的句子向量化入库"""
        files = self._get_data_files()
        
        if 'graph' not in files:
            print(f"[警告] 知识图谱文件不存在")
            return 0
            
        graph_file = files['graph']
        graph_records = []
        
        # 尝试直接加载 data.json（可能是标准 JSON 格式）
        try:
            with open(graph_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if 'sents' in data:
                    return self._index_sentences_from_list(data['sents'])
                elif 'nodes' in data and 'links' in data:
                    sents = []
                    for link in data.get('links', []):
                        if 'sent' in link:
                            sents.append(link['sent'])
                    if sents:
                        return self._index_sentences_from_list(sents)
        except json.JSONDecodeError:
            # 可能是 JSONL 格式
            graph_records = self._load_jsonl(graph_file)
            
        if not graph_records:
            print(f"[警告] 知识图谱中没有数据")
            return 0
            
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
        files = self._get_data_files()
        
        if 'graph' not in files:
            print(f"[警告] 知识图谱文件不存在")
            return 0
            
        graph_file = files['graph']
        graph_records = []
        
        # 尝试直接加载 data.json（可能是标准 JSON 格式）
        try:
            with open(graph_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if 'nodes' in data and 'links' in data:
                    return self._index_triples_from_nodes_links(data)
        except json.JSONDecodeError:
            # 可能是 JSONL 格式
            graph_records = self._load_jsonl(graph_file)
                
        if not graph_records:
            print(f"[警告] 知识图谱文件不存在")
            return 0
            
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
            name = link.get('name', '')
            if source_idx in node_map and target_idx in node_map and name:
                triple_str = f"({node_map[source_idx]} {name} {node_map[target_idx]})"
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
        
    def index_with_qwen3(self, data_source=None, index_level="all"):
        """
        使用 Qwen3-Embedding-8B 构建层级向量索引
        
        Args:
            data_source: 数据源路径（可选，默认自动检测）
            index_level: 索引层级 ("all", "sentence", "chunk", "document")
            
        Returns:
            Dict: 索引结果
        """
        print("=" * 50)
        print("使用 Qwen3-Embedding-8B 构建层级向量索引...")
        print("=" * 50)
        
        indexer = self._init_hierarchical_indexer()
        files = self._get_data_files()
        
        if data_source is None:
            # 优先使用清洗后的数据
            if 'clean' in files:
                data_source = files['clean']
            elif 'graph' in files:
                data_source = files['graph']
            elif 'raw' in files:
                data_source = files['raw']
            else:
                print("[错误] 未找到可用数据源")
                return {"error": "No data source found"}
                
        print(f"[信息] 数据源: {data_source}")
        
        documents = []
        metadata_list = []
        
        if isinstance(data_source, str):
            if data_source.endswith('.json'):
                with open(data_source, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                        if isinstance(data, list):
                            documents = [item.get('text', item.get('content', '')) for item in data if isinstance(item, dict)]
                            metadata_list = [item for item in data if isinstance(item, dict)]
                        elif isinstance(data, dict):
                            if 'nodes' in data:
                                documents = [node.get('name', '') for node in data.get('nodes', [])]
                            elif 'sents' in data:
                                documents = data['sents']
                    except json.JSONDecodeError:
                        # JSONL 格式
                        data = self._load_jsonl(data_source)
                        documents = [item.get('sentText', str(item)) for item in data]
            elif data_source.endswith('.jsonl'):
                data = self._load_jsonl(data_source)
                documents = [item.get('sentText', str(item)) for item in data]
            elif data_source.endswith('.txt'):
                with open(data_source, 'r', encoding='utf-8') as f:
                    documents = [line.strip() for line in f if line.strip()]
                    
        if not documents:
            print("[警告] 没有加载到任何文档")
            return {"error": "No documents loaded"}
            
        print(f"[信息] 加载了 {len(documents)} 条文档")
        
        result = indexer.index_documents_batch(
            documents,
            metadata_list if metadata_list else None,
            index_level=index_level
        )
        
        indexer.save_index_metadata()
        
        print("=" * 50)
        print("Qwen3-Embedding 层级索引构建完成！")
        print("=" * 50)
        
        return result
        
    def search_qwen3(self, query, top_k=5, level="chunk", representation="original"):
        """
        使用 Qwen3-Embedding 进行语义搜索
        
        Args:
            query: 查询文本
            top_k: 返回数量
            level: 搜索层级
            representation: 表征类型
            
        Returns:
            Dict: 搜索结果
        """
        indexer = self._init_hierarchical_indexer()
        return indexer.search(query, top_k=top_k, level=level, representation=representation)
        
    def hybrid_search_qwen3(self, query, top_k=5, alpha=0.7):
        """
        Qwen3 混合搜索（稠密+稀疏）
        
        Args:
            query: 查询文本
            top_k: 返回数量
            alpha: 稠密向量权重
            
        Returns:
            List: 混合搜索结果
        """
        indexer = self._init_hierarchical_indexer()
        return indexer.hybrid_search(query, top_k=top_k, alpha=alpha)
        
    def get_qwen3_index_info(self):
        """获取 Qwen3 索引信息"""
        indexer = self._init_hierarchical_indexer()
        info = {}
        for level in ["sentence", "chunk", "document"]:
            info[level] = indexer.get_collection_info(level)
        return info


def main():
    parser = argparse.ArgumentParser(description='构建向量数据库索引')
    parser.add_argument('--project', type=str, default='project_v1', 
                        help='项目名称 (default: project_v1)')
    parser.add_argument('--persist-dir', type=str, default='./data/vector_db',
                        help='向量数据库持久化路径')
    parser.add_argument('--source', type=str, choices=['raw', 'graph', 'triples', 'all'],
                        default='all', help='索引来源 (默认编码器)')
    parser.add_argument('--encoder', type=str, choices=['default', 'qwen3', 'hybrid'],
                        default='default', help='编码器类型: default=paraphrase-multilingual, qwen3=Qwen3-Embedding-8B, hybrid=混合搜索')
    parser.add_argument('--data-source', type=str, default=None,
                        help='数据源路径 (用于 Qwen3 索引)')
    parser.add_argument('--level', type=str, choices=['all', 'sentence', 'chunk', 'document'],
                        default='all', help='层级索引级别 (Qwen3 模式)')
    parser.add_argument('--search', type=str, default=None,
                        help='执行搜索查询')
    parser.add_argument('--top-k', type=int, default=5,
                        help='返回结果数量')
    args = parser.parse_args()
    
    indexer = VectorIndexer(
        project_name=args.project,
        persist_dir=args.persist_dir,
        encoder_type=args.encoder
    )
    
    if args.encoder == 'default':
        if args.source == 'raw':
            indexer.index_raw_documents()
        elif args.source == 'graph':
            indexer.index_graph_sentences()
        elif args.source == 'triples':
            indexer.index_graph_triples()
        else:
            indexer.index_all()
    else:
        if args.search:
            if args.encoder == 'hybrid':
                results = indexer.hybrid_search_qwen3(args.search, top_k=args.top_k)
            else:
                results = indexer.search_qwen3(args.search, top_k=args.top_k)
            print(f"\n搜索结果 (查询: {args.search}):")
            print("-" * 50)
            if isinstance(results, list):
                for i, r in enumerate(results, 1):
                    print(f"\n{i}. {r.get('document', '')[:200]}...")
                    print(f"   分数: {r.get('final_score', r.get('distance', 'N/A')):.4f}")
            else:
                for level, level_results in results.items():
                    if level_results.get('documents'):
                        print(f"\n[{level.upper()} 层级]")
                        for doc, dist in zip(level_results['documents'][:3], level_results['distances'][:3]):
                            print(f"  - {doc[:100]}... (距离: {dist:.4f})")
        else:
            result = indexer.index_with_qwen3(
                data_source=args.data_source,
                index_level=args.level
            )

            print("\n索引信息:")
            info = indexer.get_qwen3_index_info()
            for level, level_info in info.items():
                if 'error' not in level_info:
                    print(f"  {level}: {level_info.get('count', 0)} 条向量")


if __name__ == '__main__':
    main()
