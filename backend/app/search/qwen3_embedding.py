"""
Qwen3-Embedding-8b 向量化编码器模块
支持多表征技术和层级向量索引
"""
import os
import json
import torch
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
from functools import lru_cache


class Qwen3EmbeddingEncoder:
    """
    Qwen3-Embedding-8b 向量化编码器
    
    支持:
    - 多表征技术 (Multi-representation): 原始文本、关键词、实体、三元组
    - 层级向量索引: sentence、chunk、document 三级
    - 分布式 ChromaDB 存储
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-8B",
        local_model_path: Optional[str] = None,
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
        batch_size: int = 8,
        max_length: int = 8192
    ):
        """
        初始化 Qwen3-Embedding 编码器
        
        Args:
            model_name: HuggingFace 模型名称
            local_model_path: 本地模型路径（优先使用）
            device: 设备类型，优先使用 GPU
            normalize_embeddings: 是否归一化向量
            batch_size: 批处理大小
            max_length: 最大序列长度
        """
        self.model_name = model_name
        self.local_model_path = local_model_path or self._get_default_model_path()
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self.max_length = max_length
        
        # 自动选择设备
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self._model = None
        self._tokenizer = None
        
    def _get_default_model_path(self) -> str:
        """获取默认本地模型路径"""
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        return os.path.join(project_root, "models", "Qwen3-Embedding-8B")
        
    @property
    def model(self):
        """延迟加载模型"""
        if self._model is None:
            self._load_model()
        return self._model
    
    @property
    def tokenizer(self):
        """延迟加载分词器"""
        if self._tokenizer is None:
            self._load_model()
        return self._tokenizer
    
    def _load_model(self):
        """加载模型和分词器"""
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
        
        print(f"[信息] 加载 Qwen3-Embedding-8B 模型...")
        
        # 确定模型路径
        model_path = self.local_model_path
        if not os.path.exists(model_path):
            print(f"[警告] 本地模型不存在: {model_path}，将从 HuggingFace 下载")
            model_path = self.model_name
            
        # 加载分词器
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # 加载模型
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            trust_remote_code=True,
            device_map="auto" if self.device == "cuda" else None,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        
        if self.device == "cpu":
            self._model = self._model.to("cpu")
            
        self._model.eval()
        
        print(f"[完成] 模型加载成功，设备: {self.device}")
        
    def encode(
        self,
        texts: Union[str, List[str]],
        representation_type: str = "dense",
        prompt_name: Optional[str] = None,
        prompt: Optional[str] = None
    ) -> np.ndarray:
        """
        编码文本为向量
        
        Args:
            texts: 单个文本或文本列表
            representation_type: 表征类型 ('dense', 'sparse', 'bm25')
            prompt_name: 预定义的 prompt 名称
            prompt: 自定义 prompt
            
        Returns:
            numpy.ndarray: 嵌入向量数组
        """
        if isinstance(texts, str):
            texts = [texts]
            
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            # 构建输入
            if prompt:
                inputs = [prompt.format(text=t) for t in batch_texts]
            elif prompt_name:
                inputs = self._apply_prompt(batch_texts, prompt_name)
            else:
                inputs = batch_texts
                
            # Tokenize
            encoded = self.tokenizer(
                inputs,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            # 移动到设备
            if self.device != "cpu":
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                
            # 获取嵌入
            with torch.no_grad():
                outputs = self.model(**encoded)
                embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
                
                if self.normalize_embeddings:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                    
            all_embeddings.append(embeddings.cpu().numpy())
            
        return np.vstack(all_embeddings)
    
    def _apply_prompt(self, texts: List[str], prompt_name: str) -> List[str]:
        """应用预定义的 prompt"""
        prompts = {
            "retrieve.query": "query: {text}",
            "retrieve.passage": "passage: {text}",
            "embeddings": "{text}",
            "symmetric": "{text}",
            "asymmetric": "query: {text}"
        }
        
        template = prompts.get(prompt_name, "{text}")
        return [template.format(text=t) for t in texts]
    
    def encode_multi_representation(
        self,
        text: str,
        representations: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        多表征编码
        
        Args:
            text: 原始文本
            representations: 要生成的表征类型列表
            
        Returns:
            Dict: 各种表征的嵌入向量
        """
        if representations is None:
            representations = ["original", "keywords", "entities", "triples", "summary"]
            
        results = {}
        
        # 1. 原始文本表征
        if "original" in representations:
            results["original"] = self.encode(text)
            
        # 2. 关键词表征
        if "keywords" in representations:
            keywords = self._extract_keywords(text)
            if keywords:
                results["keywords"] = self.encode(" ".join(keywords))
            else:
                results["keywords"] = results.get("original", self.encode(""))
                
        # 3. 实体表征
        if "entities" in representations:
            entities = self._extract_entities(text)
            if entities:
                results["entities"] = self.encode(" ".join(entities))
            else:
                results["entities"] = results.get("original", self.encode(""))
                
        # 4. 三元组表征
        if "triples" in representations:
            triples = self._extract_triples(text)
            if triples:
                results["triples"] = self.encode(" ".join(triples))
            else:
                results["triples"] = results.get("original", self.encode(""))
                
        # 5. 摘要表征
        if "summary" in representations:
            summary = self._generate_summary(text)
            results["summary"] = self.encode(summary) if summary else results.get("original", self.encode(""))
            
        # 6. 问句表征 (query-oriented)
        if "query" in representations:
            results["query"] = self.encode(f"query: {text}", prompt_name="asymmetric")
            
        return results
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        # 使用简单的 TF-IDF 风格关键词提取
        # 实际应用中可使用更复杂的 NER 或关键词提取模型
        try:
            import jieba
            import jieba.analyse
            
            # 使用 TF-IDF 提取 top 5 关键词
            keywords = jieba.analyse.extract_tags(text, topK=5, withWeight=False)
            return keywords
        except ImportError:
            # 降级：使用简单的分词
            import jieba
            words = jieba.cut(text)
            return [w for w in words if len(w) >= 2][:5]
        except Exception:
            return []
            
    def _extract_entities(self, text: str) -> List[str]:
        """提取实体"""
        # 使用简单的规则提取实体
        # 实际应用中可使用 NER 模型
        try:
            import jieba
            import jieba.posseg as pseg
            
            words = pseg.cut(text)
            entities = []
            for word, flag in words:
                # 名词性实体 (n 系列) 和专有名词 (nr, ns, nt 等)
                if flag.startswith('n') or flag.startswith('nr') or flag.startswith('ns'):
                    if len(word) >= 2:
                        entities.append(word)
            return list(set(entities))[:5]
        except Exception:
            return []
            
    def _extract_triples(self, text: str) -> List[str]:
        """从文本中提取三元组"""
        # 简化版本：使用关系模式匹配
        # 实际应用中应使用专门的 RE 模型
        relation_patterns = [
            (r'(\w+)(?:是|为|等于)(\w+)', '是'),
            (r'(\w+)(?:位于|在)(\w+)', '位于'),
            (r'(\w+)(?:属于|归于)(\w+)', '属于'),
        ]
        
        triples = []
        for pattern, rel in relation_patterns:
            import re
            matches = re.findall(pattern, text)
            for match in matches:
                if len(match) >= 2:
                    triples.append(f"({match[0]} {rel} {match[1]})")
                    
        return triples[:3]
        
    def _generate_summary(self, text: str, max_len: int = 100) -> str:
        """生成摘要"""
        if len(text) <= max_len:
            return text
            
        # 简单截取策略
        # 实际应用中可使用摘要生成模型
        sentences = text.replace('。', '。').split('。')
        summary = sentences[0] if sentences else text[:max_len]
        
        if len(summary) > max_len:
            summary = summary[:max_len]
            
        return summary + "..."
        
    def get_embedding_dimension(self) -> int:
        """获取嵌入向量维度"""
        # Qwen3-Embedding-8B 输出 768 维向量
        return 768
        
    def encode_batch_parallel(
        self,
        texts: List[str],
        num_workers: int = 4,
        representation_type: str = "dense"
    ) -> np.ndarray:
        """
        并行批量编码
        
        Args:
            texts: 文本列表
            num_workers: 并行工作进程数
            representation_type: 表征类型
            
        Returns:
            numpy.ndarray: 嵌入向量数组
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        all_embeddings = []
        chunk_size = max(1, len(texts) // num_workers)
        chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
        
        def encode_chunk(chunk):
            return self.encode(chunk, representation_type=representation_type)
            
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(encode_chunk, chunk) for chunk in chunks]
            for future in as_completed(futures):
                all_embeddings.append(future.result())
                
        return np.vstack(all_embeddings)


class HybridEncoder:
    """
    混合编码器 - 结合稀疏和稠密向量
    """
    
    def __init__(self, dense_encoder: Optional[Qwen3EmbeddingEncoder] = None):
        self.dense_encoder = dense_encoder or Qwen3EmbeddingEncoder()
        
    def encode_hybrid(
        self,
        texts: Union[str, List[str]],
        return_sparse: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        混合编码：同时返回稠密和稀疏向量
        
        Args:
            texts: 文本或文本列表
            return_sparse: 是否返回稀疏向量
            
        Returns:
            Dict: 包含 dense 和可选的 sparse 向量
        """
        if isinstance(texts, str):
            texts = [texts]
            
        results = {
            "dense": self.dense_encoder.encode(texts)
        }
        
        if return_sparse:
            results["sparse"] = self._compute_sparse_vectors(texts)
            
        return results
        
    def _compute_sparse_vectors(self, texts: List[str]) -> np.ndarray:
        """计算稀疏向量 (BM25 风格)"""
        from collections import Counter
        import math
        
        # 简化的 BM25 实现
        def tokenize(text):
            try:
                import jieba
                return list(jieba.cut(text))
            except:
                return text.split()
                
        # 构建词表
        vocab = {}
        for text in texts:
            tokens = tokenize(text)
            for token in tokens:
                if token not in vocab:
                    vocab[token] = len(vocab)
                    
        # 计算 TF-IDF 风格稀疏向量
        N = len(texts)
        idf = {}
        
        for token, idx in vocab.items():
            df = sum(1 for text in texts if token in tokenize(text))
            idf[token] = math.log((N - df + 0.5) / (df + 0.5) + 1)
            
        sparse_vectors = []
        avg_dl = sum(len(tokenize(t)) for t in texts) / N
        k1, b = 1.5, 0.75
        
        for text in texts:
            tokens = tokenize(text)
            tf = Counter(tokens)
            vector = np.zeros(len(vocab))
            
            for token, freq in tf.items():
                if token in vocab:
                    idx = vocab[token]
                    # BM25 公式
                    tf_norm = (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * len(tokens) / avg_dl))
                    vector[idx] = tf_norm * idf.get(token, 0)
                    
            sparse_vectors.append(vector)
            
        return np.array(sparse_vectors)


class MultiRepresentationBuilder:
    """
    多表征构建器
    为每个文档生成多种向量表征
    """
    
    def __init__(self, encoder: Optional[Qwen3EmbeddingEncoder] = None):
        self.encoder = encoder or Qwen3EmbeddingEncoder()
        
    def build_representations(
        self,
        document: str,
        representations: Optional[List[str]] = None,
        chunk_size: int = 512,
        overlap: int = 50
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        为文档构建多表征
        
        Args:
            document: 完整文档
            representations: 要生成的表征类型
            chunk_size: 分块大小
            overlap: 块间重叠大小
            
        Returns:
            Dict: 包含不同层级的表征
        """
        if representations is None:
            representations = ["original", "keywords", "entities", "triples"]
            
        results = {
            "document_level": {},
            "chunk_level": {},
            "sentence_level": {}
        }
        
        # 1. 文档级别表征
        doc_repr = self.encoder.encode_multi_representation(document, representations)
        results["document_level"] = doc_repr
        
        # 2. 句子级别表征
        sentences = self._split_sentences(document)
        if sentences:
            sentence_reprs = self.encoder.encode_multi_representation(
                sentences,
                representations
            )
            results["sentence_level"] = sentence_reprs
            
        # 3. 分块级别表征
        chunks = self._chunk_text(document, chunk_size, overlap)
        if chunks:
            chunk_reprs = self.encoder.encode_multi_representation(
                chunks,
                representations
            )
            results["chunk_level"] = chunk_reprs
            
        return results
        
    def _split_sentences(self, text: str) -> List[str]:
        """分句"""
        # 简单分句
        import re
        sentences = re.split(r'[。！？\n]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
        
    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """文本分块"""
        if len(text) <= chunk_size:
            return [text]
            
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
            
        return chunks


def download_qwen3_embedding_model(save_path: Optional[str] = None) -> str:
    """
    下载 Qwen3-Embedding-8B 模型
    
    Args:
        save_path: 保存路径
        
    Returns:
        str: 模型保存路径
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    
    model_name = "Qwen/Qwen3-Embedding-8B"
    
    if save_path is None:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        save_path = os.path.join(project_root, "models", "Qwen3-Embedding-8B")
        
    os.makedirs(save_path, exist_ok=True)
    
    print(f"正在下载模型: {model_name}")
    print(f"保存路径: {save_path}")
    
    # 下载分词器
    print("下载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    tokenizer.save_pretrained(save_path)
    
    # 下载模型
    print("下载模型...")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    model.save_pretrained(save_path)
    
    print(f"模型下载完成: {save_path}")
    return save_path


if __name__ == "__main__":
    # 测试编码器
    encoder = Qwen3EmbeddingEncoder()
    
    # 测试单文本编码
    text = "知识图谱是一种用图来表达实体和它们之间关系的技术。"
    embedding = encoder.encode(text)
    print(f"单文本嵌入维度: {embedding.shape}")
    
    # 测试批量编码
    texts = [
        "人工智能是计算机科学的一个分支。",
        "机器学习是人工智能的一个子领域。",
        "深度学习是机器学习的一种方法。"
    ]
    embeddings = encoder.encode(texts)
    print(f"批量嵌入维度: {embeddings.shape}")
    
    # 测试多表征编码
    multi_repr = encoder.encode_multi_representation(text)
    print(f"多表征类型: {list(multi_repr.keys())}")
