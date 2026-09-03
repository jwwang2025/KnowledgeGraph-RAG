"""Qwen3-Embedding-8B 向量化编码器，支持多表征与混合编码。"""
import os
import re
import math
import torch
import numpy as np
from typing import List, Dict, Optional, Union
from collections import Counter

PROMPT_TEMPLATES = {
    "retrieve.query": "query: {text}",
    "retrieve.passage": "passage: {text}",
    "embeddings": "{text}",
    "symmetric": "{text}",
    "asymmetric": "query: {text}",
}

# 模块级模型缓存：同一 (模型路径, 精度) 只加载一次，多实例共享 model/tokenizer
_MODEL_CACHE: Dict[tuple, tuple] = {}


def _default_model_path() -> str:
    """默认本地模型路径"""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    return os.path.join(project_root, "models", "Qwen3-Embedding-8B")


def split_sentences(text: str, pattern: str = r'[。！？\n]+') -> List[str]:
    """按标点分句"""
    return [s.strip() for s in re.split(pattern, text) if s.strip()]


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """滑动窗口分块"""
    if len(text) <= chunk_size:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks


class Qwen3EmbeddingEncoder:
    """Qwen3-Embedding-8B 编码器：模型/tokenizer 延迟加载并全局缓存，仅实例化一次。"""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-8B",
        local_model_path: Optional[str] = None,
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
        batch_size: int = 8,
        max_length: int = 8192
    ):
        self.model_name = model_name
        self.local_model_path = local_model_path or _default_model_path()
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._tokenizer = None

    @property
    def model(self):
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._load_model()
        return self._tokenizer

    def _load_model(self):
        """加载模型和分词器（带模块级缓存，避免重复加载）"""
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

        model_path = self.local_model_path
        if not os.path.exists(model_path):
            print(f"[警告] 本地模型不存在: {model_path}，将从 HuggingFace 下载")
            model_path = self.model_name

        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        cache_key = (model_path, str(torch_dtype), self.device)
        if cache_key in _MODEL_CACHE:
            self._model, self._tokenizer = _MODEL_CACHE[cache_key]
            return

        print(f"[信息] 加载 Qwen3-Embedding-8B 模型...")
        self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            trust_remote_code=True,
            device_map="auto" if self.device == "cuda" else None,
            torch_dtype=torch_dtype
        )
        if self.device == "cpu":
            self._model = self._model.to("cpu")
        self._model.eval()
        print(f"[完成] 模型加载成功，设备: {self.device}")

        _MODEL_CACHE[cache_key] = (self._model, self._tokenizer)

    def _encode_batch(self, batch_texts: List[str], prompt: Optional[str], prompt_name: Optional[str]) -> np.ndarray:
        """编码单个 batch"""
        if prompt:
            inputs = [prompt.format(text=t) for t in batch_texts]
        elif prompt_name:
            template = PROMPT_TEMPLATES.get(prompt_name, "{text}")
            inputs = [template.format(text=t) for t in batch_texts]
        else:
            inputs = batch_texts

        encoded = self.tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        if self.device != "cpu":
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = self.model(**encoded)
            embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
            if self.normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy()

    def encode(
        self,
        texts: Union[str, List[str]],
        representation_type: str = "dense",
        prompt_name: Optional[str] = None,
        prompt: Optional[str] = None
    ) -> np.ndarray:
        """编码文本为向量（自动分批处理）"""
        if isinstance(texts, str):
            texts = [texts]

        batches = [
            self._encode_batch(texts[i:i + self.batch_size], prompt, prompt_name)
            for i in range(0, len(texts), self.batch_size)
        ]
        return np.vstack(batches)

    def encode_multi_representation(
        self,
        text: str,
        representations: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """多表征编码：original/keywords/entities/triples/summary/query"""
        if representations is None:
            representations = ["original", "keywords", "entities", "triples", "summary"]

        results: Dict[str, np.ndarray] = {}

        # 原始文本表征（其余表征在无内容时回退到它）
        if "original" in representations:
            results["original"] = self.encode(text)

        def fallback() -> np.ndarray:
            return results.get("original", self.encode(""))

        extractors = {
            "keywords": self._extract_keywords,
            "entities": self._extract_entities,
            "triples": self._extract_triples,
        }
        for name, extractor in extractors.items():
            if name in representations:
                items = extractor(text)
                results[name] = self.encode(" ".join(items)) if items else fallback()

        if "summary" in representations:
            summary = self._generate_summary(text)
            results["summary"] = self.encode(summary) if summary else fallback()

        if "query" in representations:
            results["query"] = self.encode(f"query: {text}", prompt_name="asymmetric")

        return results

    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词（TF-IDF，降级为分词）"""
        try:
            import jieba.analyse
            return jieba.analyse.extract_tags(text, topK=5, withWeight=False)
        except ImportError:
            import jieba
            return [w for w in jieba.cut(text) if len(w) >= 2][:5]
        except Exception:
            return []

    def _extract_entities(self, text: str) -> List[str]:
        """基于词性的简易实体提取"""
        try:
            import jieba.posseg as pseg
            entities = [
                word for word, flag in pseg.cut(text)
                if flag.startswith('n') and len(word) >= 2
            ]
            return list(set(entities))[:5]
        except Exception:
            return []

    _TRIPLE_PATTERNS = [
        (r'(\w+)(?:是|为|等于)(\w+)', '是'),
        (r'(\w+)(?:位于|在)(\w+)', '位于'),
        (r'(\w+)(?:属于|归于)(\w+)', '属于'),
    ]

    def _extract_triples(self, text: str) -> List[str]:
        """基于关系模式的简易三元组提取"""
        triples = []
        for pattern, rel in self._TRIPLE_PATTERNS:
            for match in re.findall(pattern, text):
                if len(match) >= 2:
                    triples.append(f"({match[0]} {rel} {match[1]})")
        return triples[:3]

    def _generate_summary(self, text: str, max_len: int = 100) -> str:
        """简单截取式摘要（取第一个句号前的内容）"""
        if len(text) <= max_len:
            return text
        return text.split('。')[0][:max_len] + "..."

    def get_embedding_dimension(self) -> int:
        """Qwen3-Embedding-8B 输出 768 维向量"""
        return 768

    def encode_batch_parallel(
        self,
        texts: List[str],
        num_workers: int = 4,
        representation_type: str = "dense"
    ) -> np.ndarray:
        """并行批量编码"""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        chunk_size = max(1, len(texts) // num_workers)
        chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self.encode, chunk, representation_type) for chunk in chunks]
            all_embeddings = [future.result() for future in as_completed(futures)]

        return np.vstack(all_embeddings)


class HybridEncoder:
    """混合编码器：稠密向量 + BM25 风格稀疏向量"""

    def __init__(self, dense_encoder: Optional[Qwen3EmbeddingEncoder] = None):
        self.dense_encoder = dense_encoder or Qwen3EmbeddingEncoder()

    def encode_hybrid(
        self,
        texts: Union[str, List[str]],
        return_sparse: bool = True
    ) -> Dict[str, np.ndarray]:
        """混合编码：同时返回稠密和可选的稀疏向量"""
        if isinstance(texts, str):
            texts = [texts]

        results = {"dense": self.dense_encoder.encode(texts)}
        if return_sparse:
            results["sparse"] = self._compute_sparse_vectors(texts)
        return results

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        try:
            import jieba
            return list(jieba.cut(text))
        except Exception:
            return text.split()

    def _compute_sparse_vectors(self, texts: List[str]) -> np.ndarray:
        """计算 BM25 风格稀疏向量"""
        vocab: Dict[str, int] = {}
        tokenized = [self._tokenize(text) for text in texts]
        for tokens in tokenized:
            for token in tokens:
                if token not in vocab:
                    vocab[token] = len(vocab)

        N = len(texts)
        idf = {}
        for token in vocab:
            df = sum(1 for tokens in tokenized if token in tokens)
            idf[token] = math.log((N - df + 0.5) / (df + 0.5) + 1)

        avg_dl = sum(len(tokens) for tokens in tokenized) / N
        k1, b = 1.5, 0.75

        sparse_vectors = []
        for tokens in tokenized:
            vector = np.zeros(len(vocab))
            for token, freq in Counter(tokens).items():
                if token in vocab:
                    tf_norm = (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * len(tokens) / avg_dl))
                    vector[vocab[token]] = tf_norm * idf.get(token, 0)
            sparse_vectors.append(vector)

        return np.array(sparse_vectors)


class MultiRepresentationBuilder:
    """多表征构建器：为文档生成 document/chunk/sentence 三级表征"""

    def __init__(self, encoder: Optional[Qwen3EmbeddingEncoder] = None):
        self.encoder = encoder or Qwen3EmbeddingEncoder()

    def build_representations(
        self,
        document: str,
        representations: Optional[List[str]] = None,
        chunk_size: int = 512,
        overlap: int = 50
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """为文档构建多级多表征"""
        if representations is None:
            representations = ["original", "keywords", "entities", "triples"]

        results = {
            "document_level": self.encoder.encode_multi_representation(document, representations),
            "chunk_level": {},
            "sentence_level": {},
        }

        sentences = split_sentences(document)
        if sentences:
            results["sentence_level"] = self.encoder.encode_multi_representation(sentences, representations)

        chunks = chunk_text(document, chunk_size, overlap)
        if chunks:
            results["chunk_level"] = self.encoder.encode_multi_representation(chunks, representations)

        return results


def download_qwen3_embedding_model(save_path: Optional[str] = None) -> str:
    """下载 Qwen3-Embedding-8B 模型到本地"""
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    model_name = "Qwen/Qwen3-Embedding-8B"
    if save_path is None:
        save_path = _default_model_path()

    os.makedirs(save_path, exist_ok=True)
    print(f"正在下载模型: {model_name}")
    print(f"保存路径: {save_path}")

    print("下载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.save_pretrained(save_path)

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
