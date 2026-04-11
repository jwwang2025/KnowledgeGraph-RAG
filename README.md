# KnowledgeGraph-RAG：知识图谱+RAG 双驱动大模型对话系统

## 🔗项目简介

本项目提出一种 "结构化知识图谱+非结构化文档库" 双检索驱动的检索增强生成（RAG）对话系统，核心目标是解决大模型在事实性问答中存在的 幻觉 问题，同时提升回答的可追溯性与上下文相关性。系统以知识图谱（结构化知识）和文档库（非结构化知识）为双重外部知识源，通过 RAG 技术将检索到的精准事实证据动态注入生成模型（ChatGLM-6B），最终实现 有依据、可验证、高准确的智能对话服务。

## 🎯 技术特性

1. **UIE 抽取**：高效完成实体、关系、属性的自动化抽取。
2. **SPN4RE**：提升关系三元组抽取的精度与效率。
3. **双源 RAG 检索增强**：核心技术支撑，融合知识图谱结构化检索与文档库非结构化检索。
4. **Adaptive-RAG 智能路由**：根据问题类型自适应选择检索策略和知识源。
5. **Self-RAG 结果评估**：评估检索结果相关性，智能过滤低质量内容。
6. **CoT 思维链推理**：支持 Zero-shot / Few-shot / Self-Consistency 多种推理模式。
7. **引用溯源（Citations）**：确保回答中关键细节与检索来源可追溯验证。

## 🛠️ 技术栈

### 后端
- **Python 3.x**
- **Flask**：Web框架
- **PyTorch**：深度学习框架
- **Transformers**：预训练模型库（ChatGLM-6B）
- **PaddlePaddle/PaddleNLP**：UIE模型支持
- **SPN4RE**：关系抽取模型

### 前端
- **Vue 3**：前端框架
- **Ant Design Vue**：UI组件库
- **ECharts**：数据可视化
- **D3.js**：知识图谱可视化
- **Vite**：构建工具

---

## 🧠 系统架构

### Adaptive-RAG + Self-RAG + CoT 三层架构

```
用户问题
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                    RAG 模块 (app/rag/)                      │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐      ┌─────────────────┐              │
│  │   QueryRouter   │ ───→ │ RetrievalDecider │              │
│  │   问题路由      │      │   检索决策       │              │
│  └─────────────────┘      └────────┬────────┘              │
│                                    │                        │
│                    ┌───────────────┼───────────────┐       │
│                    ▼               ▼               ▼       │
│              ┌──────────┐  ┌────────────┐  ┌──────────┐    │
│              │   Search │  │   Search   │  │  Search  │    │
│              │  (向量库) │  │  (知识图谱) │  │  (Wiki)  │    │
│              └────┬─────┘  └─────┬──────┘  └────┬─────┘    │
│                   │             │              │           │
│                   └─────────────┼──────────────┘           │
│                                 ▼                            │
│                      ┌─────────────────┐                     │
│                      │ ResultEvaluator │                     │
│                      │   结果评估      │                     │
│                      └────────┬────────┘                     │
│                               │                              │
│                               ▼                              │
│                      ┌─────────────────┐                   │
│                      │  CoTReasoner     │                   │
│                      │  思维链推理      │                   │
│                      └────────┬────────┘                   │
└──────────────────────────────┼───────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    Model 模块 (app/model/)                    │
│                      ChatGLM 模型生成                        │
└─────────────────────────────────────────────────────────────┘
```

### 流程对比

| 特性 | 传统 RAG | Adaptive-RAG + Self-RAG |
|------|----------|-------------------------|
| 检索触发 | 固定流程 | 自适应判断 |
| 知识源选择 | 全部检索 | 按需选择 |
| 结果质量 | 全部使用 | 智能过滤 |
| 检索决策 | 盲目检索 | 反思决策 |
| 推理能力 | 直接生成 | 思维链推理 |

---

## 🧩 CoT 思维链模块

### CoT 工作原理

```
问题 → 分解子问题 → 逐步推理 → 综合结论
```

### CoT 模式

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| **Zero-shot CoT** | "让我们一步步思考" 引导推理 | 通用推理 |
| **Few-shot CoT** | 基于示例学习推理模式 | 复杂概念 |
| **Self-Consistency** | 多路径推理取最优 | 需要准确性的问题 |

### 问题类型 → CoT 模式映射

| 问题类型 | CoT 模式 | 原因 |
|----------|----------|------|
| 比较类 | self_consistency | 需要多角度分析 |
| 解释类 | few_shot | 需要示例引导 |
| 分析类 | self_consistency | 需要深度推理 |
| 事实类 | zero_shot | 简单直接推理 |
| 闲聊 | direct | 无需推理 |

---

## 🔄 Self-RAG 多轮检索策略

### 架构概述

基于 Self-RAG 思想，实现两轮后处理优化排序结果：

```
多源检索结果 (知识图谱 + 向量库 + Wikipedia)
        │
        ▼
┌───────────────────────────────┐
│   第一轮: RRF 融合与去重       │
│   Reciprocal Rank Fusion      │
│   - 多源结果融合排序           │
│   - 智能去重处理              │
└───────────────────────────────┘
        │
        ▼
┌───────────────────────────────┐
│   第二轮: Cohere 语义重排序    │
│   Semantic Reranking          │
│   - 语义感知精细化排序        │
│   - 上下文相关性优化          │
└───────────────────────────────┘
        │
        ▼
   精排后的高质量结果
```

### RRF 融合算法

Reciprocal Rank Fusion (RRF) 公式：

$$\text{RRF}(d) = \sum_{s \in S} \frac{1}{k + \text{rank}_s(d)}$$

其中：
- \(d\): 文档/结果项
- \(S\): 检索源集合
- \(k\): 平滑因子 (默认60)
- \(\text{rank}_s(d)\): 文档在源 \(s\) 中的排名

**特点**：
- 无需学习权重参数
- 对各来源平等对待
- 排名越靠前的结果权重越高

### Cohere 语义重排序

使用 Cohere Rerank API 进行语义级别的精确排序：

**支持的模型**：
- `rerank-multilingual-v3.0` (推荐，支持多语言)
- `rerank-english-v2.0` (英文专用)

**本地备选方案**：
当 API 不可用时，使用本地相似度计算作为后备。

### 使用示例

```python
from app.rag import (
    AdaptiveRAGEngine,
    RRFusion,
    CohereReranker,
    fuse_results,
    quick_rerank
)

# 方式1: 通过引擎自动使用多轮检索
engine = AdaptiveRAGEngine(
    project_name="project_v1",
    enable_multi_round=True,           # 启用多轮检索
    cohere_api_key="your-api-key"     # Cohere API 密钥
)

context = engine.process("人工智能的发展历史")
print(f"RRF融合统计: {context.fusion_stats}")
print(f"Cohere重排统计: {context.rerank_stats}")

# 方式2: 直接使用 RRF 融合
fusion = RRFusion(k=60.0, enable_deduplication=True)
source_results = {
    'kg': [("人工智能", "属于", "计算机科学")],
    'vector': ["人工智能是研究开发..."],
    'wiki': {"summary": "人工智能（AI）是..."}
}
fusion_result = fusion.fuse("人工智能是什么", source_results)
print(f"融合后结果数: {fusion_result.total_output}")

# 方式3: 快速重排序
from app.rag import FusionItem, quick_rerank
candidates = [...]
reranked = quick_rerank("查询", candidates, api_key="your-key")
```

### 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `rrf_k` | 60.0 | RRF 平滑因子，越大排名差异越小 |
| `dedup_threshold` | 0.85 | 去重相似度阈值 |
| `cohere_model` | `rerank-multilingual-v3.0` | Cohere 重排序模型 |
| `final_top_k` | 10 | 最终返回结果数量 |
| `fusion_candidates` | 50 | 融合候选数量 |

---

## 📺系统展示

### 问答页面（无检索增强）
![alt text](proj-docs/QAPagenG.png)

### 问答页面（有检索增强）
![alt text](proj-docs/QAPagewG.png)

### 图谱页面
![alt text](proj-docs/graphPage.png)

---

## 📁 项目结构

```
KnowledgeGraph-RAG/
├── main.py                    # 主入口：知识图谱构建流程
├── config/                    # 配置文件
│   └── settings.py            # 项目配置
├── backend/                   # 后端服务与 API（Flask）
│   ├── main.py                # 后端服务入口
│   └── app/                   # Flask 应用
│       ├── __init__.py       # 应用初始化
│       ├── logger.py          # 日志工具
│       ├── views/             # API 路由
│       │   ├── chat.py        # 对话接口
│       │   └── graph.py       # 图谱查询接口
│       ├── rag/               # RAG 核心模块
│       │   ├── __init__.py   # 模块导出
│       │   ├── query_router.py         # Adaptive-RAG：问题路由
│       │   ├── retrieval_decider.py    # Adaptive-RAG：检索决策
│       │   ├── result_evaluator.py    # Self-RAG：结果评估
│       │   ├── cot_reasoner.py        # CoT：思维链推理
│       │   ├── citation.py            # Citation：引用溯源机制
│       │   └── adaptive_rag_engine.py  # RAG 核心引擎
│       ├── search/             # 检索适配器模块
│       │   ├── __init__.py   # 模块导出
│       │   ├── vector_searcher.py    # 向量数据库检索
│       │   ├── wiki_searcher.py      # Wikipedia 搜索
│       │   └── image_searcher.py      # 图像搜索
│       ├── nlp/                # NLP 组件模块
│       │   ├── __init__.py   # 模块导出
│       │   └── ner.py               # 命名实体识别
│       ├── model/              # 模型调用模块
│       │   ├── __init__.py   # 模块导出
│       │   └── chatglm.py           # ChatGLM 调用
│       └── kg/                 # 知识图谱组件
│           ├── __init__.py   # 模块导出
│           └── graph_utils.py       # 图谱工具函数
├── modules/                   # 核心模块（知识图谱构建、模型训练等）
│   ├── knowledge_graph_builder.py
│   ├── model_trainer.py
│   └── prepare/               # 数据预处理
├── data/                      # 数据目录（原始数据与项目迭代数据）
│   └── project_v1/
│       └── history/           # 检查点历史记录
├── frontend/                  # 前端 Vue 应用（可视化界面）
│   ├── index.html
│   └── src/
├── models/                    # 模型文件与预训练权重
│   ├── chatglm-6b/            # ChatGLM-6B 模型权重与实现
│   ├── uie-base/              # UIE 模型与权重（信息抽取）
│   └── bert-base-chinese/     # BERT 中文模型及词表
├── utils/                     # 工具脚本
├── README.md
├── requirements.txt
└── package-lock.json
```

---

## 🏗️ 模块组织说明

项目采用分层模块化设计，`app/` 目录结构如下：

### RAG 模块 (`app/rag/`)

检索增强生成核心组件，负责问题分析、检索决策、结果评估和引用溯源：

| 文件 | 说明 | 核心类/函数 |
|------|------|-------------|
| `query_router.py` | 问题路由器 | `QueryRouter`, `QuestionType`, `RetrievalPlan` |
| `retrieval_decider.py` | 检索决策器 | `RetrievalDecider`, `MultiSourceRetrievalResult` |
| `result_evaluator.py` | 结果评估器 | `ResultEvaluator`, `EvaluationReport` |
| `cot_reasoner.py` | 思维链推理 | `CoTReasoner`, `CoTMode`, `ReasoningChain` |
| `citation.py` | 引用溯源机制 | `Citation`, `CitationSet`, `CitationGenerator`, `CitationEmbedder` |
| `fusion.py` | RRF 融合算法 | `RRFusion`, `FusionResult`, `MultiRoundRetrieval` |
| `reranker.py` | Cohere 语义重排序 | `CohereReranker`, `RerankReport`, `SelfRAGRefiner` |
| `adaptive_rag_engine.py` | RAG 核心引擎 | `AdaptiveRAGEngine`, `RetrievalContext` |

### 检索模块 (`app/search/`)

多种检索源适配器，为 RAG 提供数据检索能力：

| 文件 | 说明 | 核心类 |
|------|------|--------|
| `vector_searcher.py` | 向量数据库检索 | `VectorSearcher` (ChromaDB) |
| `wiki_searcher.py` | Wikipedia 搜索 | `WikiSearcher` |
| `image_searcher.py` | 图像搜索 | `ImageSearcher` |

### NLP 模块 (`app/nlp/`)

自然语言处理组件：

| 文件 | 说明 | 核心类 |
|------|------|--------|
| `ner.py` | 命名实体识别 | `Ner` (PaddleNLP UIE) |

### 模型模块 (`app/model/`)

大模型调用和配置：

| 文件 | 说明 | 核心函数 |
|------|------|----------|
| `chatglm.py` | ChatGLM 调用 | `start_model`, `stream_predict`, `init_rag_engine` |

### 知识图谱模块 (`app/kg/`)

知识图谱查询和处理工具：

| 文件 | 说明 | 核心函数 |
|------|------|----------|
| `graph_utils.py` | 图谱工具 | `search_node_item`, `convert_graph_to_triples` |

---

## 🗝️ 模块使用说明

### 1. 核心引擎 - AdaptiveRAGEngine

融合 Adaptive-RAG + Self-RAG + CoT 的智能问答引擎：

```python
from app.rag import AdaptiveRAGEngine

engine = AdaptiveRAGEngine(
    project_name="project_v1",
    enable_evaluation=True,     # 启用 Self-RAG 评估
    enable_iteration=False,     # 可选：启用迭代检索
    enable_cot=True,           # 启用 CoT 思维链
    default_cot_mode="zero_shot"  # 默认 Zero-shot CoT 模式
)

context = engine.process("人工智能的发展历史")
print(f"问题类型: {context.retrieval_plan.question_type}")
print(f"检索结果: {context.retrieval_result.total_sources_used} 个知识源")
print(f"评估结果: {context.evaluation_report.overall_relevance}")
print(f"CoT 推理: {context.reasoning_chain}")
print(f"最终 Prompt: {context.assembled_prompt[:100]}...")
```

### 2. 问题路由 - QueryRouter

分析用户问题，决定检索策略：

```python
from app.rag import QueryRouter

router = QueryRouter()
plan = router.route("什么是人工智能？")

print(f"问题类型: {plan.question_type}")      # QuestionType.DEFINITION
print(f"知识源: {plan.priority_sources}")      # ['wiki', 'vector']
print(f"需要检索: {plan.need_retrieval}")     # True
print(f"启用 CoT: {plan.use_cot}")            # True
print(f"CoT 模式: {plan.cot_mode}")           # 'zero_shot'
```

### 3. 检索决策 - RetrievalDecider

执行多源自适应检索：

```python
from app.rag import RetrievalDecider

decider = RetrievalDecider(project_name="project_v1")
result = decider.retrieve("人工智能是谁发明的？", plan)

print(f"三元组: {result.triples}")
print(f"文档: {result.documents}")
print(f"Wikipedia: {result.wiki_summary}")
```

### 4. 结果评估 - ResultEvaluator

评估检索结果的相关性（Self-RAG）：

```python
from app.rag import ResultEvaluator

evaluator = ResultEvaluator()
report = evaluator.evaluate(query, qtype, triples, docs, wiki)

print(f"整体相关性: {report.overall_relevance}")  # HIGH/MEDIUM/LOW
print(f"决策: {report.action}")                    # USE_RETRIEVAL/GENERATE_DIRECT
print(f"高质量结果数: {report.high_relevant_count}")
```

### 5. 思维链推理 - CoTReasoner

提供多种思维链推理模式：

```python
from app.rag import CoTReasoner, CoTMode

# Zero-shot CoT
reasoner = CoTReasoner(mode=CoTMode.ZERO_SHOT)
prompt = reasoner.build_cot_prompt("什么是人工智能？", knowledge)
print(prompt)

# Few-shot CoT
reasoner = CoTReasoner(mode=CoTMode.FEW_SHOT)
prompt = reasoner.build_cot_prompt("AI和机器学习的区别？", knowledge)

# Self-Consistency
reasoner = CoTReasoner(mode=CoTMode.SELF_CONSISTENCY)
chain = reasoner.reason("复杂分析问题", knowledge, depth=2)
print(f"推理步骤: {len(chain.steps)}")
print(f"一致性得分: {chain.consistency_score}")
```

### 6. 引用溯源 (Citation)

引用溯源机制确保回答中关键细节与检索来源可追溯验证：

```python
from app.rag import Citation, CitationSet, CitationGenerator, CitationEmbedder

# 从检索结果生成引用
triples = [("人工智能", "属于", "计算机科学")]
citations = CitationGenerator.generate_triple_citations(triples, query="什么是人工智能")

# 处理文档引用
docs = ["人工智能是计算机科学的一个分支..."]
doc_citations = CitationGenerator.generate_document_citations(docs, query="人工智能定义")

# 处理 Wikipedia 引用
wiki_result = {"title": "人工智能", "summary": "人工智能（AI）是..."}
wiki_citation = CitationGenerator.generate_wiki_citation(wiki_result, query="人工智能")

# 合并引用集合
citation_set = CitationSet()
citation_set.add_batch(citations + doc_citations + [wiki_citation])
print(f"总引用数: {len(citation_set.citations)}")

# 按相关性排序
sorted_citations = citation_set.get_sorted_by_relevance()

# 格式化输出
for citation in sorted_citations:
    print(f"来源: {citation.source_name}")
    print(f"片段: {citation.excerpt}")
    print(f"相关度: {citation.relevance_score}")

# 使用嵌入器添加引用标记
embedder = CitationEmbedder(format_type="superscript")
formatted_citations = embedder.format_citations_for_api(citation_set.citations)
```

### 8. Self-RAG 多轮检索 (RRF + Cohere)

基于 Self-RAG 的多轮检索策略，实现两轮后处理优化排序：

```python
from app.rag import (
    AdaptiveRAGEngine,  # 核心引擎（自动使用多轮检索）
    RRFusion,          # RRF 融合器
    CohereReranker,    # Cohere 重排序器
    FusionItem,        # 融合项
    fuse_results,      # 便捷融合函数
    quick_rerank       # 便捷重排序函数
)

# 方式1: 通过引擎自动使用多轮检索（推荐）
engine = AdaptiveRAGEngine(
    project_name="project_v1",
    enable_multi_round=True,           # 启用多轮检索
    cohere_api_key="your-cohere-key"   # Cohere API 密钥
)

context = engine.process("人工智能的发展历史")
print(f"RRF融合统计: {context.fusion_stats}")
print(f"Cohere重排统计: {context.rerank_stats}")

# 获取多轮检索摘要
multi_round_summary = context.get_multi_round_summary()
print(f"融合输入: {multi_round_summary['fusion']['input_count']}")
print(f"融合输出: {multi_round_summary['fusion']['output_count']}")

# 方式2: 直接使用 RRF 融合
fusion = RRFusion(k=60.0, enable_deduplication=True)
source_results = {
    'kg': [("人工智能", "属于", "计算机科学")],
    'vector': ["人工智能是研究开发..."],
    'wiki': {"summary": "人工智能（AI）是..."}
}
fusion_result = fusion.fuse("人工智能是什么", source_results)
print(f"融合后结果数: {fusion_result.total_output}")
print(f"去重数量: {fusion_result.duplicates_removed}")

# 获取融合后的结果
for item in fusion_result.get_top_k(5):
    print(f"内容: {item.content}, RRF得分: {item.rrf_score:.3f}")

# 方式3: 使用 Cohere 重排序器
reranker = CohereReranker(
    api_key="your-cohere-key",
    model="rerank-multilingual-v3.0",
    enable_local_fallback=True  # API不可用时使用本地方案
)

# 重排序
reranked_items = reranker.rerank(query, candidates, top_k=10)

# 重排序并获取报告
reranked_items, report = reranker.rerank_with_report(query, candidates, top_k=10)
print(f"重排序模型: {report.model_used.value}")
print(f"平均得分: {report.avg_score:.3f}")
print(f"位置变化: {report.position_changes}")

# 方式4: 快速便捷函数
reranked = quick_rerank("查询", candidates, api_key="your-key")

# 多轮检索控制器（高级用法）
from app.rag import MultiRoundRetrieval
controller = MultiRoundRetrieval(k=60.0, enable_dedup=True)
controller.set_reranker(reranker)

# 执行完整流程
result = controller.execute_full_pipeline(
    query="人工智能",
    kg_results=[("AI", "是", "技术")],
    vector_results=["AI相关文档..."],
    wiki_result={"summary": "AI是..."},
    final_top_k=10
)
print(f"总候选数: {result['total_candidates']}")
print(f"最终数量: {result['final_count']}")
```

### 9. 统一导入

通过各模块的 `__init__.py` 提供统一导出，推荐使用子模块路径导入：

```python
# 推荐：使用子模块路径
from app.rag import (
    AdaptiveRAGEngine, QueryRouter, RetrievalDecider, ResultEvaluator, CoTReasoner,
    Citation, CitationSet, CitationContext, CitationGenerator, CitationEmbedder,
    RRFusion, FusionItem, CohereReranker, SelfRAGRefiner
)
from app.search import VectorSearcher, WikiSearcher, ImageSearcher
from app.nlp import Ner
from app.model import start_model, stream_predict
from app.kg import search_node_item, convert_graph_to_triples

# 兼容：使用统一导出（通过 app.utils）
from app.utils import AdaptiveRAGEngine, QueryRouter, VectorSearcher
```

---

## 🚀 快速开始

### 第一步：环境准备

#### 1.1 系统要求

- Python 3.8+
- Node.js 16+

#### 1.2 安装Python依赖

```bash
conda create -n KnowledgeGraph-RAG python=3.11
conda activate KnowledgeGraph-RAG
pip install -r requirements.txt
```

#### 1.3 安装前端依赖（可选）

```bash
# 进入前端目录
cd frontend

# 安装依赖
npm install

# 返回项目根目录
cd ..
```

---

### 第二步：准备数据

#### 2.1 准备原始文本数据

确保 `data/raw_data/raw_data.txt` 文件存在，包含待处理的原始文本数据。

**数据格式要求：**
- 纯文本格式（.txt）
- 每段文本建议包含完整的语义信息

---

### 第三步：构建知识图谱

#### 3.1 首次运行（从零开始构建）

```bash
# 在项目根目录执行
python main.py --project project_v1 --gpu 0
```

**参数说明：**
- `--project project_v1`：项目名称，决定数据存储路径（默认：`project_v1`）
- `--gpu 0`：指定使用的GPU ID（根据实际情况修改，CPU模式可省略此参数）

**运行过程：**
1. 从原始文本构建基础知识图谱
2. 进行迭代优化，自动扩展知识图谱
3. 当扩展比率低于阈值时自动停止迭代
4. 每个迭代版本会自动保存检查点

#### 3.2 从检查点恢复运行

如果之前运行中断，可以从检查点恢复：

```bash
# 查看可用的检查点文件
# Windows PowerShell
dir data\project_v1\history\
# Linux/Mac
ls data/project_v1/history/

# 从检查点恢复
python main.py --project project_v1 --resume data/project_v1/history/20230327-001537_iter_v1.json --gpu 0
```

**参数说明：**
- `--resume <检查点路径>`：从指定检查点恢复运行

---

### 第四步：启动后端服务

#### 4.1 配置环境变量

在项目根目录创建 `.env` 文件，配置服务器参数。

#### 4.2 启动Flask服务

```bash
# 在项目根目录执行
cd backend
python main.py
```

**服务启动后：**
- 后端API服务默认运行在 `http://localhost:5000`
- 首次启动会自动加载ChatGLM-6B模型
- 确保已构建知识图谱，否则对话功能可能无法正常工作

**API端点：**
- `/api/chat`：对话接口
- `/api/graph`：知识图谱查询接口

---

### 第五步：启动前端应用

#### 5.1 启动前端开发服务器

```bash
# 进入前端目录
cd chat-kg

# 启动服务器
npm run server
```

---

## 📄 许可证

本项目仅供学习和研究使用。



# 1. 安装新依赖
pip install chromadb sentence-transformers

# 2. 构建向量索引
python main.py --project project_v1 --build-vector-index

# 或者单独构建某个来源
python modules/vector_indexer.py --project project_v1 --source raw
python modules/vector_indexer.py --project project_v1 --source graph
python modules/vector_indexer.py --project project_v1 --source triples

# 3. 启动后端服务（向量检索自动生效）
cd server && python main.py
