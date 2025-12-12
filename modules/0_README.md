# Modules 目录说明文档

本文档详细说明了 `modules` 文件夹的目录结构以及各个文件和文件夹的作用。

## 📁 目录结构

```
modules/
├── 0_README.md                    # 本说明文档
├── knowledge_graph_builder.py     # 知识图谱构建器主类
├── model_trainer.py               # SPN4RE 模型训练器
├── prepare/                       # 数据预处理模块
│   ├── __init__.py
│   ├── cprint.py                  # 彩色打印工具
│   ├── filter.py                  # 自动过滤错误三元组
│   ├── preprocess.py              # 文本清洗和句子分割
│   ├── process.py                 # UIE 关系抽取执行
│   └── utils.py                   # 知识图谱精炼工具函数
├── SPN4RE/                        # SPN4RE 关系抽取模型实现
│   ├── main.py                    # 模型训练主入口
│   ├── README.md                  # SPN4RE 使用说明
│   ├── models/                    # 模型定义
│   │   ├── __init__.py
│   │   ├── matcher.py
│   │   ├── seq_encoder.py
│   │   ├── set_criterion.py
│   │   ├── set_decoder.py
│   │   └── setpred4RE.py
│   ├── trainer/                   # 训练器
│   │   ├── __init__.py
│   │   └── trainer.py
│   ├── utils/                     # 工具函数
│   │   ├── __init__.py
│   │   ├── alphabet.py
│   │   ├── average_meter.py
│   │   ├── data.py
│   │   ├── functions.py
│   │   └── metric.py
│   ├── data/                      # 数据目录
│   │   ├── generated_data/        # 生成的数据
│   │   ├── NYT/                   # NYT 数据集
│   │   ├── WebNLG/                # WebNLG 数据集
│   │   └── zhijian_data_v*/       # 自定义数据集
│   └── bert_pretrain/             # BERT 预训练模型配置
├── fewshot_model/                 # 少样本学习模型
│   ├── preprocess.py
│   ├── process.py
│   └── run_fewshot.py
└── Uie-finetune/                  # UIE 模型微调相关
    ├── annotation/                # 标注数据
    └── deploy/                    # 部署相关代码
```

## 📄 文件详细说明

### 核心文件

#### `knowledge_graph_builder.py`
**作用**: 知识图谱构建器主类，负责整个知识图谱构建的完整流程。

**主要功能**:
- `get_base_kg_from_txt()`: 从原始文本文件生成基础知识图谱
  - 使用 UIE 进行关系抽取
  - 自动过滤错误三元组
  - 人工筛选和精炼知识图谱
- `run_iteration()`: 运行一次迭代训练
  - 训练 SPN4RE 模型
  - 对齐预测结果
  - 扩展知识图谱
- `extend_ratio()`: 计算知识图谱扩展比例，用于判断是否收敛
- `save()` / `load()`: 保存和加载训练状态，支持断点续训

#### `model_trainer.py`
**作用**: SPN4RE 模型训练器，封装了模型训练、测试和结果处理的全流程。

**主要功能**:
- `split_data()`: 将数据集按 5:2:3 比例切分为训练集、验证集和测试集
- `generate_running_cmd()`: 生成 SPN4RE 训练命令
- `train_and_test()`: 执行模型训练和测试
- `relation_align()`: 将预测结果与测试集对齐，去除重复三元组
- `refine_and_extend()`: 精炼知识图谱并与原始数据合并

### prepare/ 数据预处理模块

#### `process.py`
**作用**: 使用 PaddleNLP 的 UIE (Universal Information Extraction) 模型进行关系抽取。

**主要功能**:
- `uie_execute()`: 批量执行关系抽取，将文本转换为 SPN 格式的三元组数据
- `rel_json()`: 将 UIE 抽取结果转换为标准三元组格式

#### `preprocess.py`
**作用**: 文本预处理，包括清洗和句子分割。

**主要功能**:
- `clean_to_sentence()`: 清洗文本，去除特殊字符，繁体转简体
- `process_text()`: 将文本按句子分割，并控制每行最大长度（默认 480 字符）

#### `filter.py`
**作用**: 自动过滤错误的三元组。

**主要功能**:
- `auto_filter()`: 使用 BERT Tokenizer 验证实体是否在句子中存在
  - 检查实体长度（不超过 15 个 token）
  - 验证实体在句子中的位置
  - 过滤掉无效的三元组

#### `utils.py`
**作用**: 知识图谱精炼工具函数。

**主要功能**:
- `refine_knowledge_graph()`: 人工筛选知识图谱
  - 支持交互式筛选（逐条确认三元组是否正确）
  - 支持快速模式（fast_mode=True，直接保存）
  - 支持断点续筛

#### `cprint.py`
**作用**: 彩色打印工具，提供不同颜色的终端输出。

**主要功能**:
- 提供 `red()`, `green()`, `yellow()`, `blue()`, `purple()`, `cyan()`, `white()` 等函数
- 用于美化控制台输出，提高代码可读性

### SPN4RE/ 关系抽取模型

#### `main.py`
**作用**: SPN4RE 模型训练的主入口文件。

**主要功能**:
- 解析命令行参数（数据集路径、模型参数、训练超参数等）
- 构建数据集
- 初始化 SetPred4RE 模型
- 执行训练流程

**SPN4RE 简介**: 
- Set Prediction Networks for Relation Extraction
- 基于集合预测网络的关系抽取模型
- 支持联合实体和关系抽取
- 使用 BERT 作为编码器，Transformer 解码器生成三元组集合

#### `models/`
**作用**: 模型定义目录。

- `setpred4RE.py`: SetPred4RE 主模型类
- `seq_encoder.py`: 序列编码器（BERT）
- `set_decoder.py`: 集合解码器（Transformer）
- `matcher.py`: 匹配器（用于计算损失）
- `set_criterion.py`: 损失函数定义

#### `trainer/trainer.py`
**作用**: 模型训练器，负责训练循环、验证和测试。

#### `utils/`
**作用**: 工具函数目录。

- `data.py`: 数据加载和预处理
- `alphabet.py`: 关系标签字典管理
- `metric.py`: 评估指标计算
- `functions.py`: 通用工具函数
- `average_meter.py`: 平均值计算器

### fewshot_model/ 少样本学习模型

**作用**: 少样本学习相关代码，用于在数据量较少的情况下进行关系抽取。

**文件**:
- `preprocess.py`: 少样本数据预处理
- `process.py`: 少样本处理逻辑
- `run_fewshot.py`: 少样本模型运行入口

### Uie-finetune/ UIE 微调

**作用**: UIE 模型微调和部署相关代码。

**目录**:
- `annotation/`: 标注数据目录，包含不同版本的标注数据
- `deploy/`: 部署相关代码
  - `python/`: Python 推理代码（CPU/GPU）
  - `serving/`: 服务化部署代码

## 🔄 工作流程

1. **数据预处理** (`prepare/`)
   - 文本清洗和分割 (`preprocess.py`)
   - UIE 关系抽取 (`process.py`)
   - 自动过滤错误三元组 (`filter.py`)

2. **基础知识图谱构建** (`knowledge_graph_builder.py`)
   - 调用预处理模块生成基础三元组
   - 人工筛选精炼 (`utils.py`)

3. **迭代训练** (`knowledge_graph_builder.py` + `model_trainer.py`)
   - 使用 SPN4RE 模型训练 (`SPN4RE/main.py`)
   - 预测新三元组
   - 对齐和合并结果

4. **知识图谱扩展**
   - 计算扩展比例
   - 判断是否收敛
   - 保存最终知识图谱

## 📝 注意事项

1. **数据格式**: 所有数据采用 SPN 格式，每行一个 JSON 对象，包含：
   - `id`: 句子 ID
   - `sentText`: 句子文本
   - `relationMentions`: 三元组列表，每个三元组包含 `em1Text`, `em2Text`, `label`

2. **模型依赖**: 
   - SPN4RE 需要 BERT 预训练模型（如 `bert-base-chinese`）
   - UIE 需要 PaddleNLP 环境

3. **GPU 配置**: 训练过程需要指定 GPU 设备，通过 `--gpu` 或 `--visible_gpu` 参数设置

4. **断点续训**: 支持保存和加载训练状态，可以中断后继续训练

## 🔗 相关资源

- SPN4RE 论文: [Joint Entity and Relation Extraction with Set Prediction Networks](https://arxiv.org/abs/2011.01675)
- PaddleNLP UIE: [Universal Information Extraction](https://github.com/PaddlePaddle/PaddleNLP)

