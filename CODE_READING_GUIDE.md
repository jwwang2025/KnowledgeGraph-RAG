# KnowledgeGraph-RAG 项目代码阅读指南

## 项目概述

这是一个**基于知识图谱和知识库的大模型对话系统**，主要功能包括：
1. 从原始文本构建知识图谱
2. 使用迭代方法扩展和优化知识图谱
3. 提供基于知识图谱的对话服务
4. 提供知识图谱可视化前端

---

## 📁 项目结构概览

```
KnowledgeGraph-RAG/
├── main.py                    # 主入口：知识图谱构建流程
├── server/                    # Flask后端服务
│   ├── main.py               # 服务启动入口
│   └── app/                   # Flask应用
│       ├── views/            # API路由（chat.py, graph.py）
│       └── utils/            # 工具函数（chat_glm.py, graph_utils.py等）
├── modules/                   # 核心模块
│   ├── knowledge_graph_builder.py  # 知识图谱构建器（核心类）
│   ├── model_trainer.py      # 模型训练器
│   ├── prepare/              # 数据预处理模块
│   ├── SPN4RE/               # 关系抽取模型（Set Prediction Network）
│   └── Uie-finetune/         # UIE模型微调相关
├── data/                      # 数据目录
│   ├── raw_data.txt          # 原始文本数据
│   └── project_v1/           # 项目数据（知识图谱、迭代结果等）
├── chat-kg/                   # 前端Vue应用
└── weights/                   # 模型权重文件
```

---

## 🚀 阅读路径建议

### 第一阶段：理解整体流程（推荐顺序）

#### 1. **入口文件：`main.py`**
   - **作用**：知识图谱构建的主流程
   - **关键点**：
     - 创建 `KnowledgeGraphBuilder` 实例
     - 首次运行：调用 `get_base_kg_from_txt()` 从文本构建基础知识图谱
     - 迭代运行：循环调用 `run_iteration()` 扩展知识图谱
     - 收敛判断：通过 `extend_ratio()` 判断是否停止迭代
   - **阅读时间**：5-10分钟

#### 2. **核心类：`modules/knowledge_graph_builder.py`**
   - **作用**：知识图谱构建的核心逻辑
   - **关键方法**：
     - `__init__()`: 初始化路径和参数
     - `get_base_kg_from_txt()`: 从原始文本构建基础知识图谱
       - 调用 `process_text()` 预处理文本
       - 调用 `uie_execute()` 使用UIE抽取三元组
       - 调用 `auto_filter()` 过滤无效三元组
       - 调用 `refine_knowledge_graph()` 人工筛选
     - `run_iteration()`: 运行一次迭代
       - 创建 `ModelTrainer` 训练模型
       - 进行关系对齐和扩展
     - `extend_ratio()`: 计算扩展比例
   - **阅读时间**：20-30分钟

#### 3. **模型训练：`modules/model_trainer.py`**
   - **作用**：封装SPN模型的训练和预测流程
   - **关键方法**：
     - `split_data()`: 将数据切分为train/valid/test
     - `train_and_test()`: 训练SPN模型并预测
     - `relation_align()`: 将预测结果与测试集对齐
     - `refine_and_extend()`: 精炼并扩展知识图谱
   - **阅读时间**：20-30分钟

### 第二阶段：深入理解数据处理

#### 4. **数据预处理：`modules/prepare/`**
   - `preprocess.py`: 文本预处理（切分句子）
   - `process.py`: UIE模型执行（关系抽取）
   - `filter.py`: 自动过滤无效三元组
   - `utils.py`: 知识图谱精炼工具函数
   - **阅读时间**：30-40分钟

#### 5. **SPN模型：`modules/SPN4RE/`**
   - `main.py`: SPN模型的主入口
   - `models/`: 模型定义（setpred4RE.py为核心）
   - `trainer/`: 训练器
   - `utils/`: 工具函数（数据处理、评估指标等）
   - **阅读时间**：1-2小时（如果深入理解模型架构）

### 第三阶段：理解服务端和前端

#### 6. **后端服务：`server/`**
   - `main.py`: Flask服务启动入口
   - `app/__init__.py`: Flask应用初始化
   - `app/views/chat.py`: 对话API路由
   - `app/views/graph.py`: 知识图谱API路由
   - `app/utils/chat_glm.py`: ChatGLM模型调用
   - `app/utils/graph_utils.py`: 知识图谱查询工具
   - **阅读时间**：30-40分钟

#### 7. **前端应用：`chat-kg/`**
   - Vue.js应用，提供对话和知识图谱可视化界面
   - `src/views/ChatView.vue`: 对话界面
   - `src/views/GraphView.vue`: 知识图谱可视化
   - **阅读时间**：20-30分钟（如果熟悉Vue）

---

## 🔍 关键概念理解

### 1. **知识图谱构建流程**

```
原始文本 (raw_data.txt)
    ↓
文本预处理 (process_text)
    ↓
UIE关系抽取 (uie_execute)
    ↓
自动过滤 (auto_filter)
    ↓
人工筛选 (refine_knowledge_graph)
    ↓
基础知识图谱 (base_refined.json)
    ↓
迭代扩展 (run_iteration)
    ├─ 训练SPN模型
    ├─ 预测新三元组
    ├─ 关系对齐
    └─ 精炼扩展
    ↓
最终知识图谱 (knowledge_graph.json)
```

### 2. **迭代扩展机制**

- **目的**：从已有知识图谱中学习，发现新的三元组
- **流程**：
  1. 使用当前知识图谱训练SPN模型
  2. 在测试集上预测新三元组
  3. 过滤和精炼预测结果
  4. 合并到现有知识图谱
  5. 计算扩展比例，判断是否收敛

### 3. **数据格式**

**SPN格式（知识图谱数据）：**
```json
{
  "id": 0,
  "sentText": "句子文本",
  "relationMentions": [
    {
      "em1Text": "实体1",
      "em2Text": "实体2",
      "label": "关系类型"
    }
  ]
}
```

---

## 📝 阅读技巧

### 1. **从入口开始**
   - 先看 `main.py` 了解整体流程
   - 再看 `KnowledgeGraphBuilder` 类理解核心逻辑

### 2. **关注数据流**
   - 跟踪数据从原始文本到最终知识图谱的转换过程
   - 注意每个步骤的数据格式变化

### 3. **理解关键参数**
   - `project`: 项目名称，决定数据存储路径
   - `version`: 迭代版本号
   - `extend_ratio`: 扩展比例阈值（默认0.01）

### 4. **注意文件路径**
   - 数据存储在 `data/project_v1/` 目录下
   - 每次迭代结果保存在 `iteration_v{version}/` 目录
   - 历史状态保存在 `history/` 目录

### 5. **调试和恢复**
   - 使用 `--resume` 参数可以从checkpoint恢复训练
   - 检查点文件保存在 `data/project_v1/history/` 目录

---

## 🛠️ 常见问题

### Q1: 如何理解迭代过程？
**A**: 迭代过程是自举（bootstrap）学习：
- 第0次迭代：使用UIE从原始文本抽取基础三元组
- 第1次迭代：用基础三元组训练SPN，预测新三元组
- 第2次迭代：用扩展后的三元组训练SPN，继续扩展
- 直到扩展比例 < 1% 时停止

### Q2: UIE和SPN的区别？
**A**: 
- **UIE**: 通用信息抽取模型，用于初始三元组抽取
- **SPN**: Set Prediction Network，用于关系抽取和扩展，基于已有知识图谱训练

### Q3: 如何查看知识图谱？
**A**: 
- 启动后端服务：`python server/main.py`
- 访问前端：`http://localhost:5173`（Vue开发服务器）
- 或直接查看 `data/project_v1/iteration_v*/knowledge_graph.json`

---

## 📚 推荐阅读顺序总结

**快速了解（1-2小时）：**
1. `main.py`
2. `modules/knowledge_graph_builder.py`
3. `modules/model_trainer.py`

**深入理解（3-5小时）：**
4. `modules/prepare/` 目录下的文件
5. `server/` 目录下的服务端代码
6. `modules/SPN4RE/main.py` 和模型定义

**完整掌握（1-2天）：**
7. 阅读SPN模型的完整实现
8. 理解前端Vue应用的实现
9. 运行代码并调试

---

## 💡 实践建议

1. **运行代码**：先运行 `main.py` 看整体流程
2. **查看数据**：检查 `data/project_v1/` 目录下的JSON文件
3. **调试断点**：在关键方法处设置断点，观察数据变化
4. **修改参数**：尝试修改迭代次数、扩展比例等参数
5. **阅读日志**：查看 `iteration_v*/running_log.txt` 了解训练过程

---

**最后更新**: 2024年
**项目类型**: 知识图谱构建 + RAG对话系统

