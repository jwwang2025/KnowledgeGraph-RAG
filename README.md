# KnowledgeGraph-RAG：知识图谱 + RAG 双驱动大模型对话系统

## 项目简介

本项目提出一种 **“结构化知识图谱 + 非结构化文档库” 双检索驱动的检索增强生成（RAG）对话系统 **，核心目标是解决大模型在事实性问答中存在的 “幻觉” 问题，同时提升回答的可追溯性与上下文相关性。系统以知识图谱（结构化知识）和文档库（非结构化知识）为双重外部知识源，通过 RAG 技术将检索到的精准事实证据动态注入生成模型（如 ChatGLM-6B），最终实现 “有依据、可验证、高准确” 的智能对话服务。

--- 

核心功能（按流程逻辑排序）
知识图谱自动化构建：从非结构化原始文本中自动抽取实体、关系与属性，生成结构化知识图谱，为 RAG 提供精准的结构化检索载体。
知识图谱迭代优化：通过多轮校验、补充与去重机制，持续扩展知识图谱的覆盖范围、修正错误关系，为检索环节提供高质量知识支撑。
双源 RAG 知识增强对话：融合知识图谱的结构化检索（精准匹配实体关系）与文档库的非结构化检索（补充上下文细节），将双重证据注入大模型，生成兼具事实性与完整性的回答，并支持回答来源溯源。
交互式可视化展示：提供直观的知识图谱可视化界面，同步展示检索到的实体关系、文档片段等证据，支持用户交互式探索知识关联、验证回答的真实性。
核心特性（突出技术优势与价值）
双检索源协同增强：以 “知识图谱（结构化）+ 文档库（非结构化）” 为核心，兼顾检索的精准性（实体关系匹配）与完整性（上下文补充），解决单一检索源的覆盖局限。
自动化知识构建与迭代：无需大量人工标注，从原始文本自动生成知识图谱，并通过迭代机制持续优化，降低知识工程成本，提升知识更新效率。
可追溯的事实性生成：所有回答均关联明确的检索证据（知识图谱三元组 / 文档片段），支持 “答案 - 证据” 双向溯源，彻底解决大模型 “幻觉” 问题，满足学术、专业场景的可信度要求。
上下文感知的智能对话：在 RAG 框架基础上保留大模型的上下文理解能力，支持多轮对话中的逻辑连贯与多步推理，兼顾事实准确性与交互自然性。
交互式知识验证：通过可视化界面直观呈现知识图谱的实体关联与检索证据链，用户可手动探索知识结构、校验回答依据，增强系统的透明性与可信任度。
---

## 技术栈

**后端**
- Python 3.x、Flask（API 服务）  
- PyTorch、Transformers（模型推理与微调）  
- PaddlePaddle / PaddleNLP（UIE 模型支持）  
- SPN4RE（关系抽取）


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
## 系统流程
![alt text](proj-docs/structure.png)


## 问答页面（无检索增强）
![alt text](proj-docs/QAPagenG.png)

## 问答页面（有检索增强）
![alt text](proj-docs/QAPagewG.png)

## 图谱页面
![alt text](proj-docs/graphPage.png)

## 📁 项目结构

```
KnowledgeGraph-RAG/
├── main.py                    # 主入口：知识图谱构建流程
├── config/                    # 配置文件
│   └── settings.py           # 项目配置
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
│   ├── raw_data/             # 原始文本数据
│   └── project_v1/           # 项目数据（知识图谱、迭代结果等）
│       └── history/          # 检查点历史记录
├── models/                    # 模型文件
│   ├── chatglm-6b/          # ChatGLM-6B模型
│   ├── uie-base/            # UIE模型
│   └── bert-base-chinese/   # BERT中文模型
├── chat-kg/                   # 前端Vue应用
│   ├── src/                  # 源代码
│   └── package.json          # 前端依赖配置
├── utils/                     # 工具脚本
└── requirements.txt           # Python依赖列表
```

---

## 🚀 快速开始

### 第一步：环境准备

#### 1.1 系统要求

- Python 3.8+
- Node.js 16+（如需运行前端）
- CUDA支持的GPU（推荐，用于模型推理）

#### 1.2 安装Python依赖

```bash
# 在项目根目录执行
pip install -r requirements.txt
```

**注意**：如果遇到依赖冲突，建议使用虚拟环境：

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

#### 1.3 安装前端依赖（可选）

如需运行前端可视化界面，需要安装Node.js依赖：

```bash
# 进入前端目录
cd chat-kg

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
- 支持中文和英文文本

如果文件不存在，需要创建该文件并添加文本内容：

```bash
# 创建数据目录（如果不存在）
mkdir -p data/raw_data

# 编辑或创建 raw_data.txt 文件
# 添加你的原始文本数据
```

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

**检查点文件命名格式：** `YYYYMMDD-HHMMSS_iter_v<版本号>.json`

---

### 第四步：启动后端服务

#### 4.1 配置环境变量（可选）

在项目根目录创建 `.env` 文件，配置服务器参数：

```env
SERVER_HOST=0.0.0.0
SERVER_PORT=5000
DEBUG=False
SECRET_KEY=your-secret-key-here
```

#### 4.2 启动Flask服务

```bash
# 在项目根目录执行
cd server
python main.py
```

**服务启动后：**
- 后端API服务默认运行在 `http://localhost:5000`
- 首次启动会自动加载ChatGLM-6B模型（需要一些时间）
- 确保已构建知识图谱，否则对话功能可能无法正常工作

**API端点：**
- `/api/chat`：对话接口
- `/api/graph`：知识图谱查询接口

---

### 第五步：启动前端应用（可选）

#### 5.1 启动前端开发服务器

```bash
# 进入前端目录
cd chat-kg

# 启动开发服务器
npm run dev
# 或使用服务器模式（允许外部访问）
npm run server
```

**前端访问地址：**
- 开发模式：`http://localhost:5173`（Vite默认端口）
- 服务器模式：`http://0.0.0.0:5173`（允许局域网访问）

---






