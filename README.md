# 基于知识图谱和知识库的大模型对话系统

## 📋 项目概述

这是一个**基于知识图谱和知识库的大模型对话系统**，主要功能包括：
1. 从原始文本构建知识图谱
2. 使用迭代方法扩展和优化知识图谱
3. 提供基于知识图谱的对话服务
4. 提供知识图谱可视化前端

### 核心特性

- **知识图谱构建**：从原始文本自动抽取实体和关系，构建知识图谱
- **迭代优化**：通过迭代方法不断扩展和优化知识图谱
- **智能对话**：基于知识图谱和大模型提供智能问答服务
- **可视化展示**：提供直观的知识图谱可视化界面

---

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

## 📝 使用说明

### 知识图谱构建

构建完成后，知识图谱数据保存在 `data/project_v1/` 目录下：
- `base.json`：基础知识图谱
- `iteration_v<N>/`：各迭代版本的数据
- `history/`：检查点历史记录

### 对话服务

启动后端服务后，可以通过API或前端界面进行对话：
- 支持基于知识图谱的问答
- 支持实体识别和关系查询
- 支持知识图谱可视化展示

---

## ⚙️ 配置说明

主要配置文件位于 `config/settings.py`，可以配置：
- 最大迭代次数（`MAX_ITERATION`）
- 扩展比率阈值（`EXTEND_RATIO_THRESHOLD`）
- 服务器配置（`SERVER_HOST`、`SERVER_PORT`）
- 模型路径配置

---


## 📄 许可证

本项目仅供学习和研究使用。

---

## 🤝 贡献

欢迎提交Issue和Pull Request！

---

## 📧 联系方式

如有问题或建议，请通过Issue反馈。






