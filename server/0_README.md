# Server 目录说明文档(路径中不要有中文cd /root/KnowledgeGraph-RAG-1/server
python main.py)

本文档详细说明了 `server` 文件夹的目录结构以及各个文件和文件夹的作用。

## 📁 目录结构

```
server/
├── 0_README.md                    # 本说明文档
├── main.py                         # Flask 应用启动入口
├── app/                            # Flask 应用主模块
│   ├── __init__.py                # Flask 应用初始化，注册蓝图
│   ├── views/                      # 视图层（API 路由）
│   │   ├── __init__.py
│   │   ├── chat.py                # 聊天对话 API
│   │   └── graph.py                # 知识图谱查询 API
│   └── utils/                      # 工具函数模块
│       ├── __init__.py
│       ├── chat_glm.py            # ChatGLM 模型加载和预测
│       ├── graph_utils.py         # 知识图谱查询和转换工具
│       ├── ner.py                 # 命名实体识别（NER）
│       ├── query_wiki.py          # Wikipedia 查询工具
│       ├── image_searcher.py      # 图片搜索工具
│       └── logger.py              # 日志配置
└── data/                          # 数据目录
    └── data.json                  # 知识图谱数据（节点、边、句子）
```

## 📄 文件详细说明

### 核心文件

#### `main.py`
**作用**: Flask 应用的主启动文件，负责启动 Web 服务器。

**主要功能**:
- 设置 GPU 可见设备（`CUDA_VISIBLE_DEVICES`）
- 启动 ChatGLM 模型
- 配置 Flask 应用密钥
- 启动 Flask 服务器（默认运行在 `0.0.0.0:8000`）
- 启用多线程模式

**关键代码**:
```python
start_model()  # 加载 ChatGLM 模型
apps.run(host='0.0.0.0', port=8000, debug=False, threaded=True)
```

### app/ Flask 应用模块

#### `app/__init__.py`
**作用**: Flask 应用初始化文件，配置应用并注册路由蓝图。

**主要功能**:
- 创建 Flask 应用实例
- 配置 CORS（跨域资源共享），允许前端跨域访问
- 注册 `chat` 和 `graph` 两个蓝图
- 定义根路由和错误处理

**注册的路由**:
- `/` - 根路径，返回成功消息
- `/chat/*` - 聊天相关路由（由 `chat.py` 处理）
- `/graph/*` - 知识图谱相关路由（由 `graph.py` 处理）

### app/views/ 视图层

#### `app/views/chat.py`
**作用**: 聊天对话 API 视图，处理用户聊天请求。

**主要功能**:
- `GET /chat/` - 测试接口，返回 "Chat Get!"
- `POST /chat/` - 处理聊天请求
  - 接收用户输入（`prompt`）和对话历史（`history`）
  - 调用 `stream_predict` 进行流式预测
  - 返回流式响应（Server-Sent Events 格式）

**请求格式**:
```json
{
  "prompt": "用户输入的问题",
  "history": [["问题1", "回答1"], ["问题2", "回答2"]]
}
```

#### `app/views/graph.py`
**作用**: 知识图谱查询 API 视图。

**主要功能**:
- `GET /graph/` - 获取完整的知识图谱数据
  - 读取 `data/data.json` 文件
  - 返回图谱的节点、边和句子信息

**返回格式**:
```json
{
  "data": {
    "nodes": [...],
    "links": [...],
    "sents": [...]
  },
  "message": "You Got It!"
}
```

### app/utils/ 工具函数模块

#### `app/utils/chat_glm.py`
**作用**: ChatGLM 模型的核心处理模块，负责模型加载、预测和增强问答。

**主要功能**:
- `start_model()`: 加载 ChatGLM-6B 模型
  - 从指定路径加载 tokenizer 和 model
  - 将模型加载到 GPU 并设置为评估模式
  - 初始化对话历史
- `stream_predict()`: 流式预测函数
  - **实体识别**: 使用 NER 提取用户输入中的实体
  - **知识图谱查询**: 根据实体查询相关三元组
  - **图片搜索**: 根据用户输入搜索相关图片
  - **Wikipedia 查询**: 搜索实体相关的 Wikipedia 信息
  - **增强问答**: 将知识图谱三元组和 Wikipedia 信息作为参考资料，增强模型回答
  - 返回流式响应（逐字生成）

**工作流程**:
1. 从用户输入中提取实体（物体类、人物类、地点类等）
2. 根据实体查询知识图谱，获取相关三元组
3. 搜索相关图片和 Wikipedia 信息
4. 将参考资料整合到 prompt 中
5. 使用 ChatGLM 模型生成回答（流式输出）

#### `app/utils/graph_utils.py`
**作用**: 知识图谱查询和转换工具。

**主要功能**:
- `search_node_item()`: 在知识图谱中搜索节点及其关联信息
  - 根据用户输入搜索匹配的节点
  - 递归搜索节点的邻居节点（深度为 1）
  - 构建子图（包含节点、边和句子）
  - 返回精简的知识图谱结构
- `convert_graph_to_triples()`: 将图谱结构转换为三元组列表
  - 从图谱的 links 中提取三元组（主体-关系-客体）
  - 支持按实体过滤三元组

**图谱结构**:
```python
{
  'nodes': [...],  # 节点列表
  'links': [...],  # 边列表（关系）
  'sents': [...]   # 句子列表（来源文本）
}
```

#### `app/utils/ner.py`
**作用**: 命名实体识别（Named Entity Recognition）工具。

**主要功能**:
- 使用 PaddleNLP 的 NER 模型进行实体识别
- `predict()`: 对文本进行实体识别
- `get_entities()`: 获取指定类型的实体
  - 支持多种实体类型：物体类、人物类、地点类、组织机构类、事件类、世界地区类、术语类
  - 返回实体列表

**模型路径**: `weights/model_41_100`（PaddleNLP 格式的 NER 模型）

#### `app/utils/query_wiki.py`
**作用**: Wikipedia 查询工具，用于获取实体的百科信息。

**主要功能**:
- `search()`: 搜索 Wikipedia 页面
  - 使用 `wikipediaapi` 库查询中文 Wikipedia
  - 如果简体中文查询失败，自动尝试繁体中文
  - 返回 Wikipedia 页面对象（包含标题和摘要）

**返回内容**:
- `title`: 页面标题
- `summary`: 页面摘要

#### `app/utils/image_searcher.py`
**作用**: 图片搜索工具，根据关键词返回相关图片 URL。

**主要功能**:
- `search()`: 根据查询关键词搜索图片
  - 维护一个关键词到图片 URL 的映射表
  - 如果查询中包含关键词，返回对应的图片 URL
  - 支持的关键词包括：江南大学、军舰、消防手套、灭火剂等

**使用场景**: 在聊天回答中提供相关图片，增强用户体验

#### `app/utils/logger.py`
**作用**: 日志配置模块。

**主要功能**:
- 配置日志格式：`%(asctime)s %(levelname)s %(name)s %(message)s`
- 创建名为 `server` 的 logger 实例
- 用于记录服务器运行日志

### data/ 数据目录

#### `data/data.json`
**作用**: 存储知识图谱的完整数据。

**数据结构**:
```json
{
  "nodes": [
    {
      "id": "0",
      "name": "节点名称",
      "category": 2,
      "value": 43,
      "symbolSize": 113.5,
      ...
    }
  ],
  "links": [
    {
      "source": 0,
      "target": 1,
      "name": "关系名称",
      "sent": 0,
      ...
    }
  ],
  "sents": [
    "句子文本1",
    "句子文本2",
    ...
  ]
}
```

**说明**:
- `nodes`: 知识图谱的节点（实体）列表
- `links`: 知识图谱的边（关系）列表，包含源节点、目标节点和关系名称
- `sents`: 句子列表，记录每个关系的来源文本

## 🔄 工作流程

### 聊天问答流程

1. **用户发送请求** → `POST /chat/`
2. **实体识别** → `ner.py` 提取实体
3. **知识图谱查询** → `graph_utils.py` 查询相关三元组
4. **增强信息收集**:
   - 图片搜索 → `image_searcher.py`
   - Wikipedia 查询 → `query_wiki.py`
5. **构建增强 Prompt** → 将三元组和 Wikipedia 信息作为参考资料
6. **模型预测** → `chat_glm.py` 使用 ChatGLM 生成回答
7. **流式返回** → 逐字返回生成结果

### 知识图谱查询流程

1. **用户请求** → `GET /graph/`
2. **读取数据** → 从 `data/data.json` 读取完整图谱
3. **返回数据** → 返回 JSON 格式的图谱数据

## 🔧 技术栈

- **Web 框架**: Flask
- **跨域支持**: Flask-CORS
- **语言模型**: ChatGLM-6B（基于 Transformers）
- **实体识别**: PaddleNLP NER
- **知识图谱**: 自定义 JSON 格式
- **百科查询**: wikipediaapi
- **字符转换**: OpenCC（繁简转换）

## 📝 配置说明

### 模型路径配置

在 `chat_glm.py` 中需要配置 ChatGLM 模型路径：
```python
tokenizer = AutoTokenizer.from_pretrained("/fast/zwj/ChatGLM-6B/weights", ...)
model = AutoModel.from_pretrained("/fast/zwj/ChatGLM-6B/weights", ...)
```

### NER 模型路径

在 `ner.py` 中配置 NER 模型路径：
```python
self.model = Taskflow("ner", task_path="weights/model_41_100")
```

### 服务器配置

在 `main.py` 中配置服务器参数：
- 主机: `0.0.0.0`（监听所有网络接口）
- 端口: `8000`
- 调试模式: `False`
- 多线程: `True`

## 🚀 启动方式

```bash
cd server
python main.py
```

服务器将在 `http://0.0.0.0:8000` 启动。

## 📌 注意事项

1. **GPU 要求**: ChatGLM 模型需要 GPU 支持（CUDA）
2. **模型路径**: 需要确保 ChatGLM 和 NER 模型路径正确
3. **数据文件**: 确保 `data/data.json` 文件存在且格式正确
4. **依赖安装**: 需要安装所有依赖包（见 `requirements.txt`）
5. **内存要求**: ChatGLM-6B 模型需要较大的显存（建议 12GB+）

## 🔗 API 端点

- `GET /` - 健康检查
- `GET /chat/` - 聊天测试
- `POST /chat/` - 聊天对话（流式响应）
- `GET /graph/` - 获取知识图谱数据

