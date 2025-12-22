# Server 目录说明文档（路径中不要有中文，先 `cd /root/KnowledgeGraph-RAG-1/server` 后执行 `python main.py`）

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

### app/utils/ 工具函数模块

#### `app/utils/chat_glm.py`
**作用**: ChatGLM 模型的核心处理模块，负责模型加载、预测和增强问答。

**主要功能**:
- `start_model()`: 加载 ChatGLM-6B 模型
  - 从指定路径加载 tokenizer 和 model
  - 将模型加载到 GPU 并设置为评估模式
  - 初始化对话历史
- `stream_predict()`: 流式预测函数
  - **实体识别**: 使用 NE实体查询相关三元组
  - **图片搜索**: 根据用户R 提取用户输入中的实体
  - **知识图谱查询**: 根据输入搜索相关图片
  - **Wikipedia 查询**: 搜索实体相关的 Wikipedia 信息
  - **增强问答**: 将知识图谱三元组和 Wikipedia 信息作为参考资料，增强模型回答
  - 返回流式响应（逐字生成）

**工作流程**:
1. 从用户输入中提取实体（物体类、人物类、地点类等）
2. 根据实体查询知识图谱，获取相关三元组
3. 搜索相关图片和 Wikipedia 信息
4. 将参考资料整合到 prompt 中
5. 使用 ChatGLM 模型生成回答（流式输出）

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

## 🚀 启动方式

```bash
cd server
python main.py
```