# ChatGLM-6B 模型手动下载指南

## 方法1：使用Python脚本下载（推荐）

### 步骤：

1. **安装依赖**
```powershell
pip install huggingface_hub
```

2. **运行下载脚本**
```powershell
# 从项目根目录运行
python models/1_model-download/download_chatglm_model.py
```

3. **使用镜像源（如果遇到SSL问题）**
```powershell
# 设置镜像源
$env:HF_ENDPOINT='https://hf-mirror.com'
python models/1_model-download/download_chatglm_model.py
```

4. **指定下载路径**
```powershell
python models/1_model-download/download_chatglm_model.py --local-dir "G:\path\to\chatglm-6b"
```

下载完成后，模型会保存在项目根目录下的 `models\chatglm-6b` 目录（或你指定的路径）。

---

## 方法2：使用Git LFS手动下载

### 步骤：

1. **安装Git LFS**
   - 下载：https://git-lfs.github.com/
   - 安装后运行：`git lfs install`

2. **克隆模型仓库**
```powershell
# 在项目根目录下
mkdir models
cd models
git clone https://huggingface.co/THUDM/chatglm-6b
```

3. **使用镜像源（推荐）**
```powershell
# 在项目根目录下
mkdir models
cd models
git clone https://hf-mirror.com/THUDM/chatglm-6b
```

4. **等待下载完成**
   - 模型文件较大（约12GB），下载需要时间
   - Git LFS会自动下载大文件

---

## 方法3：使用HuggingFace镜像站直接下载

### 步骤：

1. **访问镜像站**
   - 打开：https://hf-mirror.com/THUDM/chatglm-6b
   - 或访问：https://huggingface.co/THUDM/chatglm-6b

2. **手动下载文件**
   需要下载的主要文件：
   - `config.json` - 模型配置
   - `tokenizer_config.json` - 分词器配置
   - `tokenizer.json` - 分词器文件
   - `modeling_chatglm.py` - 模型代码（需要trust_remote_code=True）
   - `pytorch_model.bin` 或 `model.safetensors` - 模型权重（最大文件，约12GB）
   - `vocab.txt` - 词汇表
   - 其他配置文件

3. **创建目录结构**
```powershell
# 在项目根目录下
mkdir models\chatglm-6b
# 将所有文件放入此目录
```

---

## 方法4：使用国内镜像源（清华大学）

### 步骤：

1. **设置镜像源环境变量**
```powershell
$env:HF_ENDPOINT='https://hf-mirror.com'
```

2. **使用Python下载**
```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="THUDM/chatglm-6b",
    local_dir="models/chatglm-6b",  # 项目根目录下的 models/chatglm-6b
    local_dir_use_symlinks=False
)
```

---

## 下载完成后配置

### 方式1：设置环境变量（推荐）
```powershell
# 设置模型路径（使用相对路径或绝对路径）
$env:CHATGLM_MODEL_PATH='models\chatglm-6b'
# 或使用绝对路径
# $env:CHATGLM_MODEL_PATH='G:\3_LLM_AppDev\0_Resume_Projects\KnowledgeGraph-RAG\models\chatglm-6b'

# 运行服务器
cd server
python main.py
```

### 方式2：修改代码
编辑 `server/app/utils/chat_glm.py`，修改第126行：
```python
# 使用相对路径（从server目录到根目录的models）
model_path = os.getenv('CHATGLM_MODEL_PATH', '../models/chatglm-6b')
# 或使用绝对路径
# model_path = os.getenv('CHATGLM_MODEL_PATH', 'G:/3_LLM_AppDev/0_Resume_Projects/KnowledgeGraph-RAG/models/chatglm-6b')
```

---

## 验证模型文件

下载完成后，确保以下文件存在：
- ✅ `config.json`
- ✅ `tokenizer_config.json`
- ✅ `tokenizer.json`
- ✅ `pytorch_model.bin` 或 `model.safetensors`（约12GB）
- ✅ `modeling_chatglm.py`
- ✅ `vocab.txt`

---

## 常见问题

### Q: 下载速度慢怎么办？
A: 使用国内镜像源 `https://hf-mirror.com`

### Q: SSL证书错误？
A: 设置镜像源或使用 `DISABLE_SSL_VERIFY='true'`（仅测试环境）

### Q: 磁盘空间不足？
A: ChatGLM-6B模型需要约12-15GB空间，请确保有足够空间

### Q: 下载中断怎么办？
A: 使用 `snapshot_download` 的 `resume_download=True` 参数支持断点续传

---

## 推荐方案

**最简单的方式**：
```powershell
# 1. 设置镜像源
$env:HF_ENDPOINT='https://hf-mirror.com'

# 2. 安装依赖
pip install huggingface_hub

# 3. 运行下载脚本（从项目根目录）
python models/1_model-download/download_chatglm_model.py

# 4. 设置模型路径并运行
$env:CHATGLM_MODEL_PATH='models\chatglm-6b'
cd server
python main.py
```

