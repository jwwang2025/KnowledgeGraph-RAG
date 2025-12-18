# UIE 模型下载和使用指南

## 📥 下载模型

### 方法1: 使用下载脚本（推荐）

```bash
# 在项目根目录执行
python models/model-download/download_uie_model.py
```

脚本会从 Hugging Face 镜像站下载所有必需的文件到 `models/uie-base/` 目录。

### 方法2: 使用 huggingface_hub 库

```bash
# 安装 huggingface_hub
pip install huggingface_hub

# 下载模型
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='xusenlin/uie-base', local_dir='models/uie-base')"
```

## 📦 下载的文件

- `config.json` - 模型配置文件
- `pytorch_model.bin` - 模型权重（约472MB）
- `tokenizer.json`, `tokenizer_config.json`, `vocab.txt` - Tokenizer 文件
- `special_tokens_map.json`, `added_tokens.json` - 特殊 token 配置
- `modeling_uie.py` - 自定义模型类（必需）
- `decode_utils.py` - 解码工具（必需）

## 🚀 使用方式

### 方式1: 使用 transformers 库（PyTorch 格式）

```python
from transformers import AutoTokenizer, AutoModel
from pathlib import Path

# 模型路径
model_path = Path("models/uie-base").absolute()

# 加载模型（需要 trust_remote_code=True）
tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
model = AutoModel.from_pretrained(str(model_path), trust_remote_code=True)

# 使用模型进行推理
text = "张三在北京大学工作。"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
# 注意: 实际的 UIE 推理需要特定的解码逻辑，参考 decode_utils.py
```

### 方式2: 使用 PaddleNLP Taskflow（推荐，与项目兼容）

```python
from paddlenlp import Taskflow

# 定义 schema
schema = {
    "人物": ["工作单位", "研究方向"],
}

# 初始化 Taskflow（会自动下载 PaddlePaddle 格式的模型）
ie = Taskflow("information_extraction", schema=schema, model="uie-base")

# 使用
text = "张三在北京大学工作，他的研究方向是自然语言处理。"
result = ie(text)
print(result)
```

**注意**: 
- PaddleNLP 的 Taskflow 会自动从 PaddlePaddle 模型库下载模型，不是使用 Hugging Face 的 PyTorch 格式
- 如果要从本地加载，需要将 PyTorch 模型转换为 PaddlePaddle 格式

### 方式3: 在项目中使用（修改 process.py）

项目中的 `modules/prepare/process.py` 已经配置了 UIE 模型使用方式。要使用本地下载的模型，需要：

1. **如果使用 PaddleNLP**: 需要将 PyTorch 模型转换为 PaddlePaddle 格式，或者直接使用 PaddleNLP 的自动下载功能

2. **如果使用 transformers**: 需要修改 `process.py` 中的代码，使用 transformers 库加载模型

## 🔄 模型格式转换（可选）

如果需要将 Hugging Face 的 PyTorch 模型转换为 PaddlePaddle 格式以用于 PaddleNLP：

1. 使用 PaddleNLP 提供的转换工具
2. 或者直接使用 PaddleNLP 的预训练模型（推荐）

## 📝 测试模型

运行示例脚本测试模型是否正常工作：

```bash
python models/model-download/uie_model_usage_example.py
```

## ⚠️ 注意事项

1. **模型格式**: Hugging Face 上的模型是 PyTorch 格式，而项目主要使用 PaddleNLP（PaddlePaddle 格式）
2. **自定义代码**: 加载模型时需要 `trust_remote_code=True`，因为使用了自定义的 `modeling_uie.py`
3. **解码逻辑**: UIE 模型需要特定的解码逻辑（`decode_utils.py`），直接使用 transformers 可能无法直接进行信息抽取
4. **推荐方式**: 对于本项目，建议使用 PaddleNLP 的 Taskflow，它会自动处理所有细节

## 🔗 相关链接

- Hugging Face 模型页面: https://huggingface.co/xusenlin/uie-base
- PaddleNLP UIE 文档: https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/uie
- 项目中的 UIE 使用代码: `modules/prepare/process.py`

