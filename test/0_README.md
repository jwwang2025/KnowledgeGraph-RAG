# 测试说明（test）

此目录用于存放用于仓库功能验证与演示的最小测试与说明文件。

- **文件**：`0_README.md` — 本说明文件。
- **示例脚本**：`../models/model-download/download_chatglm_model.py`（下载 ChatGLM 模型的示例脚本）。

test 文件介绍

本节概述 `test` 目录下可能包含的文件类型与当前已有文件，帮助贡献者快速定位与运行测试或示例。

- `0_README.md`：本说明文件，包含目录用途、运行说明与注意事项（当前文件）。
 - `0_README.md`：本说明文件，包含目录用途、运行说明与注意事项（当前文件）。

目录下文件说明（逐项）

- `test_chatglm_model.py`：演示/测试脚本，展示如何下载并加载 ChatGLM 模型并执行一次简单的推理或验证流程。可作为功能测试或示例运行，运行方式：
  - 从项目根目录执行：`python -m pytest test/test_chatglm_model.py` 或直接运行 `python test/test_chatglm_model.py`（取决于脚本是否包含 `if __name__ == "__main__":`）。
  - 注意：脚本可能依赖于已下载的模型文件或网络访问；若需要，请先运行相应的下载脚本或通过环境变量指定模型路径。

- `uie_model_usage_example.py`：UIE（信息抽取）模型的使用示例脚本，包含示例输入与调用流程，帮助理解如何在项目中集成 UIE 模型。运行方式：
  - 从项目根目录执行：`python test/uie_model_usage_example.py`。
  - 脚本为演示用途，实际集成时可将核心调用逻辑提取到项目模块中并在服务/流水线中复用。
（示例）常见测试文件与说明：

- `test_*.py`：单元/集成测试文件，通常可用 `pytest` 执行。命名建议以 `test_` 开头以便测试工具自动发现。
- `data/`：测试所需的示例数据或小型样本文件，可放在子目录中以便管理。
- `scripts/`：辅助脚本（如模型下载、数据准备脚本），例如项目中的 `models/model-download/download_chatglm_model.py`。

如何添加新的测试文件

1. 在 `test/` 下创建以 `test_` 开头的 Python 文件，例如 `test_example.py`，并编写可重复执行的单元测试。
2. 将任何测试依赖（测试数据或第三方库）记录在仓库根目录的依赖说明或 `requirements-dev.txt` 中。
3. 提交 PR 前，确保本地通过命令运行测试：

```bash
pip install -r requirements.txt
# 若有测试专用依赖：pip install -r requirements-dev.txt
pytest test/
```

注意事项

- 请在测试文件中避免下载大型模型或长时间运行的外部调用；若必须，请将此类脚本标记为慢速/集成测试并在 CI 中按需运行。
- 模型文件可能较大，请保证充足磁盘空间与网络带宽；对于示例脚本，建议允许通过环境变量或参数设置替代的下载路径或代理。

贡献

欢迎补充具体测试用例、数据样本或测试运行脚本。提交 PR 时请在描述中包含复现步骤与预期结果。
