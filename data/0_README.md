# Data 文件夹说明

本文件夹用于存储知识图谱构建和关系抽取任务相关的所有数据文件。

## 📁目录结构

```
data/
├── __init__.py                          # Python包初始化文件
├── 0_README.md                          # 本说明文件
├── clean_data.txt                       # 清理后的原始文本数据
├── clean_data_res_41lines_100epoch.json # 清理后的数据结果（41行，100轮训练）
├── clean_data_res_doc2_300epoch.json    # 清理后的数据结果（文档2，300轮训练）
├── clean_data_res_doc2_300epoch.txt     # 清理后的数据结果文本格式（文档2，300轮训练）
├── raw_data/                            # 原始数据文件夹
│   ├── raw_data.txt                     # 原始数据文件
│   └── raw_data_test.txt                # 原始测试数据文件
├── schema/                              # 关系抽取模式定义文件夹
│   ├── schema_v1.py                     # 关系抽取模式定义版本1
│   ├── schema_v2.py                     # 关系抽取模式定义版本2
│   ├── schema_v3.py                     # 关系抽取模式定义版本3
│   └── schema_v4.py                     # 关系抽取模式定义版本4
└── project_v1/                          # 项目版本1数据文件夹
    ├── base.json                        # 基础数据文件（原始标注数据）
    ├── base_filtered.json               # 过滤后的基础数据
    ├── base_refined.json                # 精炼后的基础数据
    ├── iter_v0.json                     # 迭代版本0的汇总数据
    ├── history/                         # 历史版本记录文件夹
    │   ├── 20230326-205422_iter_v0.json # 历史版本记录（时间戳格式）
    │   ├── 20230326-235601_iter_v0.json
    │   ├── 20230327-001537_iter_v1.json
    │   └── 20230327-001538_iter_v1.json
    ├── iteration_v0/                    # 迭代版本0数据文件夹
    │   ├── train.json                   # 训练集数据（包含句子和关系标注）
    │   ├── valid.json                   # 验证集数据
    │   ├── test.json                    # 测试集数据
    │   ├── prediction.json              # 模型预测结果
    │   ├── test_result_format.json      # 格式化后的测试结果
    │   ├── test_result_refine.json      # 精炼后的测试结果
    │   ├── knowledge_graph.json         # 构建的知识图谱数据
    │   ├── alphabet.json                # 字符/实体映射表
    │   └── running_log.txt              # 运行日志文件
    └── iteration_v1/                    # 迭代版本1数据文件夹
        ├── train.json                   # 训练集数据
        ├── valid.json                   # 验证集数据
        ├── test.json                    # 测试集数据
        ├── prediction.json              # 模型预测结果
        ├── test_result_format.json      # 格式化后的测试结果
        ├── test_result_refine.json      # 精炼后的测试结果
        ├── knowledge_graph.json         # 构建的知识图谱数据
        └── alphabet.json                # 字符/实体映射表
```

## 文件与文件夹说明

### 根目录文件

- **`__init__.py`**: Python包初始化文件，使data文件夹可以作为Python包导入
- **`0_README.md`**: 本说明文档，描述data文件夹的结构和用途
- **`clean_data.txt`**: 经过清理和预处理后的原始文本数据
- **`clean_data_res_*.json`**: 不同配置下清理数据的处理结果（包含训练轮数等信息）
- **`clean_data_res_*.txt`**: 清理数据的文本格式结果

### raw_data/ 文件夹

存储最原始的输入数据，通常是从外部来源获取的未处理数据。

- **`raw_data.txt`**: 原始数据文件，包含待处理的文本内容
- **`raw_data_test.txt`**: 原始测试数据文件，用于测试和验证

### schema/ 文件夹

存储关系抽取任务中使用的模式定义文件。每个版本可能包含不同的关系类型定义和实体类型定义。

- **`schema_v1.py` ~ `schema_v4.py`**: 不同版本的关系抽取模式定义，包含：
  - 关系类型定义（如"事故"、"原因"等）
  - 每个关系类型下的具体关系标签列表
  - 实体类型定义

### project_v1/ 文件夹

项目版本1的所有数据文件，包含从原始数据到最终知识图谱的完整数据处理流程。

#### 根级文件

- **`base.json`**: 基础数据文件，包含原始标注的句子和关系信息
- **`base_filtered.json`**: 经过过滤处理的基础数据，去除不符合要求的数据
- **`base_refined.json`**: 经过精炼处理的基础数据，数据质量更高
- **`iter_v0.json`**: 迭代版本0的汇总数据文件

#### history/ 文件夹

存储历史版本的数据快照，文件名格式为 `YYYYMMDD-HHMMSS_iter_vX.json`，用于版本管理和回溯。

#### iteration_v0/ 和 iteration_v1/ 文件夹

每个迭代版本包含完整的数据处理流程文件：

- **`train.json`**: 训练集数据，格式为JSON Lines，每行包含：
  - `id`: 数据ID
  - `sentText`: 句子文本
  - `relationMentions`: 关系标注列表，每个关系包含：
    - `em1Text`: 实体1文本
    - `em2Text`: 实体2文本
    - `label`: 关系标签
    - `em1Start`, `em1End`, `em2Start`, `em2End`: 实体位置信息

- **`valid.json`**: 验证集数据，格式同训练集
- **`test.json`**: 测试集数据，格式同训练集
- **`prediction.json`**: 模型在测试集上的预测结果
- **`test_result_format.json`**: 格式化后的测试结果，便于分析和评估
- **`test_result_refine.json`**: 经过后处理精炼的测试结果
- **`knowledge_graph.json`**: 从关系抽取结果构建的知识图谱，包含实体和关系三元组
- **`alphabet.json`**: 字符或实体的映射表，用于模型训练和推理
- **`running_log.txt`**: 运行日志，记录数据处理和模型训练的详细信息（仅iteration_v0包含）

## 数据流程

1. **原始数据**: `raw_data/` 文件夹中的原始文本数据
2. **数据清理**: 生成 `clean_data.txt` 和 `clean_data_res_*.json` 文件
3. **数据标注**: 生成 `project_v1/base.json` 等基础数据文件
4. **数据过滤与精炼**: 生成 `base_filtered.json` 和 `base_refined.json`
5. **数据划分**: 将数据划分为训练集、验证集和测试集，存储在 `iteration_vX/` 文件夹中
6. **模型训练**: 使用训练集训练关系抽取模型
7. **模型预测**: 在测试集上进行预测，生成 `prediction.json`
8. **结果处理**: 对预测结果进行格式化和精炼
9. **知识图谱构建**: 从关系抽取结果构建知识图谱，生成 `knowledge_graph.json`

## 注意事项

- 不同迭代版本（iteration_v0, iteration_v1等）可能使用不同的数据处理策略或模型配置
- 历史版本文件保存在 `history/` 文件夹中，便于版本管理和结果对比
- 模式定义文件（schema）的版本更新会影响关系抽取的结果
- 所有JSON文件通常使用UTF-8编码，每行一个JSON对象（JSON Lines格式）

