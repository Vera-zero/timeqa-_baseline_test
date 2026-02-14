## 📁 项目结构

```
ETE-Graph/
├── DyG-RAG/                      # DyG-RAG 核心实现
│   ├── graphrag/                 # 源代码
│   ├── examples/                 # 使用示例
│   │   ├── timeqa_run.py
│   │   └── temreason_run.py
│   ├── reproduce/                # 论文复现脚本
│   │   ├── timeqa.py
│   │   ├── tempreason.py
│   │   └── complextr.py
│   └── models/                   # 模型下载脚本
├── GraphRAG/                     # GraphRAG 基线方法
│   ├── Option/                   # 配置文件
│   │   ├── Config2.yaml          # 全局配置
│   │   └── Method/               # 方法配置
│   │       ├── LGraphRAG.yaml
│   │       ├── GGraphRAG.yaml
│   │       ├── HippoRAG.yaml
│   │       ├── LightRAG.yaml
│   │       └── RAPTOR.yaml
│   ├── Core/                     # 核心实现
│   ├── Data/                     # 数据加载器
│   └── main.py                   # 主入口 (待添加)
├── dataset/                      # 数据集文件
│   ├── timeqa/
│   │   └── test_processed.json
│   └── tempreason/
├── QA-result/                    # QA系统输出结果
│   ├── timeqa/
│   │   ├── DyG-RAG/
│   │   └── HippoRAG/
│   └── tempreason/
├── evaluation_results/           # 评估输出
├── evaluate_qa_results.py        # 评估脚本
├── graph_baseline.md             # 基线配置文档
├── change.md                     # 修改记录
└── README.md                     # 本文件
```

## 🚀 快速开始

### 1. 环境配置

#### 安装依赖

```bash
# 创建 conda 环境
conda create -n ete-graph python=3.10
conda activate ete-graph

# 安装项目依赖
cd /workspace/ETE-Graph
pip install -r requirements.txt
```

#### 下载必需模型

```bash
# DyG-RAG 所需模型 (NER 和 Cross-Encoder)
cd DyG-RAG/models
python download.py
```

### 2. 启动本地 LLM 服务

使用 vLLM 启动 Qwen3-32B 模型服务:

```bash
# 方式1: 单 GPU
python -m vllm.entrypoints.openai.api_server \
    --model /workspace/models/Qwen3-32B \
    --served-model-name qwen3-32b \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 32768

# 方式2: 多 GPU (张量并行)
CUDA_VISIBLE_DEVICES=2,3 python -m vllm.entrypoints.openai.api_server \
    --model /workspace/models/Qwen3-32B \
    --served-model-name qwen3-32b \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --max-model-len 32768 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.85
```

### 3. 运行示例

#### DyG-RAG 快速示例

```bash
# 运行 TimeQA 数据集示例
cd DyG-RAG/examples
python timeqa_run.py

# 运行 TempReason 数据集示例
python temreason_run.py
```

#### GraphRAG 方法运行

```bash
# 运行 HippoRAG 方法
cd GraphRAG
python main.py -opt Option/Method/HippoRAG.yaml \
               -dataset_name timeqa \
               -data_root /workspace/ETE-Graph/dataset

# 运行 LightRAG 方法
python main.py -opt Option/Method/LightRAG.yaml \
               -dataset_name tempreason \
               -data_root /workspace/ETE-Graph/dataset

# 运行 LGraphRAG (Local search)
python main.py -opt Option/Method/LGraphRAG.yaml \
               -dataset_name timeqa

# 运行 GGraphRAG (Global search)
python main.py -opt Option/Method/GGraphRAG.yaml \
               -dataset_name timeqa
```

## 📊 评估工具

### 评估所有 QA 结果

```bash
cd /workspace/ETE-Graph
python evaluate_qa_results.py
```

评估完成后,结果将保存在 `evaluation_results/` 目录:
- `results_table.md` - Markdown 格式的评估表格
- `results_table.csv` - CSV 格式的评估表格

### 高级评估选项

```bash
# 仅评估特定数据集
python evaluate_qa_results.py --dataset timeqa

# 仅评估特定方法
python evaluate_qa_results.py --method HippoRAG

# 指定输出格式
python evaluate_qa_results.py --output-format markdown

# 自定义路径
python evaluate_qa_results.py \
    --qa-result-dir /path/to/QA-result \
    --dataset-dir /path/to/dataset \
    --output-dir /path/to/output
```

### 评估指标

- **EM (Exact Match)**: 精确匹配率 - 预测答案与标准答案完全匹配的比例
- **F1 Score**: Token 级别的 F1 分数 - 衡量预测答案与标准答案的重叠程度

详细评估说明请参考[评估文档](#评估指标说明)。

## 📖 DyG-RAG 详细说明

### DyG-RAG 核心创新

1. **首个动态图结构**: 从事件中心视角构建和存储时序文本知识
2. **事件粒度显式时序编码**: 提出动态事件单元(DEU)粒度,在知识组织阶段显式嵌入时序信息
3. **RAG-推理集成**: 自然支持检索增强生成与时序推理的集成,启用 Time-CoT prompting
4. **实验验证**: 在三种不同类型的时序问答数据集上验证了优越性能

### DyG-RAG 架构

<details>
<summary>查看架构图</summary>

DyG-RAG 的整体框架包括:
- **事件抽取与时序编码**: 从文本中抽取事件并编码时序信息
- **动态图构建**: 构建事件中心的动态知识图谱
- **时序感知检索**: 基于时序约束的相关事件检索
- **Time-CoT 推理**: 集成时序链式推理的答案生成

详细架构请参考 [DyG-RAG README](DyG-RAG/README.md) 和[论文](https://www.arxiv.org/abs/2507.13396)。
</details>

### 论文复现

```bash
cd DyG-RAG/reproduce

# TimeQA 数据集复现
python timeqa.py

# TempReason 数据集复现
python tempreason.py

# ComplexTR 数据集复现
python complextr.py
```

## 🔧 配置说明

### 统一基线配置

为确保不同 RAG 方法之间的公平比较,项目采用统一的基线配置。详细配置说明请参考 [graph_baseline.md](graph_baseline.md)。

#### 核心配置参数

| 配置项 | 值 | 说明 |
|--------|-----|------|
| **LLM 模型** | Qwen3-32B | 本地 VLLM 部署 |
| **LLM Base URL** | `http://localhost:8000/v1` | VLLM API 地址 |
| **LLM Temperature** | 0.0 | 确定性输出 |
| **LLM Max Token** | 32768 | 最大上下文长度 |
| **Embedding 模型** | Qwen3-Embedding-8B | 本地模型 |
| **Embedding 路径** | `/workspace/models/Qwen3-Embedding-8B` | 本地路径 |
| **Embedding 维度** | 4096 | 向量维度 |
| **Embedding 上下文** | 32768 | 最大上下文 |
| **Chunk Size** | 1200 tokens | 文本分块大小 |
| **Chunk Overlap** | 100 tokens | 分块重叠大小 |
| **Max Token for Text Unit** | 12000 tokens | 文本单元最大 token 数 |

#### Qwen3-32B 思考功能配置

DyG-RAG 默认禁用 Qwen3-32B 的思考(Thinking)功能以提升性能:
- **性能提升**: 响应速度提升约 20-40%
- **输出简洁**: 直接给出答案,无中间推理过程
- **配置方式**: 通过 `extra_body` 参数设置 `enable_thinking: false`

详见 [graph_baseline.md - Qwen3-32B 思考功能配置](graph_baseline.md#qwen3-32b-思考功能配置)。

### GraphRAG 配置文件

GraphRAG 方法的配置位于 `GraphRAG/Option/Method/` 目录:

```yaml
# 示例: HippoRAG.yaml
llm:
  api_type: "open_llm"
  base_url: 'http://localhost:8000/v1'
  model: "qwen3-32b"
  api_key: "EMPTY"
  max_token: 32768
  temperature: 0.0

embedding:
  api_type: "hf"
  model: "/workspace/models/Qwen3-Embedding-8B"
  dimensions: 4096
  max_token_size: 32768
  embed_batch_size: 128

chunk:
  chunk_size: 1200
  chunk_overlap: 100
  token_model: "gpt-3.5-turbo"
```

## 📚 数据集

### 支持的数据集

1. **TimeQA**: 时序问答数据集,包含 easy/hard 两种难度级别
2. **TempReason**: 时序推理数据集,包含 L2/L3 两种推理深度
3. **ComplexTR**: 复杂时序推理数据集

### 数据集格式

项目支持两种数据集格式:

#### TempReason 格式
```json
{
  "content_num": 15266,
  "questions_num": 16017,
  "contents": [
    {
      "fact_context": "文档内容...",
      "question_list": [
        {
          "question": "问题?",
          "text_answers": {"text": ["答案1", "答案2"]},
          "date": "May 27, 1946",
          "id": "L2_Q367750_P39_0"
        }
      ]
    }
  ]
}
```

#### TimeQA 格式
```json
{
  "content_num": 3500,
  "datas": [
    {
      "idx": "/wiki/Knox_Cunningham#P39",
      "context": "文档内容...",
      "questions_list": [
        {
          "question": "问题?",
          "targets": ["答案1", "答案2"],
          "level": "easy"
        }
      ]
    }
  ]
}
```

## 🛠️ 高级用法

### 添加新的 QA 方法

1. 在对应数据集目录下创建方法文件夹:
   ```bash
   mkdir -p QA-result/timeqa/YourMethod
   ```

2. 将结果保存为 `results.json`,支持以下格式之一:
   - JSON 格式(类似 DyG-RAG)
   - JSONL 格式(类似 HippoRAG)

3. 确保结果包含必要字段:
   - `question`: 问题文本
   - `output` 或 `answer`: 模型输出(包含加粗实体 `**实体名**`)
   - `level`(可选): 难度级别(easy/hard)

4. 运行评估脚本:
   ```bash
   python evaluate_qa_results.py
   ```

### 自定义 GraphRAG 方法

1. 复制并修改现有配置文件:
   ```bash
   cp GraphRAG/Option/Method/HippoRAG.yaml GraphRAG/Option/Method/YourMethod.yaml
   ```

2. 修改配置参数

3. 运行自定义方法:
   ```bash
   python GraphRAG/main.py -opt Option/Method/YourMethod.yaml -dataset_name timeqa
   ```

## 📊 评估指标说明

### 文本规范化

在计算 EM 和 F1 之前,答案会经过以下规范化处理:
- 转为小写
- 移除冠词(a, an, the)
- 移除标点符号
- 移除 Unicode 字符
- 规范化空格

### F1 计算方法

F1 分数基于 Token 级别的重叠计算:
```
Precision = 匹配的 tokens 数 / 预测答案的 tokens 数
Recall = 匹配的 tokens 数 / 标准答案的 tokens 数
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

对于每个问题,会计算预测答案与所有标准答案的 F1 分数,取最大值。

### 评估结果示例

```markdown
| Method | Dataset | Level | EM (%) | F1 (%) | Questions |
|--------|---------|-------|--------|--------|-----------|
| DyG-RAG | timeqa | easy | 75.00 | 82.50 | 8 |
| DyG-RAG | timeqa | hard | 60.00 | 71.25 | 8 |
| **DyG-RAG** | **timeqa** | **overall** | **67.50** | **76.88** | **16** |
| HippoRAG | timeqa | easy | 83.33 | 88.75 | 8 |
| HippoRAG | timeqa | hard | 62.50 | 75.00 | 8 |
| **HippoRAG** | **timeqa** | **overall** | **72.92** | **81.88** | **16** |
```

## 🔍 故障排除

### 问题: 找不到结果文件

确保结果文件路径正确:
```
QA-result/<dataset>/<method>/results.json
```

### 问题: 答案提取失败

检查输出中的答案格式是否包含加粗实体标记 `**实体名**`。

### 问题: 缺少 ground truth

确保数据集文件存在:
```
dataset/<dataset>/test_processed.json
```

### 问题: VLLM 连接失败

1. 检查 VLLM 服务是否启动:
   ```bash
   curl http://localhost:8000/v1/models
   ```

2. 检查端口是否被占用:
   ```bash
   lsof -i :8000
   ```

3. 查看 VLLM 日志排查错误

### 问题: GPU 内存不足

1. 减少 tensor_parallel_size
2. 降低 max_model_len
3. 使用量化模型(如 int8/int4)
4. 调整 gpu_memory_utilization 参数

## 📦 依赖项

主要依赖包括:

- **深度学习框架**: `torch>=2.0.0`, `transformers>=4.35.0`
- **向量存储**: `faiss-gpu`, `hnswlib`, `nano-vectordb`
- **图计算**: `networkx`, `igraph`, `neo4j`, `graspologic`
- **LLM 推理**: `vllm>=0.8.4`, `openai`
- **文本处理**: `sentence-transformers`, `tiktoken`
- **其他**: `pandas`, `numpy`, `scikit-learn`, `pyyaml`

完整依赖列表见 [requirements.txt](requirements.txt)。

安装依赖:
```bash
pip install -r requirements.txt
```
