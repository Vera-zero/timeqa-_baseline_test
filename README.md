# ETE-Graph QA 评估工具

本项目包含用于评估问答（QA）系统结果的工具和脚本。

## 📁 项目结构

```
ETE-Graph/
├── dataset/                    # 数据集文件
│   ├── timeqa/
│   │   └── test_processed.json
│   └── tempreason/
├── QA-result/                  # QA系统输出结果
│   ├── timeqa/
│   │   ├── DyG-RAG/
│   │   │   └── results.json
│   │   └── HippoRAG/
│   │       └── results.json
│   └── tempreason/
├── evaluate_qa_results.py      # 评估脚本
└── evaluation_results/         # 评估输出（自动生成）
    ├── results_table.md
    └── results_table.csv
```

## 🚀 快速开始

### 评估所有QA结果

```bash
cd /workspace/ETE-Graph
python evaluate_qa_results.py
```

评估完成后，结果将保存在 `evaluation_results/` 目录下：
- `results_table.md` - Markdown格式的评估表格
- `results_table.csv` - CSV格式的评估表格

### 查看评估结果

```bash
cat evaluation_results/results_table.md
```

## 📊 评估指标

脚本计算以下指标：

- **EM (Exact Match)**: 精确匹配率（%）- 预测答案与标准答案完全匹配的比例
- **F1 Score**: Token级别的F1分数（%）- 衡量预测答案与标准答案的重叠程度

## 🔧 高级用法

### 仅评估特定数据集

```bash
python evaluate_qa_results.py --dataset timeqa
```

### 仅评估特定方法

```bash
python evaluate_qa_results.py --method HippoRAG
```

### 指定输出格式

```bash
# 仅生成Markdown
python evaluate_qa_results.py --output-format markdown

# 仅生成CSV
python evaluate_qa_results.py --output-format csv

# 同时生成两种格式（默认）
python evaluate_qa_results.py --output-format both
```

### 自定义路径

```bash
python evaluate_qa_results.py \
    --qa-result-dir /path/to/QA-result \
    --dataset-dir /path/to/dataset \
    --output-dir /path/to/output
```

### 查看所有选项

```bash
python evaluate_qa_results.py --help
```

## 📝 支持的结果格式

评估脚本支持两种结果格式：

### 1. DyG-RAG 格式（JSON）

```json
{
  "metadata": {
    "dataset": "timeqa",
    "total_questions": 16
  },
  "results": [
    {
      "question_idx": 0,
      "question": "Which team did...",
      "answer": "**Answer:** ... **Thai Port FC**. **Justification:** ..."
    }
  ]
}
```

**答案提取规则**：从 `answer` 字段中的 `**Answer:**` 部分提取第一个加粗实体

### 2. 其他方法格式（JSONL）

如 HippoRAG 等方法使用 JSONL 格式（每行一个JSON对象）：

```json
{"id":0,"question":"Which team did...","answer":["Port F.C"],"level":"hard","output":"Based on... **Thai Port FC Authority of Thailand** ..."}
```

**答案提取规则**：从 `output` 字段的第一句话中提取加粗实体（`**实体名**`）

## 📈 评估结果示例

生成的Markdown表格示例：

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

## 🔄 添加新的QA方法

要评估新的QA方法：

1. 在对应数据集目录下创建方法文件夹：
   ```bash
   mkdir -p QA-result/timeqa/YourMethod
   ```

2. 将结果保存为 `results.json`，支持以下格式之一：
   - JSON格式（类似DyG-RAG）
   - JSONL格式（类似HippoRAG）

3. 确保结果包含必要字段：
   - `question`: 问题文本
   - `output` 或 `answer`: 模型输出（包含加粗实体 `**实体名**`）
   - `level`（可选）: 难度级别（easy/hard）

4. 运行评估脚本：
   ```bash
   python evaluate_qa_results.py
   ```

## 🛠️ 依赖项

脚本使用 Python 标准库，主要依赖：

- `unidecode` - 用于文本规范化

安装依赖：
```bash
pip install unidecode
```

## 📚 评估指标说明

### 文本规范化

在计算EM和F1之前，答案会经过以下规范化处理：
- 转为小写
- 移除冠词（a, an, the）
- 移除标点符号
- 移除Unicode字符
- 规范化空格

### F1计算方法

F1分数基于Token级别的重叠计算：
```
Precision = 匹配的tokens数 / 预测答案的tokens数
Recall = 匹配的tokens数 / 标准答案的tokens数
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

对于每个问题，会计算预测答案与所有标准答案的F1分数，取最大值。

## 🐛 故障排除

### 问题：找不到结果文件

确保结果文件路径正确：
```
QA-result/<dataset>/<method>/results.json
```

### 问题：答案提取失败

检查输出中的答案格式是否包含加粗实体标记 `**实体名**`。

### 问题：缺少ground truth

确保数据集文件存在：
```
dataset/<dataset>/test_processed.json
```

## 📄 许可证

本项目遵循与ETE-Graph相同的许可证。

## 🤝 贡献

欢迎提交Issue和Pull Request！

---

**最后更新**: 2026-02-13
