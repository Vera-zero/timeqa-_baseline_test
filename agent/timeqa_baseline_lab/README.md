TimeQA baseline 实验项目，已集成并迁移以下方法：
- ReAct（迁移自 `ReAct-master`）
- QAaP（迁移自 `qaap-main`）
- MRAG（迁移自 `MRAG-master`）

详细迁移核对请见：`migration_report.md`

支持能力：
- 骨干模型可配置（默认 `Qwen/Qwen3-4B-Instruct-2507`）
- 支持本地模型与远程模型二选一（`model.provider: local/remote`）
- 无上下文：`zero_shot`、`zero_shot_cot`
- 有上下文：`rag_cot`、`react`、`qaap`、`mrag`
- chunk 参数可配置：`chunk_size`、`chunk_overlap`、`min_chunk_size`
- 检索器可配置：Contriever（默认 `facebook/contriever`），`top_k` 可配置
- 默认前 100 题，可切全量
- 断点续跑（每题写盘，自动跳过已完成 idx）

## 本地/远程模型（二选一）
- 在 `configs/default.yaml` 设置：
  - `model.provider: local` 使用本地 HF 模型（读取 `model.model_name`）
  - `model.provider: remote` 使用远程 Chat Completions 接口（读取 `model.model` + `model.base_url`）
- 远程模式需要设置环境变量（默认）：`DEEPSEEK_API_KEY`
- `model.max_retries`、`model.timeout` 仅在远程模式生效

## 批量规则
- `model.batch_size` 支持按题批量请求（仅非交互策略）
- 非交互策略：`zero_shot`、`zero_shot_cot`、`rag_cot`（可批量）
- 交互策略：`react`、`qaap`、`mrag`（自动强制 `batch_size=1`）

## 迁移代码位置
- ReAct 迁移实现：`src/timeqa_baseline_lab/migrated/react_agent.py`
- QAaP 迁移实现：`src/timeqa_baseline_lab/migrated/qaap_agent.py`
- QAaP 工具函数迁移：`src/timeqa_baseline_lab/migrated/qaap_utils.py`
- MRAG 迁移实现：`src/timeqa_baseline_lab/migrated/mrag_agent.py`
- MRAG 时间重排工具迁移：`src/timeqa_baseline_lab/migrated/mrag_utils.py`
- MRAG 原始提示词函数迁移：`src/timeqa_baseline_lab/migrated/mrag_prompts.py`
- 迁移的提示词资源：
  - `src/timeqa_baseline_lab/assets/qaap_timeqa_prompt.json`
  - `src/timeqa_baseline_lab/assets/react_prompts_naive.json`

## 方法独立配置
- 总配置：`configs/default.yaml`
- 方法配置目录：`configs/methods/`
  - `zero_shot.yaml`
  - `zero_shot_cot.yaml`
  - `rag_cot.yaml`
  - `react.yaml`
  - `qaap.yaml`
  - `mrag.yaml`
- 运行时会根据 `run.strategy` 自动加载 `configs/methods/<strategy>.yaml`
- 如需关闭自动加载：
  - 单策略：`python -m timeqa_baseline_lab.run --disable_strategy_config ...`
  - 全策略：`python -m timeqa_baseline_lab.run_all --disable_strategy_config ...`

## 目录
- `configs/default.yaml`: 总配置
- `configs/methods/*.yaml`: 各方法独立配置
- `migration_report.md`: 迁移一致性与可运行性核对报告
- `src/timeqa_baseline_lab/run.py`: 单策略入口
- `src/timeqa_baseline_lab/run_all.py`: 全策略批跑
- `outputs/*.jsonl`: 每题结果
- `outputs/*_metrics.json`: 汇总指标
- `cache/retriever/`: chunk 与检索 embedding 缓存

## 安装
```bash
cd timeqa_baseline_lab
pip install -r requirements.txt
pip install -e .
```

## 运行
单策略：
```bash
python -m timeqa_baseline_lab.run --config configs/default.yaml --strategy react
```

全策略：
```bash
python -m timeqa_baseline_lab.run_all --config configs/default.yaml
```

全量测试（不截断）：
```bash
python -m timeqa_baseline_lab.run --config configs/default.yaml --strategy mrag --max_questions -1
```

## 评估与结果保存
- 每题结果会保存在 `outputs/<run_name>_<strategy>.jsonl`
- 每次运行会保存配置快照：`outputs/<run_name>_<strategy>_config_used.json`
- 每题字段包含：`prediction`、`raw_output`、`em`、`f1`、`substring_recall`、`retrieved`、`trace`
- 汇总结果保存在 `outputs/<run_name>_<strategy>_metrics.json`，包含：`avg_em`、`avg_f1`、`avg_recall`
- EM/F1 评估口径参考 `qaap-main/utils.py` 与 ReAct 常用 token-F1 规范化

## 单元测试
- 测试目录：`tests/unit/`
- 测试文件：`tests/unit/test_strategies_single_question.py`
- 覆盖方法：`zero_shot`、`zero_shot_cot`、`rag_cot`、`react`、`qaap`、`mrag`
- 约束：仅测试 1 道题，但检索语料使用全量 corpus（通过测试内 full-corpus retriever fixture）

运行：
```bash
python -m unittest discover -s tests/unit -p "test_*.py" -v
```
