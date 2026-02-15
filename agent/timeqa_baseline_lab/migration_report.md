# Migration Report (TimeQA Baseline Lab)

更新日期：2026-02-05

## 1. 迁移目标

本项目将以下三个外部方法迁移到统一实验框架 `timeqa_baseline_lab` 中，并保证：
- 方法主流程保持一致（核心思想与执行阶段不变）
- 在同一数据/检索/评估管线下可运行、可对比
- 支持断点续跑、结果留痕、指标统一统计（EM/F1）

源仓库：
- `C:/Users/19924/Desktop/agent/ReAct-master`
- `C:/Users/19924/Desktop/agent/qaap-main`
- `C:/Users/19924/Desktop/agent/MRAG-master`

---

## 2. 文件映射（源实现 -> 迁移实现）

### ReAct
- `ReAct-master/hotpotqa.ipynb`（ReAct loop）
- `ReAct-master/prompts/prompts_naive.json`（few-shot prompt）

迁移到：
- `src/timeqa_baseline_lab/migrated/react_agent.py`
- `src/timeqa_baseline_lab/assets/react_prompts_naive.json`

### QAaP
- `qaap-main/main.py`（parse/search/extract 主流程）
- `qaap-main/utils.py`（时间匹配/代码抽取/指标工具）
- `qaap-main/prompts/timeqa.json`（timeqa 提示词）

迁移到：
- `src/timeqa_baseline_lab/migrated/qaap_agent.py`
- `src/timeqa_baseline_lab/migrated/qaap_utils.py`
- `src/timeqa_baseline_lab/assets/qaap_timeqa_prompt.json`

### MRAG
- `MRAG-master/metriever.py`（候选检索+重排核心逻辑）
- `MRAG-master/prompts.py`（keyword/QFS/combiner 等提示词函数）
- `MRAG-master/utils.py`（部分辅助逻辑）

迁移到：
- `src/timeqa_baseline_lab/migrated/mrag_agent.py`
- `src/timeqa_baseline_lab/migrated/mrag_utils.py`
- `src/timeqa_baseline_lab/migrated/mrag_prompts.py`

---

## 3. 方法一致性检查结论

### 3.1 ReAct

已保留的一致性：
- Thought -> Action -> Observation 的迭代流程
- Action 类型：`Search[]` / `Lookup[]` / `Finish[]`
- few-shot 示例驱动行为（来自原 `prompts_naive.json`）

适配改动（为统一实验框架）：
- 原版通过在线 Wikipedia env 交互；迁移版使用本地 Contriever 检索结果模拟 `Search/Lookup` 观察。
- 改动原因：当前实验要求固定语料库 + 可批量评测 + 可复现。

判定：**核心方法一致，执行环境做了可复现适配**。

### 3.2 QAaP

已保留的一致性：
- `Question parsing` -> `Search` -> `Extract information` -> `code execution/time matching`
- 使用 python 代码块 (`query`, `answer_key`, `information.append`) 作为中间程序表示
- 时间匹配逻辑（`calc_time_iou`）沿用原思想

适配改动：
- 原版可直接在线搜索 Wikipedia；迁移版改为在本地检索结果中执行同样的程序化抽取。
- 增加安全执行边界（限制执行环境）以保证批跑稳定性。

判定：**主流程一致，数据来源从在线检索切换为本地检索**。

### 3.3 MRAG

已保留的一致性：
- 多阶段思路：候选召回 -> keyword/temporal 信号融合 -> QFS 摘要 -> combiner 最终回答
- 保留 `prompts.py` 中关键 prompt 函数（`get_keyword_prompt/get_QFS_prompt/combiner`）
- 保留时间关系判别与 temporal coefficient 的核心逻辑

适配改动：
- 原仓库包含 vLLM、多模型 reranker、复杂外部依赖；迁移版采用本地 Contriever 相似度 + temporal/keyword 混合打分。
- 目的：在你的统一基线框架下稳定运行并支持参数可配置。

判定：**方法框架一致，工程实现做了轻量化适配**。

---

## 4. 可运行性与结果留痕检查

已实现：
- 断点续跑：每题写入 `jsonl`，按 `idx` 自动跳过已完成样本
- 每题输出保留：
  - `prediction`
  - `raw_output`
  - `trace`
  - `retrieved`
  - `em` / `f1` / `substring_recall`
- 汇总输出：
  - `outputs/<run_name>_<strategy>_metrics.json`
- 配置快照输出：
  - `outputs/<run_name>_<strategy>_config_used.json`

评估口径：
- EM/F1 参考 `qaap-main/utils.py` 与 ReAct 常用 token-F1 规范化流程

---

## 5. 方法独立配置化（已完成）

全局配置：
- `configs/default.yaml`

方法独立配置：
- `configs/methods/zero_shot.yaml`
- `configs/methods/zero_shot_cot.yaml`
- `configs/methods/rag_cot.yaml`
- `configs/methods/react.yaml`
- `configs/methods/qaap.yaml`
- `configs/methods/mrag.yaml`

加载机制：
- 指定 `run.strategy` 后自动加载 `configs/methods/<strategy>.yaml`
- 命令行可关闭：`--disable_strategy_config`

参数分发机制：
- `run.strategy_params` 会按函数签名过滤后注入对应方法，避免无效参数导致崩溃

---

## 6. 目前边界与建议

边界：
- 本迁移不是逐行 1:1 复刻原仓库全部外部依赖（如在线 Wikipedia env、vLLM 多模型重排全栈）。
- 目标是“方法一致 + 在统一本地实验框架下稳定可复现”。

建议：
- 若后续做论文复现实验，可在此基础上新增“strict 模式”：逐步恢复原方法的外部组件并做 A/B 对照。

---

## 7. 快速核验命令

```bash
# 单策略（自动加载方法配置）
python -m timeqa_baseline_lab.run --config configs/default.yaml --strategy react

# 全策略
python -m timeqa_baseline_lab.run_all --config configs/default.yaml

# 关闭方法配置自动加载（调试用）
python -m timeqa_baseline_lab.run --config configs/default.yaml --strategy mrag --disable_strategy_config
```

---

## 8. 逐方法逐函数对照表（原函数 -> 新函数）

### 8.1 ReAct

| 原函数名（位置） | 新函数名（位置） | 一致/适配说明 |
|---|---|---|
| `webthink`（`ReAct-master/hotpotqa.ipynb`） | `run_react`（`src/timeqa_baseline_lab/migrated/react_agent.py`） | **一致**：保留 Thought/Action/Observation 迭代主循环；**适配**：执行环境从在线 WikiEnv 改为本地检索语料。 |
| `step`（`ReAct-master/hotpotqa.ipynb`） | `run_react` 内动作分支处理 | **一致**：按步执行并写回 Observation；**适配**：不再调用 Gym env，改为本地检索动作仿真。 |
| `WikiEnv.search_step`（`ReAct-master/wikienv.py`） | `retriever.search` + `run_react` 中 Search 分支 | **适配**：保留“Search -> 获取页面观察”的语义，数据源改为本地 chunk 检索。 |
| `WikiEnv.construct_lookup_list` / `WikiEnv.step` 的 `lookup[]` 逻辑 | `run_react` 中 Lookup 分支（按句检索） | **一致**：保留 `Lookup[keyword]` 的逐条返回行为；**适配**：在当前检索到的 chunk 文本上执行。 |

### 8.2 QAaP

| 原函数名（位置） | 新函数名（位置） | 一致/适配说明 |
|---|---|---|
| `qaap`（`qaap-main/main.py`） | `run_qaap`（`src/timeqa_baseline_lab/migrated/qaap_agent.py`） | **一致**：Parse -> Search -> Extract -> Code Exec -> Time Match 主流程保持不变。 |
| `post`（`qaap-main/main.py`） | `HFGenerator.generate`（`src/timeqa_baseline_lab/llm.py`） | **适配**：调用接口从 OpenAI API 改为本地 HF 模型推理。 |
| `search`（`qaap-main/search_wiki.py`） | `retriever.search`（在 `run_qaap` 中调用） | **适配**：检索源从在线 Wikipedia 改为本地 TimeQA 语料库。 |
| `create_context_slices`（`qaap-main/utils.py`） | `create_context_slices`（`src/timeqa_baseline_lab/migrated/qaap_utils.py`） | **一致**：分片逻辑保持一致。 |
| `extract_code_from_string`（`qaap-main/utils.py`） | `extract_code_from_string`（`src/timeqa_baseline_lab/migrated/qaap_utils.py`） | **一致**：Python fenced code 抽取逻辑保持一致。 |
| `calc_time_iou`（`qaap-main/utils.py`） | `calc_time_iou`（`src/timeqa_baseline_lab/migrated/qaap_utils.py`） | **基本一致**：时间匹配核心逻辑保留；**适配**：执行上下文加入安全边界。 |
| `extract_answer`（`qaap-main/utils.py`） | `extract_answer`（`src/timeqa_baseline_lab/migrated/qaap_utils.py`） | **一致**：从结构化 `information` 提取答案逻辑保持一致。 |

### 8.3 MRAG

| 原函数名（位置） | 新函数名（位置） | 一致/适配说明 |
|---|---|---|
| 检索与重排主流程（`MRAG-master/metriever.py` 主循环） | `run_mrag`（`src/timeqa_baseline_lab/migrated/mrag_agent.py`） | **一致**：候选召回 -> temporal/keyword 融合 -> QFS -> combine；**适配**：工程栈简化为本地可复现版本。 |
| `get_keyword_prompt`（`MRAG-master/prompts.py`） | `get_keyword_prompt`（`src/timeqa_baseline_lab/migrated/mrag_prompts.py`） | **一致（同名迁移）**：提示词函数直接迁移。 |
| `get_QFS_prompt`（`MRAG-master/prompts.py`） | `get_QFS_prompt`（`src/timeqa_baseline_lab/migrated/mrag_prompts.py`） | **一致（同名迁移）**：QFS 提示词函数直接迁移。 |
| `combiner`（`MRAG-master/prompts.py`） | `combiner`（`src/timeqa_baseline_lab/migrated/mrag_prompts.py`） | **一致（同名迁移）**：最终回答融合提示词函数直接迁移。 |
| `get_spline_function`（`MRAG-master/metriever.py`） | `get_spline_function`（`src/timeqa_baseline_lab/migrated/mrag_utils.py`） | **基本一致**：保留 temporal coefficient 核心思想；**适配**：去除 scipy 依赖，使用轻量实现。 |
| `get_temporal_coeffs`（`MRAG-master/metriever.py`） | `get_temporal_coeffs`（`src/timeqa_baseline_lab/migrated/mrag_utils.py`） | **一致**：时间约束下的句子系数计算逻辑保留。 |
| `year_identifier` / `replace_dates` / `expand_year_range`（`MRAG-master/utils.py`） | 同名函数（`src/timeqa_baseline_lab/migrated/mrag_utils.py`） | **一致（同名迁移）**：年份抽取与范围展开逻辑保留。 |
| `call_pipeline`（`MRAG-master/utils.py`） | `HFGenerator.generate`（`src/timeqa_baseline_lab/llm.py`） | **适配**：推理后端从 vLLM/OpenAI 路径统一为本地 HF 模型。 |

### 8.4 统一框架补充（非源仓库函数）

| 新函数名（位置） | 作用 | 说明 |
|---|---|---|
| `run_experiment`（`src/timeqa_baseline_lab/runner.py`） | 统一实验执行入口 | 统一六种策略执行、断点续跑、结果落盘、指标汇总。 |
| `em_f1`（`src/timeqa_baseline_lab/evaluation.py`） | 统一 EM/F1 计算 | 口径对齐 QAaP/ReAct 常见实现。 |
| `search_with_scores`（`src/timeqa_baseline_lab/retriever.py`） | 返回检索分数 | 为 MRAG 混合重排提供语义分数输入。 |
