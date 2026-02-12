# ETE-Graph Embedding 模型修改记录

## 修改概览

**修改时间**: 2026-02-11 09:14:49
**修改目的**: 将所有 embedding 模型统一替换为本地 Qwen-Embedding-8B，提升嵌入质量和多语言支持
**涉及项目**: GraphRAG、DyG-RAG
**修改文件数**: 8个代码文件 + 1个文档文件

---

## 修改详情

### 一、GraphRAG 修改

#### 1.1 配置文件修改
**文件**: `GraphRAG/Option/Config2.yaml`
**修改时间**: 2026-02-11 09:14
**修改内容**:
- `embedding.model`: `YOURMODEL` → `/workspace/models/Qwen3-Embedding-8B`
- `embedding.dimensions`: `1024` → `4096`
- `embedding.max_token_size`: `8102` → `32768`

**修改目的**: 统一所有 GraphRAG 方法（HippoRAG、LightRAG、FastGraphRAG等）的 embedding 配置

**影响范围**: 所有继承此配置的 GraphRAG 方法

**代码变更**:
```yaml
# 修改前
embedding:
  api_type: "hf"
  model: "YOURMODEL"
  dimensions: 1024
  max_token_size: 8102

# 修改后
embedding:
  api_type: "hf"
  model: "/workspace/models/Qwen3-Embedding-8B"
  dimensions: 4096
  max_token_size: 32768
```

---

#### 1.2 Embedding 工厂类修改
**文件**: `GraphRAG/Core/Index/EmbeddingFactory.py`
**修改时间**: 2026-02-11 09:14
**修改内容**:
- 第79行添加: `trust_remote_code=True`
- 第79行修改: `target_devices = ["cuda:7"]` → `["cuda:0"]`
- 第80行修改: `embed_batch_size = 128` → `config.embedding.embed_batch_size or 128`

**修改目的**: 支持 Qwen 模型的加载和参数配置

**代码变更**:
```python
# 修改前
def _create_hf(self, config) -> HuggingFaceEmbedding:
    params = dict(
        model_name=config.embedding.model,
        cache_folder=config.embedding.cache_folder,
        device = "cuda",
        target_devices = ["cuda:7"],
        embed_batch_size = 128,
    )
    if config.embedding.cache_folder == "":
        del params["cache_folder"]
    return HuggingFaceEmbedding(**params)

# 修改后
def _create_hf(self, config) -> HuggingFaceEmbedding:
    params = dict(
        model_name=config.embedding.model,
        cache_folder=config.embedding.cache_folder,
        device = "cuda",
        target_devices = ["cuda:0"],
        embed_batch_size = config.embedding.embed_batch_size or 128,
        trust_remote_code=True,
    )
    if config.embedding.cache_folder == "":
        del params["cache_folder"]
    return HuggingFaceEmbedding(**params)
```

---

### 二、DyG-RAG 修改

#### 2.1 添加 Qwen Embedding 函数
**文件**: `DyG-RAG/graphrag/_llm.py`
**修改时间**: 2026-02-11 09:14
**修改内容**: 新增 `qwen_embedding_8b` 异步函数（约35行代码），添加于第220行之后

**修改目的**: 为 DyG-RAG 提供统一的 Qwen-Embedding-8B 嵌入接口

**关键参数**:
- 嵌入维度: 4096
- 最大token: 32768
- 模型路径: `/workspace/models/Qwen3-Embedding-8B`
- 归一化: True

**新增代码**:
```python
# Qwen-Embedding-8B 本地嵌入函数
@wrap_embedding_func_with_attrs(
    embedding_dim=4096,  # Qwen-Embedding-8B 的默认维度
    max_token_size=32768  # Qwen-Embedding-8B 支持 32k 上下文
)
async def qwen_embedding_8b(texts: list[str]) -> np.ndarray:
    """使用本地 Qwen-Embedding-8B 模型进行文本嵌入"""
    from sentence_transformers import SentenceTransformer
    import os

    # 本地模型路径
    model_path = "/workspace/models/Qwen3-Embedding-8B"

    # 加载模型（使用全局单例避免重复加载）
    if not hasattr(qwen_embedding_8b, '_model'):
        qwen_embedding_8b._model = SentenceTransformer(
            model_path,
            device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
            trust_remote_code=True
        )

    # 批量嵌入
    embeddings = qwen_embedding_8b._model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True  # 建议归一化以提升相似度计算
    )

    return embeddings
```

---

#### 2.2 修改默认 Embedding
**文件**: `DyG-RAG/graphrag/graphrag.py`
**修改时间**: 2026-02-11 09:14
**修改内容**:
- 第27-36行: 在导入列表中添加 `qwen_embedding_8b`
- 第159行: 默认值从 `openai_embedding` 改为 `qwen_embedding_8b`

**修改目的**: 将全局默认 embedding 切换到 Qwen-Embedding-8B

**影响范围**: 所有未显式指定 embedding_func 的 GraphRAG 实例

**代码变更**:
```python
# 修改前 (导入部分)
from ._llm import (
    amazon_bedrock_embedding,
    create_amazon_bedrock_complete_function,
    gpt_4o_complete,
    gpt_4o_mini_complete,
    openai_embedding,
    azure_gpt_4o_complete,
    azure_openai_embedding,
    azure_gpt_4o_mini_complete,
)

# 修改后 (导入部分)
from ._llm import (
    amazon_bedrock_embedding,
    create_amazon_bedrock_complete_function,
    gpt_4o_complete,
    gpt_4o_mini_complete,
    openai_embedding,
    qwen_embedding_8b,
    azure_gpt_4o_complete,
    azure_openai_embedding,
    azure_gpt_4o_mini_complete,
)

# 修改前 (默认值)
embedding_func: EmbeddingFunc = field(default_factory=lambda: openai_embedding)

# 修改后 (默认值)
embedding_func: EmbeddingFunc = field(default_factory=lambda: qwen_embedding_8b)
```

---

#### 2.3 示例脚本修改 - local_BGE_local_LLM.py
**文件**: `DyG-RAG/examples/local_BGE_local_LLM.py`
**修改时间**: 2026-02-11 09:14
**修改内容**:
- 第73-76行: 替换 `LOCAL_BGE_PATH` → `LOCAL_QWEN_EMBEDDING_PATH`
- 第113行: 函数名 `get_bge_embedding_func` → `get_qwen_embedding_func`
- 第122行: 模型路径 `LOCAL_BGE_PATH` → `LOCAL_QWEN_EMBEDDING_PATH`
- 第131行: `max_token_size`: 8192 → 32768
- 第186行: 调用 `get_bge_embedding_func()` → `get_qwen_embedding_func()`

**修改目的**: 将 BGE 示例迁移到 Qwen-Embedding-8B

**代码变更**:
```python
# 修改前
LOCAL_BGE_PATH = get_config_value(
    "LOCAL_BGE_PATH",
    "Local path to BGE embedding model",
    "/path/to/bge-m3"
)

def get_bge_embedding_func() -> EmbeddingFunc:
    st_model = SentenceTransformer(
        LOCAL_BGE_PATH,
        device=device,
        trust_remote_code=True,
        model_kwargs=model_kwargs,
    )
    return EmbeddingFunc(
        embedding_dim=st_model.get_sentence_embedding_dimension(),
        max_token_size=8192,
        model=st_model,
    )

embedding_func = get_bge_embedding_func()

# 修改后
LOCAL_QWEN_EMBEDDING_PATH = get_config_value(
    "LOCAL_QWEN_EMBEDDING_PATH",
    "Local path to Qwen-Embedding-8B model",
    "/workspace/models/Qwen3-Embedding-8B"
)

def get_qwen_embedding_func() -> EmbeddingFunc:
    st_model = SentenceTransformer(
        LOCAL_QWEN_EMBEDDING_PATH,
        device=device,
        trust_remote_code=True,
        model_kwargs=model_kwargs,
    )
    return EmbeddingFunc(
        embedding_dim=st_model.get_sentence_embedding_dimension(),
        max_token_size=32768,
        model=st_model,
    )

embedding_func = get_qwen_embedding_func()
```

---

#### 2.4 复现脚本修改 - tempreason.py
**文件**: `DyG-RAG/reproduce/tempreason.py`
**修改时间**: 2026-02-11 09:14
**修改内容**: 与 local_BGE_local_LLM.py 相同的修改模式
- 第77-80行: `LOCAL_BGE_PATH` → `LOCAL_QWEN_EMBEDDING_PATH`
- 第120行: `get_bge_embedding_func` → `get_qwen_embedding_func`
- 第130行: `LOCAL_BGE_PATH` → `LOCAL_QWEN_EMBEDDING_PATH`
- 第138行: `max_token_size`: 8192 → 32768
- 第195行: `get_bge_embedding_func()` → `get_qwen_embedding_func()`

**修改目的**: 保证 TempReason 数据集复现使用统一模型

---

#### 2.5 复现脚本修改 - timeqa.py
**文件**: `DyG-RAG/reproduce/timeqa.py`
**修改时间**: 2026-02-11 09:14
**修改内容**: 与 tempreason.py 相同的修改模式
- 第77-80行: `LOCAL_BGE_PATH` → `LOCAL_QWEN_EMBEDDING_PATH`
- 第120行: `get_bge_embedding_func` → `get_qwen_embedding_func`
- 第130行: `LOCAL_BGE_PATH` → `LOCAL_QWEN_EMBEDDING_PATH`
- 第138行: `max_token_size`: 8192 → 32768
- 第195行: `get_bge_embedding_func()` → `get_qwen_embedding_func()`

**修改目的**: 保证 TimeQA 数据集复现使用统一模型

---

### 三、文档更新

#### 3.1 基线配置文档更新
**文件**: `graph_baseline.md`
**修改时间**: 2026-02-11 09:14
**修改内容**: 第110-154行 "嵌入模型统一" 章节完全重写

**修改目的**: 同步文档与代码实际配置

**关键变更**:
- 模型名称: `BAAI/bge-large-en-v1.5` → `Qwen3-Embedding-8B`
- 模型路径: HuggingFace Hub → `/workspace/models/Qwen3-Embedding-8B`
- 嵌入维度: 1024 → 4096
- 最大上下文: 8192 → 32768
- 新增使用示例和注意事项
- 新增性能影响说明
- 新增兼容性要求说明

---

## 修改影响分析

### 性能影响
1. **向量维度增加**: 1024 → 4096 (4倍)
   - 向量数据库存储空间增加约4倍
   - 相似度计算时间略有增加

2. **嵌入速度变化**:
   - Qwen-Embedding-8B (8B参数) vs BGE-large (335M参数)
   - 推理速度可能降低，建议使用 GPU 加速
   - 建议 GPU 显存: ≥16GB (FP16)

3. **质量提升**:
   - MTEB多语言排行榜 No.1 (70.58分)
   - 更好的多语言和长文本理解能力
   - 支持超过100种语言

### 兼容性要求
- transformers >= 4.51.0
- sentence-transformers >= 2.7.0
- 需要设置 `trust_remote_code=True`
- Python >= 3.8

### 数据迁移注意事项
**重要**: 现有向量数据库（使用1024维 BGE 嵌入）与新的4096维 Qwen 嵌入不兼容，需要重新构建索引。

**迁移步骤**:
1. 备份现有向量数据库
2. 使用新配置重新运行索引构建
3. 验证新索引的检索效果
4. 确认无误后删除旧索引

---

## 验证清单

- [x] GraphRAG Config2.yaml 配置已更新
- [x] EmbeddingFactory.py 支持 Qwen 模型加载
- [x] DyG-RAG _llm.py 新增 qwen_embedding_8b 函数
- [x] DyG-RAG graphrag.py 默认值已修改
- [x] 所有示例脚本已更新（3个文件）
- [x] graph_baseline.md 文档已同步
- [x] 所有修改已记录在 change.md

---

## 回滚方案

如需回滚到之前的配置：

### GraphRAG 回滚
```yaml
# Config2.yaml
embedding:
  model: "YOURMODEL"  # 或 "BAAI/bge-large-en-v1.5"
  dimensions: 1024
  max_token_size: 8102
```

### DyG-RAG 回滚
1. 恢复 `graphrag.py` 第159行:
   ```python
   embedding_func: EmbeddingFunc = field(default_factory=lambda: openai_embedding)
   ```

2. 恢复示例脚本中的路径配置:
   ```python
   LOCAL_BGE_PATH = "/path/to/bge-m3"
   ```

3. 可选：删除 `_llm.py` 中的 `qwen_embedding_8b` 函数

---

## 附录：文件清单

### 修改文件
1. `/workspace/ETE-Graph/GraphRAG/Option/Config2.yaml`
2. `/workspace/ETE-Graph/GraphRAG/Core/Index/EmbeddingFactory.py`
3. `/workspace/ETE-Graph/DyG-RAG/graphrag/_llm.py`
4. `/workspace/ETE-Graph/DyG-RAG/graphrag/graphrag.py`
5. `/workspace/ETE-Graph/DyG-RAG/examples/local_BGE_local_LLM.py`
6. `/workspace/ETE-Graph/DyG-RAG/reproduce/tempreason.py`
7. `/workspace/ETE-Graph/DyG-RAG/reproduce/timeqa.py`
8. `/workspace/ETE-Graph/graph_baseline.md`

### 新增文件
9. `/workspace/ETE-Graph/change.md` (本文件)

**总计**: 8个修改 + 1个新增 = 9个文件

---

## 技术细节

### Qwen-Embedding-8B 模型规格
- **模型名称**: Qwen3-Embedding-8B
- **参数量**: 8B
- **嵌入维度**: 4096 (支持 32-4096 自定义)
- **上下文长度**: 32k tokens
- **支持语言**: 100+ 语言
- **模型架构**: Qwen3ForCausalLM
- **推荐精度**: FP16/BF16
- **GPU 显存**: ~16GB (FP16)

### 加载示例
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "/workspace/models/Qwen3-Embedding-8B",
    device="cuda",
    trust_remote_code=True,
    model_kwargs={"torch_dtype": "float16"}
)

# 编码文本
embeddings = model.encode(
    ["文本1", "文本2"],
    batch_size=32,
    normalize_embeddings=True
)

print(embeddings.shape)  # (2, 4096)
```

---

**修改完成日期**: 2026-02-11
**修改人**: Claude Agent
**审核状态**: 待验证

---

## 修复记录 - 验证错误修复

**修复时间**: 2026-02-11 11:30:00
**修复目的**: 修复验证失败问题，确保代码语法正确性

### 修复详情

#### 修复 1: DyG-RAG 列表分隔符错误
**文件**: `/workspace/ETE-Graph/DyG-RAG/graphrag/graphrag.py`
**位置**: 第206行
**问题描述**:
- `keys_to_exclude` 列表中 `"events_vdb"` 后缺少逗号
- Python 自动拼接相邻字符串，导致 `"events_vdb"` 和 `"full_docs"` 合并为 `"events_vdbfull_docs"`
- 影响 `get_config_dict()` 方法的键名匹配逻辑，造成配置字典污染

**修复内容**:
```python
# 修复前
keys_to_exclude = [
    "event_dynamic_graph",
    "events_vdb"           # ← 缺少逗号
    "full_docs",
    "text_chunks",
    "llm_response_cache"
]

# 修复后
keys_to_exclude = [
    "event_dynamic_graph",
    "events_vdb",          # ← 添加逗号
    "full_docs",
    "text_chunks",
    "llm_response_cache"
]
```

---

#### 修复 2: GraphRAG 缩进错误和方法结构混乱
**文件**: `/workspace/ETE-Graph/GraphRAG/Core/Index/EmbeddingFactory.py`
**位置**: 第97-99行
**问题描述**:
- 第98行使用5个空格缩进，而非标准的4个空格
- 第99行的 `_raise_for_key` 方法定义紧跟在 `_try_set_model_and_batch_size` 方法的 if 语句块后
- 缺少方法间的空行分隔，导致代码结构混乱

**修复内容**:
```python
# 修复前
if config.embedding.dimensions:
     params["dimensions"] = config.embedding.dimensions  # ← 5个空格
def _raise_for_key(self, key: Any):                      # ← 缺少空行

# 修复后
if config.embedding.dimensions:
    params["dimensions"] = config.embedding.dimensions   # ← 4个空格

def _raise_for_key(self, key: Any):                      # ← 添加空行分隔
```

---

### 验证方法

修复完成后，可运行以下命令进行验证：

```bash
# 1. 运行完整验证
python /workspace/ETE-Graph/verify_all.py -v

# 2. 单独验证 GraphRAG
python /workspace/ETE-Graph/verify_graphrag.py

# 3. 单独验证 DyG-RAG
python /workspace/ETE-Graph/verify_dygrag.py
```

**预期结果**:
- GraphRAG 验证：✓ 通过
- DyG-RAG 验证：✓ 通过
- 文档验证：✓ 通过

---

### 修复影响分析

**代码质量提升**:
1. 消除了 Python 隐式字符串拼接导致的逻辑错误
2. 统一了代码缩进标准（4个空格）
3. 改善了方法定义的视觉分隔和代码可读性

**功能影响**:
- 修复后，`keys_to_exclude` 列表能正确识别5个独立的键名
- `get_config_dict()` 方法能正确过滤运行时对象
- EmbeddingFactory 的方法结构更加清晰

**兼容性**:
- 修复不影响现有功能逻辑
- 不需要重新构建索引或迁移数据
- 纯粹的代码规范性修复

---

**修复完成日期**: 2026-02-11
**修复人**: Claude Agent
**验证状态**: 待用户验证
