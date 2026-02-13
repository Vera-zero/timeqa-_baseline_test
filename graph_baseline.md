### 基线方法
DyG-RAG 
LGraphRAG (Local search in GraphRAG) 
GGraphRAG (Global search in GraphRAG) 
HippoRAG 
LightRAG


### LLM配置统一

**原因**: 不同的LLM模型性能差异巨大，必须使用相同的模型以确保公平性。

#### 配置前后对比

| 配置项 | 方法 | 调整前 | 调整后 | 说明 |
|--------|------|--------|--------|------|
| **LLM模型** | DyG-RAG | `gpt_4o_complete` (默认) | `qwen3_32b_complete` (自定义) | 需要自定义函数 |
| | GraphRAG-master | `YOUR_MODEL` (需配置) | `qwen3-32b` | 统一模型名称 |
| **API类型** | DyG-RAG | OpenAI默认 | 本地VLLM客户端 | 切换到本地服务 |
| | GraphRAG-master | `open_llm` | `open_llm` | 保持不变 |
| **Base URL** | DyG-RAG | `https://api.openai.com/v1` | `http://localhost:8000/v1` | 指向本地VLLM |
| | GraphRAG-master | `YOUR_BASE_URL` (需配置) | `http://localhost:8000/v1` | 统一本地地址 |
| **API Key** | DyG-RAG | 需要真实key | `EMPTY` | VLLM不需要 |
| | GraphRAG-master | `YOUR_API_KEY` (需配置) | `EMPTY` | 统一为空 |
| **Temperature** | DyG-RAG | 默认值 | `0.0` | 确定性输出 |
| | GraphRAG-master | 默认值 | `0.0` | 统一为0 |
| **Max Token** | DyG-RAG | `32768` | `32768` | 保持不变 |
| | GraphRAG-master | `32768` | `32768` | 保持不变 |

**统一使用**: **Qwen3-32B** 本地部署（通过VLLM）

#### 本地部署VLLM服务

首先启动VLLM服务：

```bash
# 启动VLLM服务（建议在独立终端运行）
python -m vllm.entrypoints.openai.api_server \
    --model /workspace/models/Qwen3-32B \
    --served-model-name qwen3-32b \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --max-model-len 32768
```

**参数说明**:
- `--tensor-parallel-size`: 根据GPU数量调整（2表示使用2个GPU）
- `--max-model-len`: 最大上下文长度，设为32768以统一

#### DyG-RAG调整

创建本地LLM函数配置文件 `local_llm_config.py`:

```python
import os
from openai import AsyncOpenAI

# 配置本地VLLM客户端
local_llm_client = AsyncOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"  # VLLM不需要真实API key
)

async def qwen3_32b_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    response = await local_llm_client.chat.completions.create(
        model="qwen3-32b",
        messages=messages,
        temperature=kwargs.get("temperature", 0.0),
        max_tokens=kwargs.get("max_tokens", 4096),
    )
    return response.choices[0].message.content

# 在examples中使用
from graphrag import GraphRAG
graph_func = GraphRAG(
    working_dir=str(WORK_DIR),
    best_model_func=qwen3_32b_complete,  # 使用本地Qwen3-32B
    cheap_model_func=qwen3_32b_complete,  # 同样使用Qwen3-32B
    best_model_max_token_size=32768,
    cheap_model_max_token_size=32768,
)
```

#### GraphRAG-master框架调整

在`Config2.yaml`中统一配置：

```yaml
llm:
  api_type: "open_llm"  # 使用open_llm类型
  base_url: 'http://localhost:8000/v1'  # 本地VLLM服务地址
  model: "qwen3-32b"  # 本地部署的Qwen3-32B模型
  api_key: "EMPTY"  # VLLM不需要真实API key
  max_token: 32768  # 统一最大token数
  temperature: 0.0  # 统一温度参数以减少随机性
  timeout: 600  # 本地部署可以设置更长的超时时间
```
---

### 嵌入模型统一

**原因**: 嵌入质量直接影响检索性能，必须使用相同的嵌入模型。

#### 配置前后对比

| 配置项 | 方法 | 调整前 | 调整后 | 说明 |
|--------|------|--------|--------|------|
| **嵌入模型** | DyG-RAG | `openai_embedding` (默认) | `Qwen3-Embedding-8B` | 使用本地模型 |
| | GraphRAG-master | `BAAI/bge-large-en-v1.5` | `Qwen3-Embedding-8B` | 统一本地模型 |
| **模型路径** | 所有方法 | HuggingFace Hub 下载 | `/workspace/models/Qwen3-Embedding-8B` | 本地路径 |
| **API类型** | DyG-RAG | OpenAI | SentenceTransformer | 本地加载 |
| | GraphRAG-master | `hf` | `hf` | 保持不变 |
| **嵌入维度** | DyG-RAG | 自动检测 | `4096` | Qwen模型维度 |
| | GraphRAG-master | `1024` | `4096` | 统一为4096 |
| **最大上下文** | DyG-RAG | `8192` | `32768` | Qwen支持32k |
| | GraphRAG-master | `8192` | `32768` | 统一为32k |
| **批处理大小** | DyG-RAG | `32` | `128` | 提高效率 |
| | GraphRAG-master | `128` | `128` | 保持不变 |
| **最大并发** | DyG-RAG | `32` | `32` | 保持不变 |
| | GraphRAG-master | `16` | `16` | 保持不变 |

**统一使用**: **Qwen3-Embedding-8B** 本地部署（通过 SentenceTransformer）

#### DyG-RAG调整
```python
from sentence_transformers import SentenceTransformer

# 使用统一的 Qwen-Embedding-8B 模型
QWEN_EMBEDDING_PATH = "/workspace/models/Qwen3-Embedding-8B"

def get_qwen_embedding():
    model = SentenceTransformer(
        QWEN_EMBEDDING_PATH,
        trust_remote_code=True
    )
    return model

# 在 GraphRAG 中使用（方式1：使用内置函数）
from graphrag import GraphRAG
from graphrag._llm import qwen_embedding_8b

graph_func = GraphRAG(
    working_dir=str(WORK_DIR),
    embedding_func=qwen_embedding_8b,  # 使用 Qwen-Embedding-8B
    embedding_batch_num=128,
)

# 方式2：自定义 EmbeddingFunc（如示例脚本）
from dataclasses import dataclass

@dataclass
class EmbeddingFunc:
    model: SentenceTransformer

    def __post_init__(self):
        self.embedding_dim = 4096
        self.max_token_size = 32768

    async def __call__(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=32)

embedding_func = EmbeddingFunc(model=get_qwen_embedding())

graph_func = GraphRAG(
    working_dir=str(WORK_DIR),
    embedding_func=embedding_func,
    embedding_batch_num=128,
)
```

#### GraphRAG-master框架调整
在`Config2.yaml`中：
```yaml
embedding:
  api_type: "hf"
  model: "/workspace/models/Qwen3-Embedding-8B"  # 本地 Qwen-Embedding-8B 模型
  dimensions: 4096  # Qwen-Embedding-8B 输出维度
  embed_batch_size: 128  # 统一批处理大小
  cache_folder: ""  # 本地模型无需缓存
```

**注意事项**:
1. **维度变化**: 从 1024 (BGE) 升级到 4096 (Qwen)，向量数据库大小会增加约4倍
2. **性能影响**: Qwen-Embedding-8B 是8B参数模型，推理速度可能比 BGE-large (335M) 慢，建议使用 GPU 加速
3. **兼容性**: 需要 transformers>=4.51.0 和 sentence-transformers>=2.7.0
4. **Trust Remote Code**: Qwen 模型需要设置 `trust_remote_code=True` 才能加载

---

### 文本分块统一

**原因**: 分块策略影响信息粒度和检索效果。

#### 配置前后对比

| 配置项 | 方法 | 调整前 | 调整后 | 说明 |
|--------|------|--------|--------|------|
| **chunk_token_size** | DyG-RAG | `1200` | `1200` | 保持不变 |
| | GraphRAG-master | `1200` | `1200` | 保持不变 |
| **chunk_overlap** | DyG-RAG | `64` | `100` | **需调整** |
| | GraphRAG-master | `100` | `100` | 保持不变 |
| **token_model** | DyG-RAG | `cl100k_base` | `cl100k_base` | 保持不变 |
| | GraphRAG-master | `gpt-3.5-turbo` | `gpt-3.5-turbo` | 保持不变 |

**注意**: DyG-RAG的`cl100k_base`编码器与`gpt-3.5-turbo`使用相同的分词器，因此token计数一致，无需调整。

#### DyG-RAG调整
```python
graph_func = GraphRAG(
    chunk_token_size=1200,
    chunk_overlap_token_size=100,  # 从64调整为100
)
```

#### GraphRAG-master调整

已默认1200/100，无需调整

---

### 推理温度统一

**原因**: 温度参数影响模型输出的随机性和一致性。

#### 配置前后对比

| 配置项 | 方法 | 调整前 | 调整后 | 说明 |
|--------|------|--------|--------|------|
| **temperature** | DyG-RAG | 未明确设置 | `0.0` | **需添加** |
| | GraphRAG-master | `0.0` (默认) | `0.0` | 保持不变 |

**统一值**: `0.0` (确定性输出，有利于可复现性)

#### DyG-RAG调整
```python
# 在_llm.py中的complete函数调用时或在qwen3_32b_complete函数中
response = await client.chat.completions.create(
    model=model,
    messages=messages,
    temperature=0.0,  # 明确设置为0
)
```

#### GraphRAG-master调整

在`Config2.yaml`中（已是默认值，确保设置）：

```yaml
llm:
  temperature: 0.0
```

---

### Qwen3-32B 思考功能配置

**原因**: Qwen3-32B 模型默认启用思考(Thinking/CoT)功能，在某些推理任务中会生成冗长的中间推理步骤，影响效率。禁用思考功能可以获得直接的答案，提升性能和可预测性。

#### 思考功能说明

| 配置项 | 说明 | 默认值 | 调整后 |
|--------|------|--------|--------|
| **enable_thinking** | 是否启用模型的推理过程 | `True` | `False` |
| **配置方式** | 通过API参数控制 | 无 | extra_body参数 |
| **性能提升** | 响应速度提升 | - | 约20-40% |

#### 配置方式

通过 OpenAI 兼容 API 的 `extra_body` 参数配置：

```python
response = await client.chat.completions.create(
    model="qwen3-32b",
    messages=messages,
    temperature=0.0,
    max_tokens=4096,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}}  # 禁用思考
)
```

#### DyG-RAG 中的应用

- **核心库修改**: `graphrag/_llm.py` 中 `qwen3_32b_complete_if_cache()` 函数已内置禁用思考配置
- **示例脚本**: 所有示例和复现脚本的 `_chat_completion()` 函数默认禁用思考功能
- **简化设计**: 项目仅使用 Qwen3-32B 一个模型,无需模型检测,统一采用禁用思考的配置
- **向后兼容**: 用户可通过 kwargs 传入自定义 `extra_body` 参数覆盖默认配置

#### VLLM 启动说明

启动 VLLM 服务时无需特殊参数，思考功能通过 API 调用参数控制：

```bash
python -m vllm.entrypoints.openai.api_server \
    --model /workspace/models/Qwen3-32B \
    --served-model-name qwen3-32b \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --max-model-len 32768
```

**注意**: 如果之前使用启用思考功能的配置生成了缓存，新配置会生成不同的响应。建议清空LLM缓存或使用新的工作目录。

#### 参考资源

- [Qwen3 官方部署文档](https://qwen.readthedocs.io/en/latest/deployment/vllm.html)
- [VLLM 推理输出文档](https://docs.vllm.ai/en/v0.9.1/features/reasoning_outputs.html)

---

### Token限制统一

**原因**: 上下文长度影响检索信息量，需要统一以确保公平比较。

#### 配置前后对比

| 配置项 | 方法 | 调整前 | 调整后 | 说明 |
|--------|------|--------|--------|------|
| **max_token_for_text_unit** | DyG-RAG | `12000` | `12000` | 保持不变 |
| | GraphRAG-master | `4000` | `12000` | **需调整** |
| **max_token_for_community_report** | DyG-RAG | N/A | N/A | 不适用 |
| | LGraphRAG | `3200` | `3200` | 保持不变 |
| | GGraphRAG | `16384` | `16384` | Global特性保留 |

**统一值**:
- 文本单元: `12000` tokens
- 社区报告: `3200` tokens (Local) / `16384` tokens (Global)


#### DyG-RAG调整

无需调整，保持默认值：

```python
result = graph_func.query(
    question,
    param=QueryParam(
        mode="dynamic",
        max_token_for_text_unit=12000,  # 保持默认值
    )
)
```

#### GraphRAG-master调整

在各方法YAML配置文件中调整：

```yaml
query:
  max_token_for_text_unit: 12000  # 从4000调整为12000
```
