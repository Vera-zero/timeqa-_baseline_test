from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os, json, time, asyncio, logging, re
from dataclasses import dataclass
from pathlib import Path
from typing import List
from datetime import datetime

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, models
from openai import AsyncOpenAI
import aiohttp.client_exceptions
import torch, gc

from graphrag import GraphRAG, QueryParam
from graphrag.base import BaseKVStorage
from graphrag._utils import compute_args_hash, logger

from graphrag import GraphRAG, QueryParam
import json
from tqdm import tqdm
from pathlib import Path

WORK_DIR = Path("/workspace/ETE-Graph/QA-result/tempreason/DyG-RAG")
WORK_DIR.mkdir(exist_ok=True)
CORPUS_FILE = Path("/workspace/ETE-Graph/dataset/tempreason/test_l3_processed.json")

from graphrag._utils import compute_args_hash, logger

import tiktoken
tiktoken.get_encoding("cl100k_base")

logging.basicConfig(level=logging.INFO)
logging.getLogger("DyG-RAG").setLevel(logging.INFO)

################################################################################
# 0. Configuration
################################################################################
def get_config_value(env_name: str, description: str, example: str = None) -> str:
    """Get configuration value from environment variable or user input."""
    value = os.getenv(env_name)
    if value:
        return value
    
    print(f"\n⚠️  Missing configuration: {env_name}")
    print(f"Description: {description}")
    if example:
        print(f"Example: {example}")
    
    while True:
        user_input = input(f"Please enter {env_name}: ").strip()
        if user_input:
            return user_input
        print("❌ Value cannot be empty. Please try again.")

################################################################################
# VLLM Service Configuration
# Before running this script, start the VLLM service in a separate terminal:
#
# python -m vllm.entrypoints.openai.api_server \
#     --model /workspace/models/Qwen3-32B \
#     --served-model-name qwen3-32b \
#     --host 0.0.0.0 \
#     --port 8000 \
#     --tensor-parallel-size 2 \
#     --max-model-len 32768
#
# Then verify the service is running:
# curl http://localhost:8000/v1/models
################################################################################

print("🔧 Checking configuration...")
VLLM_BASE_URL = get_config_value(
    "VLLM_BASE_URL",
    "Base URL for VLLM API service",
    "http://localhost:8000/v1"
)

BEST_MODEL_NAME = get_config_value(
    "QWEN_BEST",
    "Model name for the best/primary LLM (must match --served-model-name)",
    "qwen3-32b"
)

LOCAL_QWEN_EMBEDDING_PATH = get_config_value(
    "LOCAL_QWEN_EMBEDDING_PATH",
    "Local path to Qwen3-Embedding-8B model",
    "/workspace/models/Qwen3-Embedding-8B"
)

OPENAI_API_KEY_FAKE = "EMPTY"

@dataclass
class EmbeddingFunc:
    embedding_dim: int
    max_token_size: int
    model: SentenceTransformer

    async def __call__(self, texts: List[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        loop = asyncio.get_event_loop()
        encode = lambda: self.model.encode(
            texts,
            batch_size=32,  # Smaller batch size to help with memory issues
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        if loop.is_running():
            return await loop.run_in_executor(None, encode)
        else:
            return encode()
    
    # Make the model not serializable for GraphRAG initialization
    def __getstate__(self):
        state = self.__dict__.copy()
        state['model'] = None
        return state
    
    # Restore model reference during deserialization (will be None, but structure preserved)
    def __setstate__(self, state):
        self.__dict__.update(state)

def get_qwen_embedding_func() -> EmbeddingFunc:
    gpu_count = torch.cuda.device_count()
    using_cuda = gpu_count > 0
    # Use cuda:0 explicitly to avoid device mismatch issues with multi-GPU
    device = "cuda:0" if using_cuda else "cpu"

    # Always use single device to prevent tensor device mismatch
    # When device_map="auto", tensors can end up on different GPUs (cuda:0, cuda:1)
    # causing "Expected all tensors to be on the same device" errors
    model_kwargs = {}
    if gpu_count > 1:
        model_kwargs = {"torch_dtype": torch.float16}

    st_model = SentenceTransformer(
        LOCAL_QWEN_EMBEDDING_PATH,
        device=device,
        trust_remote_code=True,
        model_kwargs=model_kwargs,
    )

    return EmbeddingFunc(
        embedding_dim=st_model.get_sentence_embedding_dimension(),
        max_token_size=32768,        # Qwen-Embedding-8B supports 32k context
        model=st_model,
    )
    
################################################################################
# 2. LLM call function (with cache)
################################################################################

def _build_async_client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=OPENAI_API_KEY_FAKE, base_url=VLLM_BASE_URL)

async def _chat_completion(model: str, messages: list[dict[str, str]], **kwargs) -> str:
    client = _build_async_client()

    # 为 Qwen3-32B 默认禁用思考功能
    extra_body = kwargs.pop("extra_body", {})
    extra_body["chat_template_kwargs"] = {"enable_thinking": False}
    kwargs["extra_body"] = extra_body

    response = await client.chat.completions.create(model=model, messages=messages, **kwargs)
    return response.choices[0].message.content

async def _llm_with_cache(
    prompt: str,
    *,
    model: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, str]] | None = None,
    hashing_kv: BaseKVStorage | None = None,
    **kwargs,
) -> str:
    """General LLM wrapper, supporting GraphRAG cache interface."""
    history_messages = history_messages or []
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.extend(history_messages)
    msgs.append({"role": "user", "content": prompt})

    if hashing_kv is not None:
        args_hash = compute_args_hash(model, msgs)
        cached = await hashing_kv.get_by_id(args_hash)
        if cached is not None:
            return cached["return"]

    answer = await _chat_completion(model=model, messages=msgs, **kwargs)

    if hashing_kv is not None:
        await hashing_kv.upsert({args_hash: {"return": answer, "model": model}})
        await hashing_kv.index_done_callback()
    return answer

async def best_model_func(prompt: str, system_prompt: str | None = None, history_messages: list[dict[str,str]] | None = None, **kwargs) -> str:
    return await _llm_with_cache(
        prompt,
        model=BEST_MODEL_NAME,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )

embedding_func = get_qwen_embedding_func()
model_ref = embedding_func.model
embedding_func.model = None 

def read_json_file(fp: Path):
    with fp.open(encoding="utf-8") as f:
        return json.load(f)

graph_func = GraphRAG(
    working_dir=str(WORK_DIR),
    embedding_func=embedding_func,
    best_model_func=best_model_func,
    cheap_model_func=best_model_func,
    enable_llm_cache=True,
    best_model_max_token_size = 16384,
    cheap_model_max_token_size = 16384,
    model_path="/workspace/models",
    ce_model="cross-encoder/ms-marco-TinyBERT-L-2-v2",  
    ner_model_name="dslim_bert_base_ner", 
)

embedding_func.model = model_ref

corpus_data = read_json_file(CORPUS_FILE)
datas_content = corpus_data["datas"]
total_docs = len(datas_content)
logger.info(f"Start processing, total {total_docs} documents to process.")

corpus_data = read_json_file(CORPUS_FILE)
datas_content = corpus_data["contents"][:2]
total_docs = len(datas_content)
#print (f"datas:{datas_content[0]}, total_docs: {total_docs}")
all_docs = []
all_questions_list = []
for idx, obj in enumerate(tqdm(datas_content, desc="Loading docs", total=total_docs)):
    # Combine metadata with content
    enriched_content = f"{obj['fact_context']}"
    all_docs.append(enriched_content)
    all_questions_list.append(obj["question_list"])

all_questions = []
for idx, obj in enumerate(tqdm(all_questions_list, desc="Loading questions", total=len(all_questions_list))):
    for idx, questions in enumerate(obj):
        question = questions["question"]
        all_questions.append(question)
 
graph_func.insert(all_docs)

# Collect results
results = []
print(f"\n开始处理 {len(all_questions)} 个问题...")
for idx, question in enumerate(tqdm(all_questions, desc="Processing questions")):
    ans = graph_func.query(question, param=QueryParam(mode="dynamic"))
    results.append({
        "question_idx": idx,
        "question": question,
        "answer": ans
    })
    print(f"\nQuestion {idx + 1}/{len(all_questions)}: {question}")
    print(f"Answer: {ans}\n")

# Save results to JSON
output_file = WORK_DIR / "results.json"
output_data = {
    "metadata": {
        "dataset": "tempreason",
        "total_questions": len(all_questions),
        "total_docs": len(all_docs),
        "processed_time": datetime.now().isoformat(),
        "corpus_file": str(CORPUS_FILE)
    },
    "results": results
}

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print(f"\n✅ 结果已保存到: {output_file}")
print(f"   - 处理问题数: {len(all_questions)}")
print(f"   - 文档数: {len(all_docs)}")