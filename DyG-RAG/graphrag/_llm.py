import json
import numpy as np
from typing import Optional, List, Any, Callable

import aioboto3
from openai import AsyncOpenAI, AsyncAzureOpenAI, APIConnectionError, RateLimitError

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import os

from ._utils import compute_args_hash, wrap_embedding_func_with_attrs
from .base import BaseKVStorage

global_openai_async_client = None
global_azure_openai_async_client = None
global_amazon_bedrock_async_client = None


def get_openai_async_client_instance():
    global global_openai_async_client
    if global_openai_async_client is None:
        global_openai_async_client = AsyncOpenAI()
    return global_openai_async_client


def get_azure_openai_async_client_instance():
    global global_azure_openai_async_client
    if global_azure_openai_async_client is None:
        global_azure_openai_async_client = AsyncAzureOpenAI()
    return global_azure_openai_async_client


def get_amazon_bedrock_async_client_instance():
    global global_amazon_bedrock_async_client
    if global_amazon_bedrock_async_client is None:
        global_amazon_bedrock_async_client = aioboto3.Session()
    return global_amazon_bedrock_async_client


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def openai_complete_if_cache(
    model, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    openai_async_client = get_openai_async_client_instance()
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    response = await openai_async_client.chat.completions.create(
        model=model, messages=messages, **kwargs
    )

    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response.choices[0].message.content, "model": model}}
        )
        await hashing_kv.index_done_callback()
    return response.choices[0].message.content


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def amazon_bedrock_complete_if_cache(
    model, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    amazon_bedrock_async_client = get_amazon_bedrock_async_client_instance()
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages = []
    messages.extend(history_messages)
    messages.append({"role": "user", "content": [{"text": prompt}]})
    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    inference_config = {
        "temperature": 0,
        "maxTokens": 4096 if "max_tokens" not in kwargs else kwargs["max_tokens"],
    }

    async with amazon_bedrock_async_client.client(
        "bedrock-runtime",
        region_name=os.getenv("AWS_REGION", "us-east-1")
    ) as bedrock_runtime:
        if system_prompt:
            response = await bedrock_runtime.converse(
                modelId=model, messages=messages, inferenceConfig=inference_config,
                system=[{"text": system_prompt}]
            )
        else:
            response = await bedrock_runtime.converse(
                modelId=model, messages=messages, inferenceConfig=inference_config,
            )

    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response["output"]["message"]["content"][0]["text"], "model": model}}
        )
        await hashing_kv.index_done_callback()
    return response["output"]["message"]["content"][0]["text"]


def create_amazon_bedrock_complete_function(model_id: str) -> Callable:
    """
    Factory function to dynamically create completion functions for Amazon Bedrock

    Args:
        model_id (str): Amazon Bedrock model identifier (e.g., "us.anthropic.claude-3-sonnet-20240229-v1:0")

    Returns:
        Callable: Generated completion function
    """
    async def bedrock_complete(
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: List[Any] = [],
        **kwargs
    ) -> str:
        return await amazon_bedrock_complete_if_cache(
            model_id,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **kwargs
        )
    
    # Set function name for easier debugging
    bedrock_complete.__name__ = f"{model_id}_complete"
    
    return bedrock_complete


async def gpt_4o_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        "gpt-3.5-turbo",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def gpt_4o_mini_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        "gpt-4o-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


@wrap_embedding_func_with_attrs(embedding_dim=1024, max_token_size=8192)
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def amazon_bedrock_embedding(texts: list[str]) -> np.ndarray:
    amazon_bedrock_async_client = get_amazon_bedrock_async_client_instance()

    async with amazon_bedrock_async_client.client(
        "bedrock-runtime",
        region_name=os.getenv("AWS_REGION", "us-east-1")
    ) as bedrock_runtime:
        embeddings = []
        for text in texts:
            body = json.dumps(
                {
                    "inputText": text,
                    "dimensions": 1024,
                }
            )
            response = await bedrock_runtime.invoke_model(
                modelId="amazon.titan-embed-text-v2:0", body=body,
            )
            response_body = await response.get("body").read()
            embeddings.append(json.loads(response_body))
    return np.array([dp["embedding"] for dp in embeddings])


@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def openai_embedding(texts: list[str]) -> np.ndarray:
    openai_async_client = get_openai_async_client_instance()
    response = await openai_async_client.embeddings.create(
        model="text-embedding-3-small", input=texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data])


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


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def azure_openai_complete_if_cache(
    deployment_name, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    azure_openai_client = get_azure_openai_async_client_instance()
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(deployment_name, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    response = await azure_openai_client.chat.completions.create(
        model=deployment_name, messages=messages, **kwargs
    )

    if hashing_kv is not None:
        await hashing_kv.upsert(
            {
                args_hash: {
                    "return": response.choices[0].message.content,
                    "model": deployment_name,
                }
            }
        )
        await hashing_kv.index_done_callback()
    return response.choices[0].message.content


async def azure_gpt_4o_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await azure_openai_complete_if_cache(
        "gpt-4o",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def azure_gpt_4o_mini_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await azure_openai_complete_if_cache(
        "gpt-4o-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def azure_openai_embedding(texts: list[str]) -> np.ndarray:
    azure_openai_client = get_azure_openai_async_client_instance()
    response = await azure_openai_client.embeddings.create(
        model="text-embedding-3-small", input=texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data])


# ============================================================================
# Local Qwen3-32B Model Functions (using VLLM)
# ============================================================================
# To use these functions, first start the VLLM service:
# python -m vllm.entrypoints.openai.api_server \
#     --model /workspace/models/Qwen3-32B \
#     --served-model-name qwen3-32b \
#     --host 0.0.0.0 \
#     --port 8000 \
#     --tensor-parallel-size 2 \
#     --max-model-len 32768
# ============================================================================

global_qwen3_32b_async_client = None


def get_qwen3_32b_async_client_instance():
    """Get or create the global Qwen3-32B AsyncOpenAI client instance."""
    global global_qwen3_32b_async_client
    if global_qwen3_32b_async_client is None:
        global_qwen3_32b_async_client = AsyncOpenAI(
            base_url="http://localhost:8000/v1",
            api_key="EMPTY"  # Local VLLM doesn't require API key
        )
    return global_qwen3_32b_async_client


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def qwen3_32b_complete_if_cache(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    """
    Complete text using local Qwen3-32B model via VLLM service.

    Args:
        prompt: The user prompt
        system_prompt: Optional system prompt
        history_messages: List of previous messages in the conversation
        **kwargs: Additional parameters, including:
            - hashing_kv: BaseKVStorage for caching
            - temperature: Sampling temperature (default: 0.0)
            - max_tokens: Maximum tokens to generate (default: 4096)

    Returns:
        Generated text response
    """
    qwen_client = get_qwen3_32b_async_client_instance()
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)

    # Build messages
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    # Check cache
    if hashing_kv is not None:
        args_hash = compute_args_hash("qwen3-32b", messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    # Call VLLM service
    response = await qwen_client.chat.completions.create(
        model="qwen3-32b",
        messages=messages,
        temperature=kwargs.get("temperature", 0.0),  # Default to 0.0 for deterministic output
        max_tokens=kwargs.get("max_tokens", 4096),
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    # Cache result
    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response.choices[0].message.content, "model": "qwen3-32b"}}
        )
        await hashing_kv.index_done_callback()

    return response.choices[0].message.content


async def qwen3_32b_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    """
    Convenience function for Qwen3-32B completion.
    This is the primary function to use for local Qwen3-32B inference.

    Args:
        prompt: The user prompt
        system_prompt: Optional system prompt
        history_messages: List of previous messages
        **kwargs: Additional parameters (temperature, max_tokens, hashing_kv, etc.)

    Returns:
        Generated text response
    """
    return await qwen3_32b_complete_if_cache(
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )
