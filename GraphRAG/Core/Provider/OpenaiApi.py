from __future__ import annotations

import numpy as np
from typing import Optional, Union
import asyncio

from openai import APIConnectionError, AsyncOpenAI, AsyncStream, RateLimitError
from openai._base_client import AsyncHttpxClientWrapper
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from tenacity import (
    after_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
    wait_exponential,
)

from Config.LLMConfig import LLMConfig, LLMType
from Core.Common.Constants import USE_CONFIG_TIMEOUT
from Core.Common.Logger import log_llm_stream, logger
from Core.Provider.BaseLLM import BaseLLM
from Core.Provider.LLMProviderRegister import register_provider
from Core.Common.Utils import  log_and_reraise,prase_json_from_response
from Core.Common.CostManager import CostManager
from Core.Utils.Exceptions import handle_exception
from Core.Utils.TokenCounter import (
    count_input_tokens,
    count_output_tokens,
    get_max_completion_tokens,
)


# ============================================================================
# Global Singleton AsyncOpenAI Client Management (DyG-RAG Style)
# ============================================================================
# This approach follows DyG-RAG's pattern of using global singleton clients
# for better resource management and consistency across requests.
# ============================================================================

# Global client instances cache
_global_clients_cache = {}


def get_openai_async_client_instance(base_url: str, api_key: str, proxy_params: dict = None):
    """
    Get or create a global AsyncOpenAI client instance for the given configuration.

    This follows DyG-RAG's pattern of using global singleton clients to avoid
    creating multiple client instances for the same configuration.

    Args:
        base_url: The base URL for the OpenAI API (or compatible local LLM service)
        api_key: The API key (use "EMPTY" for local LLM services)
        proxy_params: Optional proxy configuration

    Returns:
        AsyncOpenAI client instance
    """
    global _global_clients_cache

    # Create a cache key based on configuration
    cache_key = f"{base_url}_{api_key}"

    if cache_key not in _global_clients_cache:
        kwargs = {"api_key": api_key, "base_url": base_url}

        # Add proxy configuration if provided
        if proxy_params:
            kwargs["http_client"] = AsyncHttpxClientWrapper(**proxy_params)

        _global_clients_cache[cache_key] = AsyncOpenAI(**kwargs)

    return _global_clients_cache[cache_key]


@register_provider(
    [
        LLMType.OPENAI,
        LLMType.FIREWORKS,
        LLMType.OPEN_LLM,
    ]
)
class OpenAILLM(BaseLLM):
    """Check https://platform.openai.com/examples for examples"""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._init_client()
        self.auto_max_tokens = False
        self.cost_manager: Optional[CostManager] = None
        self.semaphore = asyncio.Semaphore(config.max_concurrent)
    def _init_client(self):
        """
        Initialize AsyncOpenAI client using global singleton pattern (DyG-RAG style).

        This follows DyG-RAG's approach of using global singleton clients for better
        resource management and to ensure consistent client instances across requests.
        """
        self.model = self.config.model  # Used in _calc_usage & _cons_kwargs
        self.pricing_plan = self.config.pricing_plan or self.model

        # Get proxy parameters if configured
        proxy_params = self._get_proxy_params()

        # Use global singleton client (DyG-RAG pattern)
        self.aclient = get_openai_async_client_instance(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            proxy_params=proxy_params if proxy_params else None
        )

    def _make_client_kwargs(self) -> dict:
        kwargs = {"api_key": self.config.api_key, "base_url": self.config.base_url}

        # to use proxy, openai v1 needs http_client
        if proxy_params := self._get_proxy_params():
            kwargs["http_client"] = AsyncHttpxClientWrapper(**proxy_params)

        return kwargs

    def _get_proxy_params(self) -> dict:
        params = {}
        if self.config.proxy:
            params = {"proxies": self.config.proxy}
            if self.config.base_url:
                params["base_url"] = self.config.base_url

        return params

    async def _achat_completion_stream(self, messages: list[dict], timeout=USE_CONFIG_TIMEOUT, max_tokens = None) -> str:
        response: AsyncStream[ChatCompletionChunk] = await self.aclient.chat.completions.create(
            **self._cons_kwargs(messages, timeout=self.get_timeout(timeout), max_tokens = max_tokens), stream=True
        )
        usage = None
        collected_messages = []
        has_finished = False
        async for chunk in response:
            chunk_message = chunk.choices[0].delta.content or "" if chunk.choices else ""  # extract the message
            finish_reason = (
                chunk.choices[0].finish_reason if chunk.choices and hasattr(chunk.choices[0], "finish_reason") else None
            )
            log_llm_stream(chunk_message)
            collected_messages.append(chunk_message)
            chunk_has_usage = hasattr(chunk, "usage") and chunk.usage
            if has_finished:
                # for oneapi, there has a usage chunk after finish_reason not none chunk
                if chunk_has_usage:
                    usage = CompletionUsage(**chunk.usage) if isinstance(chunk.usage, dict) else chunk.usage
            if finish_reason:
                if chunk_has_usage:
                    # Some services have usage as an attribute of the chunk, such as Fireworks
                    if isinstance(chunk.usage, CompletionUsage):
                        usage = chunk.usage
                    else:
                        usage = CompletionUsage(**chunk.usage)
                elif hasattr(chunk.choices[0], "usage"):
                    # The usage of some services is an attribute of chunk.choices[0], such as Moonshot
                    usage = CompletionUsage(**chunk.choices[0].usage)
                has_finished = True

        log_llm_stream("\n")
        full_reply_content = "".join(collected_messages)
        if not usage:
            # Some services do not provide the usage attribute, such as OpenAI or OpenLLM
            usage = self._calc_usage(messages, full_reply_content)

        self._update_costs(usage)
        return full_reply_content

    def _cons_kwargs(self, messages: list[dict], timeout=USE_CONFIG_TIMEOUT, max_tokens = None, **extra_kwargs) -> dict:
        kwargs = {
            "messages": messages,
            "max_tokens": self._get_max_tokens(messages),
            # "n": 1,  # Some services do not provide this parameter, such as mistral
            "stop": ["[/INST]", "<<SYS>>"] ,  # default it's None and gpt4-v can't have this one
            "temperature": self.config.temperature,
            "model": self.model,
            "timeout": self.get_timeout(timeout),
        }
        if "o1-" in self.model:
            # compatible to openai o1-series
            kwargs["temperature"] = 1
            kwargs.pop("max_tokens")
        if max_tokens != None:
            kwargs["max_tokens"] = max_tokens

        # Support for thinking mode (DyG-RAG style)
        # For vLLM-deployed models, enable_thinking is passed via extra_body
        # Using chat_template_kwargs format for consistency with DyG-RAG
        if hasattr(self.config, 'enable_thinking'):
            kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": self.config.enable_thinking}
            }

        if extra_kwargs:
            kwargs.update(extra_kwargs)
        return kwargs

    async def _achat_completion(self, messages: list[dict], timeout=USE_CONFIG_TIMEOUT, max_tokens = None) -> ChatCompletion:
        kwargs = self._cons_kwargs(messages, timeout=self.get_timeout(timeout), max_tokens=max_tokens)

        rsp: ChatCompletion = await self.aclient.chat.completions.create(**kwargs)
        self._update_costs(rsp.usage)
        return rsp

    async def acompletion(self, messages: list[dict], timeout=USE_CONFIG_TIMEOUT) -> ChatCompletion:
        return await self._achat_completion(messages, timeout=self.get_timeout(timeout))

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
    )
    async def acompletion_text(self, messages: list[dict], stream=False, timeout=USE_CONFIG_TIMEOUT, max_tokens = None, format = "text") -> str:
        """
        Asynchronous text completion with retry mechanism (DyG-RAG style).

        This follows DyG-RAG's retry configuration:
        - Max 5 attempts
        - Exponential backoff: multiplier=1, min=4s, max=10s
        - Retry on RateLimitError and APIConnectionError

        Args:
            messages: List of message dictionaries
            stream: Whether to stream the response
            timeout: Request timeout in seconds
            max_tokens: Maximum tokens to generate
            format: Output format ("text" or "json")

        Returns:
            Generated text response
        """
        if stream:
            return await self._achat_completion_stream(messages, timeout=timeout, max_tokens = max_tokens)

        rsp = await self._achat_completion(messages, timeout=self.get_timeout(timeout), max_tokens = max_tokens)

        rsp_text = self.get_choice_text(rsp)
        if format == "json":
            return prase_json_from_response(rsp_text)
        return rsp_text


 

    def get_choice_text(self, rsp: ChatCompletion) -> str:
        """Required to provide the first text of choice"""
        return rsp.choices[0].message.content if rsp.choices else ""

    def _calc_usage(self, messages: list[dict], rsp: str) -> CompletionUsage:
        usage = CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        if not self.config.calc_usage:
            return usage

        try:
            usage.prompt_tokens = count_input_tokens(messages, self.pricing_plan)
            usage.completion_tokens = count_output_tokens(rsp, self.pricing_plan)
        except Exception as e:
            logger.warning(f"usage calculation failed: {e}")

        return usage

    def _get_max_tokens(self, messages: list[dict]):
        if not self.auto_max_tokens:
            return self.config.max_token
        # FIXME
        # https://community.openai.com/t/why-is-gpt-3-5-turbo-1106-max-tokens-limited-to-4096/494973/3
        return min(get_max_completion_tokens(messages, self.model, self.config.max_token), 4096)


   
    def get_maxtokens(self) -> int:
       return ['max_tokens']

    async def openai_embedding(self, text):
        response = await self.aclient.embeddings.create(
            model = model, input = text, encoding_format = "float"
        )
        return np.array([dp.embedding for dp in response.data])