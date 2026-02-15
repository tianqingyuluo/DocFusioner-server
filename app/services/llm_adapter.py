"""
LLM 统一适配层。

封装 LLM 的同步对话和流式对话能力。
仅负责 chat/chat_stream，embed 能力留给第三批。
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any, Literal

from openai import AsyncOpenAI

from app.config import get_settings
from app.exceptions import (
    LLMAuthError,
    LLMConnectionError,
    LLMModelNotFoundError,
    LLMOutputParseError,
    LLMRateLimitError,
)
from app.services.config_service import dynamic_config_service

logger = logging.getLogger(__name__)


@dataclass
class Usage:
    """Token 用量统计。"""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass
class ChatResult:
    """单次完整回复。"""

    content: str
    finish_reason: str
    usage: Usage | None = None


@dataclass
class StreamEvent:
    """流式输出事件。"""

    type: Literal["delta", "done", "error"]
    text_delta: str | None = None
    finish_reason: str | None = None
    error_message: str | None = None


_client_cache: dict[str, tuple[AsyncOpenAI, float]] = {}
_CLIENT_TTL = 600


def _cache_key(provider: str, base_url: str, api_key: str) -> str:
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]
    return f"{provider}|{base_url}|{key_hash}"


def _get_or_create_openai_client(
    provider: str, base_url: str, api_key: str
) -> AsyncOpenAI:
    """获取或创建 AsyncOpenAI 客户端（带缓存）。"""
    key = _cache_key(provider, base_url, api_key)
    now = time.monotonic()

    cached = _client_cache.get(key)
    if cached is not None:
        client, created_at = cached
        if now - created_at < _CLIENT_TTL:
            return client

    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    _client_cache[key] = (client, now)
    return client


class LLMAdapter(ABC):
    """LLM 适配器抽象基类。"""

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: str | None = None,
        json_schema: dict | None = None,
        deadline: float | None = None,
    ) -> ChatResult:
        """单次完整回复。"""

    @abstractmethod
    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: str | None = None,
        deadline: float | None = None,
    ) -> AsyncGenerator[StreamEvent]:
        """结构化流式输出。"""
        yield  # pragma: no cover


class _OpenAICompatibleAdapter(LLMAdapter):
    """OpenAI 兼容适配器（DeepSeek / Ollama 共用实现）。"""

    def __init__(self, client: AsyncOpenAI, model: str, provider: str):
        self._client = client
        self._model = model
        self._provider = provider

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: str | None = None,
        json_schema: dict | None = None,
        deadline: float | None = None,
    ) -> ChatResult:
        """单次完整回复，含 JSON 保障和重试。"""
        start = time.monotonic()

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": list(messages),
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        if response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}
            if json_schema:
                schema_hint = f"请严格按照以下 JSON 结构输出：{json.dumps(json_schema, ensure_ascii=False)}"
                kwargs["messages"] = [
                    {"role": "system", "content": schema_hint},
                    *kwargs["messages"],
                ]

        raw_content = await self._call_with_retry(kwargs, deadline, start)

        if response_format == "json":
            raw_content = await self._ensure_json(
                raw_content, kwargs, json_schema, deadline, start
            )

        prompt_chars = sum(len(m.get("content", "")) for m in messages)
        logger.info(
            "LLM chat: provider=%s model=%s prompt_chars=%d output_chars=%d",
            self._provider,
            self._model,
            prompt_chars,
            len(raw_content),
        )

        return ChatResult(content=raw_content, finish_reason="stop")

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: str | None = None,
        deadline: float | None = None,
    ) -> AsyncGenerator[StreamEvent]:
        """流式输出，保证最终 yield done。"""
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": list(messages),
            "stream": True,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}

        finish_reason = "error"
        try:
            stream = await self._client.chat.completions.create(**kwargs)
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield StreamEvent(type="delta", text_delta=chunk.choices[0].delta.content)
                if chunk.choices and chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason
        except Exception as e:
            yield StreamEvent(type="error", error_message=str(e))
        finally:
            yield StreamEvent(type="done", finish_reason=finish_reason)

    async def _call_with_retry(
        self, kwargs: dict, deadline: float | None, start: float
    ) -> str:
        """调用 LLM API，带重试策略。"""
        import random

        from openai import APIConnectionError, APIStatusError, APITimeoutError

        max_retries = 2
        for attempt in range(max_retries + 1):
            if deadline is not None:
                remaining = deadline - (time.monotonic() - start)
                if remaining < 5 and attempt > 0:
                    break

            try:
                resp = await self._client.chat.completions.create(**kwargs)
                content = resp.choices[0].message.content or ""
                return content
            except APIStatusError as e:
                status = e.status_code
                if status == 401:
                    raise LLMAuthError(
                        message=f"API Key 无效: {self._provider}",
                        provider=self._provider,
                        model=self._model,
                    ) from e
                if status == 404:
                    raise LLMModelNotFoundError(
                        message=f"模型不存在: {self._model}",
                        provider=self._provider,
                        model=self._model,
                    ) from e
                if status in (429, 503) and attempt < max_retries:
                    backoff = min(2**attempt + random.uniform(0, 1), 3)
                    import asyncio

                    await asyncio.sleep(backoff)
                    continue
                if status in (429, 503):
                    raise LLMRateLimitError(
                        message=f"限流: {self._provider} ({status})",
                        provider=self._provider,
                        retry_count=attempt,
                    ) from e
                raise LLMConnectionError(
                    message=f"API 错误 ({status}): {self._provider}",
                    provider=self._provider,
                    base_url=str(self._client.base_url),
                ) from e
            except (APIConnectionError, APITimeoutError) as e:
                if attempt < 1:
                    import asyncio

                    await asyncio.sleep(1)
                    continue
                raise LLMConnectionError(
                    message=f"连接失败: {self._provider}",
                    provider=self._provider,
                    base_url=str(self._client.base_url),
                ) from e

        raise LLMConnectionError(
            message=f"重试耗尽: {self._provider}",
            provider=self._provider,
            base_url=str(self._client.base_url),
        )

    async def _ensure_json(
        self,
        raw: str,
        kwargs: dict,
        json_schema: dict | None,
        deadline: float | None,
        start: float,
    ) -> str:
        """JSON 保障：解析 → schema 校验 → fallback 重试。"""
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            retry_messages = list(kwargs["messages"]) + [
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": "上面的输出不是合法 JSON。请只输出合法 JSON，不要任何解释。",
                },
            ]
            retry_kwargs = {**kwargs, "messages": retry_messages}
            retry_raw = await self._call_with_retry(retry_kwargs, deadline, start)
            try:
                parsed = json.loads(retry_raw)
            except json.JSONDecodeError:
                raise LLMOutputParseError(
                    message="JSON 解析失败",
                    reason="json_invalid",
                    provider=self._provider,
                    model=self._model,
                    raw_output=retry_raw[:500],
                )
            raw = retry_raw

        if json_schema and isinstance(parsed, dict):
            expected = set(
                json_schema.get("required", [])
                or json_schema.get("properties", {}).keys()
            )
            actual = set(parsed.keys())
            if expected and not expected.issubset(actual):
                retry_messages = list(kwargs["messages"]) + [
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": f"输出缺少必要字段。需要包含: {sorted(expected)}。请重新输出完整 JSON。",
                    },
                ]
                retry_kwargs = {**kwargs, "messages": retry_messages}
                retry_raw = await self._call_with_retry(retry_kwargs, deadline, start)
                try:
                    retry_parsed = json.loads(retry_raw)
                    if expected.issubset(set(retry_parsed.keys())):
                        return retry_raw
                except (json.JSONDecodeError, AttributeError):
                    pass
                raise LLMOutputParseError(
                    message="JSON schema 不匹配",
                    reason="schema_mismatch",
                    provider=self._provider,
                    model=self._model,
                    raw_output=retry_raw[:500],
                )

        return raw


class LLMClient:
    """LLM 客户端门面。"""

    def __init__(self, chat_adapter: LLMAdapter, embedder: Any = None):
        self._chat_adapter = chat_adapter
        self.embedder = embedder

    async def chat(self, messages: list[dict[str, str]], **kwargs) -> ChatResult:
        return await self._chat_adapter.chat(messages=messages, **kwargs)

    async def chat_stream(
        self, messages: list[dict[str, str]], **kwargs
    ) -> AsyncGenerator[StreamEvent]:
        async for event in self._chat_adapter.chat_stream(messages=messages, **kwargs):
            yield event


def get_llm_client() -> LLMClient:
    """
    创建 LLMClient 实例。

    1. 读取动态配置 → provider, model
    2. 读取静态配置 → api_key, base_url
    3. 获取/创建 AsyncOpenAI client（缓存）
    4. 构造 Adapter → 包装为 LLMClient
    """
    cfg = dynamic_config_service.get()
    settings = get_settings()

    provider = cfg.llm_provider
    model = cfg.llm_model

    if provider == "deepseek":
        base_url = "https://api.deepseek.com/v1"
        api_key = settings.deepseek_api_key
    elif provider == "ollama":
        base_url = f"{settings.ollama_base_url}/v1"
        api_key = "ollama"
    else:
        raise ValueError(f"不支持的 LLM provider: {provider}")

    client = _get_or_create_openai_client(provider, base_url, api_key)
    adapter = _OpenAICompatibleAdapter(client=client, model=model, provider=provider)
    return LLMClient(chat_adapter=adapter)
