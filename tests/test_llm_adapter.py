"""测试 LLM 适配层：数据结构 + ABC + DeepSeek/Ollama 适配器。"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestLLMDataStructures:
    """LLM 数据结构测试"""

    def test_chat_result_fields(self):
        from app.services.llm_adapter import ChatResult

        result = ChatResult(content="回复", finish_reason="stop")
        assert result.content == "回复"
        assert result.finish_reason == "stop"
        assert result.usage is None

    def test_usage_fields(self):
        from app.services.llm_adapter import Usage

        usage = Usage(prompt_tokens=100, completion_tokens=50)
        assert usage.prompt_tokens == 100

    def test_stream_event_delta(self):
        from app.services.llm_adapter import StreamEvent

        ev = StreamEvent(type="delta", text_delta="你好")
        assert ev.type == "delta"
        assert ev.text_delta == "你好"

    def test_stream_event_done(self):
        from app.services.llm_adapter import StreamEvent

        ev = StreamEvent(type="done", finish_reason="stop")
        assert ev.finish_reason == "stop"

    def test_stream_event_error(self):
        from app.services.llm_adapter import StreamEvent

        ev = StreamEvent(type="error", error_message="超时")
        assert ev.error_message == "超时"


class TestLLMAdapterABC:
    """LLMAdapter 抽象基类测试"""

    def test_cannot_instantiate(self):
        from app.services.llm_adapter import LLMAdapter

        with pytest.raises(TypeError):
            LLMAdapter()  # type: ignore[abstract]


class TestLLMClientFacade:
    """LLMClient 门面测试"""

    @pytest.mark.asyncio
    async def test_chat_delegates_to_adapter(self):
        from app.services.llm_adapter import ChatResult, LLMClient

        mock_adapter = MagicMock()
        mock_adapter.chat = AsyncMock(
            return_value=ChatResult(content="ok", finish_reason="stop")
        )
        client = LLMClient(chat_adapter=mock_adapter)

        result = await client.chat(messages=[{"role": "user", "content": "hi"}])
        assert result.content == "ok"
        mock_adapter.chat.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_chat_stream_delegates_to_adapter(self):
        from app.services.llm_adapter import LLMClient, StreamEvent

        async def fake_stream(**kwargs):
            yield StreamEvent(type="delta", text_delta="hi")
            yield StreamEvent(type="done", finish_reason="stop")

        mock_adapter = MagicMock()
        mock_adapter.chat_stream = fake_stream
        client = LLMClient(chat_adapter=mock_adapter)

        events = []
        async for ev in client.chat_stream(messages=[{"role": "user", "content": "hi"}]):
            events.append(ev)

        assert len(events) == 2
        assert events[-1].type == "done"


class TestClientCache:
    """AsyncOpenAI 客户端缓存测试"""

    def test_same_config_returns_same_client(self):
        from app.services.llm_adapter import _get_or_create_openai_client

        with patch("app.services.llm_adapter.AsyncOpenAI") as MockClient:
            MockClient.return_value = MagicMock()

            c1 = _get_or_create_openai_client(
                provider="deepseek",
                base_url="https://api.deepseek.com/v1",
                api_key="sk-test123",
            )
            c2 = _get_or_create_openai_client(
                provider="deepseek",
                base_url="https://api.deepseek.com/v1",
                api_key="sk-test123",
            )

            assert c1 is c2
            assert MockClient.call_count == 1

    def test_different_config_returns_different_client(self):
        from app.services.llm_adapter import _client_cache, _get_or_create_openai_client

        _client_cache.clear()

        with patch("app.services.llm_adapter.AsyncOpenAI") as MockClient:
            MockClient.side_effect = [MagicMock(), MagicMock()]

            c1 = _get_or_create_openai_client(
                provider="deepseek",
                base_url="https://api.deepseek.com/v1",
                api_key="sk-key-a",
            )
            c2 = _get_or_create_openai_client(
                provider="ollama",
                base_url="http://localhost:11434/v1",
                api_key="ollama",
            )

            assert c1 is not c2


class TestGetLLMClient:
    """工厂函数测试"""

    def test_get_llm_client_deepseek(self):
        from app.services.llm_adapter import LLMClient, get_llm_client

        with (
            patch("app.services.llm_adapter.dynamic_config_service") as mock_cfg_svc,
            patch("app.services.llm_adapter.get_settings") as mock_settings,
            patch("app.services.llm_adapter.AsyncOpenAI"),
        ):
            mock_cfg = MagicMock()
            mock_cfg.llm_provider = "deepseek"
            mock_cfg.llm_model = "deepseek-chat"
            mock_cfg_svc.get.return_value = mock_cfg

            mock_settings.return_value.deepseek_api_key = "sk-test"

            client = get_llm_client()
            assert isinstance(client, LLMClient)

    def test_get_llm_client_ollama(self):
        from app.services.llm_adapter import LLMClient, get_llm_client

        with (
            patch("app.services.llm_adapter.dynamic_config_service") as mock_cfg_svc,
            patch("app.services.llm_adapter.get_settings") as mock_settings,
            patch("app.services.llm_adapter.AsyncOpenAI"),
        ):
            mock_cfg = MagicMock()
            mock_cfg.llm_provider = "ollama"
            mock_cfg.llm_model = "qwen2:7b"
            mock_cfg_svc.get.return_value = mock_cfg

            mock_settings.return_value.ollama_base_url = "http://localhost:11434"

            client = get_llm_client()
            assert isinstance(client, LLMClient)
