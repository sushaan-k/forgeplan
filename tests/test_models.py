"""Tests for model backends (OpenAI, Anthropic, base)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from agent_forge.exceptions import ModelError
from agent_forge.models.anthropic import AnthropicModel
from agent_forge.models.base import ModelResponse
from agent_forge.models.openai import OpenAIModel


class TestModelResponse:
    """Tests for the ModelResponse data model."""

    def test_create_empty(self) -> None:
        """ModelResponse should have sensible defaults."""
        resp = ModelResponse()
        assert resp.content == ""
        assert resp.tool_calls == []
        assert resp.raw == {}
        assert resp.usage == {}
        assert resp.model == ""

    def test_create_full(self) -> None:
        """ModelResponse should accept all fields."""
        resp = ModelResponse(
            content="Hello",
            tool_calls=[{"name": "search", "arguments": {"q": "test"}}],
            raw={"choices": []},
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            model="gpt-4o",
        )
        assert resp.content == "Hello"
        assert len(resp.tool_calls) == 1
        assert resp.usage["total_tokens"] == 30


class TestBaseModel:
    """Tests for the abstract BaseModel interface."""

    @pytest.mark.asyncio
    async def test_generate_text_convenience(self) -> None:
        """generate_text should build messages and extract content."""
        from tests.conftest import MockModel

        model = MockModel(responses=["The answer is 42"])
        result = await model.generate_text("What is the answer?")
        assert result == "The answer is 42"

    @pytest.mark.asyncio
    async def test_generate_text_with_system(self) -> None:
        """generate_text should include system message."""
        from tests.conftest import MockModel

        model = MockModel(responses=["Response"])
        result = await model.generate_text("Question", system="You are helpful")
        assert result == "Response"


class TestOpenAIModel:
    """Tests for the OpenAI model backend."""

    def test_init_defaults(self) -> None:
        """Should initialize with default values."""
        model = OpenAIModel(model_name="gpt-4o", api_key="test-key")
        assert model.model_name == "gpt-4o"
        assert model.api_key == "test-key"
        assert model.base_url == "https://api.openai.com/v1"

    def test_init_custom_base_url(self) -> None:
        """Should accept custom base URL for compatible APIs."""
        model = OpenAIModel(
            model_name="local-model",
            api_key="key",
            base_url="http://localhost:8000/v1",
        )
        assert model.base_url == "http://localhost:8000/v1"

    @pytest.mark.asyncio
    async def test_generate_success(self) -> None:
        """Should parse a successful API response."""
        mock_response = httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "Hello from GPT",
                            "tool_calls": [],
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            },
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        )

        model = OpenAIModel(model_name="gpt-4o", api_key="test-key")

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            result = await model.generate(messages=[{"role": "user", "content": "Hi"}])

        assert result.content == "Hello from GPT"
        assert result.usage["total_tokens"] == 15
        assert result.model == "gpt-4o"

    @pytest.mark.asyncio
    async def test_generate_with_tool_calls(self) -> None:
        """Should parse tool call responses."""
        mock_response = httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_123",
                                    "function": {
                                        "name": "search",
                                        "arguments": '{"query": "test"}',
                                    },
                                }
                            ],
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            },
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        )

        model = OpenAIModel(model_name="gpt-4o", api_key="test-key")

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            result = await model.generate(
                messages=[{"role": "user", "content": "Search for test"}]
            )

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "search"
        assert result.tool_calls[0]["arguments"]["query"] == "test"

    @pytest.mark.asyncio
    async def test_generate_http_error(self) -> None:
        """Should raise ModelError on HTTP errors."""
        model = OpenAIModel(model_name="gpt-4o", api_key="test-key")

        mock_request = httpx.Request(
            "POST", "https://api.openai.com/v1/chat/completions"
        )
        mock_response = httpx.Response(429, text="Rate limited", request=mock_request)
        error = httpx.HTTPStatusError(
            "Rate limited", request=mock_request, response=mock_response
        )

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = error
            with pytest.raises(ModelError) as exc_info:
                await model.generate(messages=[{"role": "user", "content": "Hi"}])
            assert exc_info.value.status_code == 429

    @pytest.mark.asyncio
    async def test_generate_network_error(self) -> None:
        """Should raise ModelError on network errors."""
        model = OpenAIModel(model_name="gpt-4o", api_key="test-key")

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.ConnectError("Connection refused")
            with pytest.raises(ModelError):
                await model.generate(messages=[{"role": "user", "content": "Hi"}])


class TestAnthropicModel:
    """Tests for the Anthropic model backend."""

    def test_init_defaults(self) -> None:
        """Should initialize with default values."""
        model = AnthropicModel(model_name="claude-sonnet-4-6", api_key="test-key")
        assert model.model_name == "claude-sonnet-4-6"
        assert model.api_key == "test-key"
        assert model.base_url == "https://api.anthropic.com/v1"
        assert model.default_params["max_tokens"] == 4096

    @pytest.mark.asyncio
    async def test_generate_success(self) -> None:
        """Should parse a successful Anthropic API response."""
        mock_response = httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "Hello from Claude"}],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
        )

        model = AnthropicModel(model_name="claude-sonnet-4-6", api_key="test-key")

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            result = await model.generate(messages=[{"role": "user", "content": "Hi"}])

        assert result.content == "Hello from Claude"
        assert result.usage["prompt_tokens"] == 10
        assert result.usage["completion_tokens"] == 5

    @pytest.mark.asyncio
    async def test_generate_extracts_system_message(self) -> None:
        """Should extract system messages and send via top-level param."""
        mock_response = httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "Response"}],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
        )

        model = AnthropicModel(model_name="claude-sonnet-4-6", api_key="test-key")

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            await model.generate(
                messages=[
                    {"role": "system", "content": "Be helpful"},
                    {"role": "user", "content": "Hi"},
                ]
            )

            call_args = mock_post.call_args
            payload = call_args.kwargs.get("json", call_args[1].get("json", {}))
            assert "system" in payload
            assert payload["system"] == "Be helpful"
            # System messages should be removed from messages list
            assert all(m["role"] != "system" for m in payload["messages"])

    @pytest.mark.asyncio
    async def test_generate_with_tool_use(self) -> None:
        """Should parse tool_use content blocks."""
        mock_response = httpx.Response(
            200,
            json={
                "content": [
                    {"type": "text", "text": "Let me search that."},
                    {
                        "type": "tool_use",
                        "id": "tu_123",
                        "name": "search",
                        "input": {"query": "test"},
                    },
                ],
                "usage": {"input_tokens": 10, "output_tokens": 15},
            },
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
        )

        model = AnthropicModel(model_name="claude-sonnet-4-6", api_key="test-key")

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            result = await model.generate(
                messages=[{"role": "user", "content": "Search for test"}]
            )

        assert "Let me search that." in result.content
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "search"

    def test_convert_tools_from_openai_format(self) -> None:
        """Should convert OpenAI tool format to Anthropic format."""
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the web",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                },
            }
        ]

        converted = AnthropicModel._convert_tools(openai_tools)
        assert len(converted) == 1
        assert converted[0]["name"] == "search"
        assert converted[0]["description"] == "Search the web"
        assert "input_schema" in converted[0]

    def test_convert_tools_native_format(self) -> None:
        """Should pass through Anthropic-native tool definitions."""
        native_tools = [
            {
                "name": "search",
                "description": "Search the web",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            }
        ]

        converted = AnthropicModel._convert_tools(native_tools)
        assert converted == native_tools

    @pytest.mark.asyncio
    async def test_generate_http_error(self) -> None:
        """Should raise ModelError on HTTP errors."""
        model = AnthropicModel(model_name="claude-sonnet-4-6", api_key="test-key")

        mock_request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        mock_response = httpx.Response(529, text="Overloaded", request=mock_request)
        error = httpx.HTTPStatusError(
            "Overloaded", request=mock_request, response=mock_response
        )

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = error
            with pytest.raises(ModelError) as exc_info:
                await model.generate(messages=[{"role": "user", "content": "Hi"}])
            assert exc_info.value.status_code == 529
