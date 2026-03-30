"""Edge case tests for model backends and base model interface.

Covers ModelResponse edge cases, generate_text error propagation,
Anthropic tool conversion edge cases, and model initialization.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from agent_forge.exceptions import ModelError
from agent_forge.models.anthropic import AnthropicModel
from agent_forge.models.base import ModelResponse
from agent_forge.models.openai import OpenAIModel
from tests.conftest import MockModel


class TestModelResponseEdgeCases:
    """Edge cases for ModelResponse data model."""

    def test_model_response_with_empty_tool_calls(self) -> None:
        """ModelResponse with empty tool_calls list should work."""
        resp = ModelResponse(content="Hello", tool_calls=[])
        assert resp.tool_calls == []

    def test_model_response_with_multiple_tool_calls(self) -> None:
        """ModelResponse should handle multiple tool calls."""
        resp = ModelResponse(
            content="",
            tool_calls=[
                {"name": "tool1", "arguments": {"a": 1}},
                {"name": "tool2", "arguments": {"b": 2}},
                {"name": "tool3", "arguments": {"c": 3}},
            ],
        )
        assert len(resp.tool_calls) == 3

    def test_model_response_preserves_raw(self) -> None:
        """Raw response data should be preserved."""
        raw_data: dict[str, Any] = {
            "id": "msg_123",
            "type": "message",
            "custom_field": [1, 2, 3],
        }
        resp = ModelResponse(raw=raw_data)
        assert resp.raw == raw_data


class TestMockModelBehavior:
    """Tests for MockModel response cycling."""

    @pytest.mark.asyncio
    async def test_mock_cycles_through_responses(self) -> None:
        """MockModel should cycle through responses in order."""
        model = MockModel(responses=["first", "second", "third"])
        r1 = await model.generate_text("q1")
        r2 = await model.generate_text("q2")
        r3 = await model.generate_text("q3")
        assert r1 == "first"
        assert r2 == "second"
        assert r3 == "third"

    @pytest.mark.asyncio
    async def test_mock_repeats_last_response(self) -> None:
        """MockModel should repeat the last response when exhausted."""
        model = MockModel(responses=["only"])
        r1 = await model.generate_text("q1")
        r2 = await model.generate_text("q2")
        r3 = await model.generate_text("q3")
        assert r1 == "only"
        assert r2 == "only"
        assert r3 == "only"

    @pytest.mark.asyncio
    async def test_mock_usage_statistics(self) -> None:
        """MockModel should return usage statistics."""
        model = MockModel(responses=["test"])
        resp = await model.generate(messages=[{"role": "user", "content": "hi"}])
        assert resp.usage["prompt_tokens"] == 10
        assert resp.usage["completion_tokens"] == 20
        assert resp.usage["total_tokens"] == 30

    @pytest.mark.asyncio
    async def test_mock_model_name(self) -> None:
        """MockModel should use the configured model name."""
        model = MockModel(model_name="test-model-v2")
        resp = await model.generate(messages=[{"role": "user", "content": "hi"}])
        assert resp.model == "test-model-v2"


class TestOpenAIModelEdgeCases:
    """Edge cases for OpenAI model backend."""

    def test_env_var_fallback(self) -> None:
        """Should fall back to env var for API key."""
        import os

        backup = os.environ.pop("OPENAI_API_KEY", None)
        try:
            os.environ["OPENAI_API_KEY"] = "env-key-123"
            model = OpenAIModel(model_name="gpt-4o")
            assert model.api_key == "env-key-123"
        finally:
            if backup:
                os.environ["OPENAI_API_KEY"] = backup
            else:
                os.environ.pop("OPENAI_API_KEY", None)

    @pytest.mark.asyncio
    async def test_generate_empty_choices_raises(self) -> None:
        """Empty choices list from API causes IndexError (known edge case)."""
        mock_response = httpx.Response(
            200,
            json={"choices": [], "usage": {}},
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        )

        model = OpenAIModel(model_name="gpt-4o", api_key="test")
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            with pytest.raises(IndexError):
                await model.generate(messages=[{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_generate_null_content(self) -> None:
        """Should handle null content in response (tool call only)."""
        mock_response = httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": None, "tool_calls": []}}],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 0,
                    "total_tokens": 5,
                },
            },
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        )

        model = OpenAIModel(model_name="gpt-4o", api_key="test")
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            result = await model.generate(messages=[{"role": "user", "content": "hi"}])
        assert result.content == ""

    @pytest.mark.asyncio
    async def test_generate_invalid_tool_arguments_raise_model_error(self) -> None:
        """Malformed tool arguments should be wrapped in ModelError."""
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
                                        "arguments": "{not-json}",
                                    },
                                }
                            ],
                        }
                    }
                ],
                "usage": {},
            },
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        )

        model = OpenAIModel(model_name="gpt-4o", api_key="test")
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            with pytest.raises(ModelError, match="Failed to parse tool call arguments"):
                await model.generate(messages=[{"role": "user", "content": "hi"}])

    def test_default_params_passed(self) -> None:
        """Default params should be stored and available."""
        model = OpenAIModel(
            model_name="gpt-4o",
            api_key="test",
            default_params={"temperature": 0.5, "max_tokens": 100},
        )
        assert model.default_params["temperature"] == 0.5
        assert model.default_params["max_tokens"] == 100


class TestAnthropicModelEdgeCases:
    """Edge cases for Anthropic model backend."""

    def test_default_max_tokens(self) -> None:
        """Should have default max_tokens of 4096."""
        model = AnthropicModel(api_key="test")
        assert model.default_params["max_tokens"] == 4096

    def test_custom_default_params_merge(self) -> None:
        """Custom default params should merge with defaults."""
        model = AnthropicModel(
            api_key="test",
            default_params={"temperature": 0.3},
        )
        assert model.default_params["max_tokens"] == 4096
        assert model.default_params["temperature"] == 0.3

    def test_custom_default_params_override(self) -> None:
        """Custom max_tokens should override default."""
        model = AnthropicModel(
            api_key="test",
            default_params={"max_tokens": 1024},
        )
        assert model.default_params["max_tokens"] == 1024

    @pytest.mark.asyncio
    async def test_generate_multiple_text_blocks(self) -> None:
        """Should concatenate multiple text blocks."""
        mock_response = httpx.Response(
            200,
            json={
                "content": [
                    {"type": "text", "text": "Part 1. "},
                    {"type": "text", "text": "Part 2."},
                ],
                "usage": {"input_tokens": 5, "output_tokens": 10},
            },
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
        )

        model = AnthropicModel(api_key="test")
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            result = await model.generate(messages=[{"role": "user", "content": "hi"}])
        assert "Part 1." in result.content
        assert "Part 2." in result.content

    @pytest.mark.asyncio
    async def test_generate_empty_content_blocks(self) -> None:
        """Should handle empty content blocks list."""
        mock_response = httpx.Response(
            200,
            json={
                "content": [],
                "usage": {"input_tokens": 5, "output_tokens": 0},
            },
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
        )

        model = AnthropicModel(api_key="test")
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            result = await model.generate(messages=[{"role": "user", "content": "hi"}])
        assert result.content == ""

    @pytest.mark.asyncio
    async def test_generate_multiple_system_messages(self) -> None:
        """Multiple system messages should be joined."""
        mock_response = httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "OK"}],
                "usage": {"input_tokens": 10, "output_tokens": 1},
            },
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
        )

        model = AnthropicModel(api_key="test")
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            await model.generate(
                messages=[
                    {"role": "system", "content": "Rule 1"},
                    {"role": "system", "content": "Rule 2"},
                    {"role": "user", "content": "Hello"},
                ]
            )

            call_args = mock_post.call_args
            payload = call_args.kwargs.get("json", call_args[1].get("json", {}))
            assert "Rule 1" in payload["system"]
            assert "Rule 2" in payload["system"]
            assert all(m["role"] != "system" for m in payload["messages"])

    def test_convert_tools_unknown_format(self) -> None:
        """Should handle tool defs that don't match any known format."""
        unknown_tools = [
            {
                "name": "custom",
                "description": "Custom tool",
                "parameters": {"type": "object"},
            }
        ]
        converted = AnthropicModel._convert_tools(unknown_tools)
        assert len(converted) == 1
        assert converted[0]["name"] == "custom"

    def test_convert_tools_empty_list(self) -> None:
        """Should handle empty tools list."""
        converted = AnthropicModel._convert_tools([])
        assert converted == []

    @pytest.mark.asyncio
    async def test_network_error_raises_model_error(self) -> None:
        """Network errors should be wrapped in ModelError."""
        model = AnthropicModel(api_key="test")
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.ConnectError("DNS resolution failed")
            with pytest.raises(ModelError) as exc_info:
                await model.generate(messages=[{"role": "user", "content": "hi"}])
            assert exc_info.value.model == "claude-sonnet-4-6"
            assert exc_info.value.status_code is None


class TestBaseModelGenerateText:
    """Edge cases for the BaseModel.generate_text convenience method."""

    @pytest.mark.asyncio
    async def test_generate_text_no_system(self) -> None:
        """generate_text without system message should only send user message."""
        model = MockModel(responses=["response"])
        result = await model.generate_text("prompt only")
        assert result == "response"

    @pytest.mark.asyncio
    async def test_generate_text_empty_prompt(self) -> None:
        """generate_text with empty prompt should still work."""
        model = MockModel(responses=["response"])
        result = await model.generate_text("")
        assert result == "response"
