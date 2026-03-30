"""Anthropic model interface.

Provides an LLM backend for Claude models via the Anthropic Messages API.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from agent_forge.exceptions import ModelError
from agent_forge.models.base import BaseModel, ModelResponse


class AnthropicModel(BaseModel):
    """LLM interface for Anthropic's Claude models.

    Args:
        model_name: Claude model identifier (e.g. "claude-sonnet-4-6").
        api_key: API key. Falls back to ANTHROPIC_API_KEY env var.
        base_url: API base URL. Defaults to Anthropic's endpoint.
        default_params: Default generation params (max_tokens, etc.).
    """

    API_VERSION = "2023-06-01"

    def __init__(
        self,
        model_name: str = "claude-sonnet-4-6",
        api_key: str | None = None,
        base_url: str | None = None,
        default_params: dict[str, Any] | None = None,
    ) -> None:
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        defaults = {"max_tokens": 4096}
        if default_params:
            defaults.update(default_params)
        super().__init__(
            model_name=model_name,
            api_key=resolved_key,
            base_url=base_url or "https://api.anthropic.com/v1",
            default_params=defaults,
        )

    async def generate(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Call the Anthropic Messages API.

        Args:
            messages: Chat messages. System messages are extracted and
                sent via the top-level `system` parameter.
            tools: Tool definitions in Anthropic tool-use format.
            **kwargs: Override generation parameters.

        Returns:
            Standardized ModelResponse.

        Raises:
            ModelError: On HTTP or API errors.
        """
        params: dict[str, Any] = {**self.default_params, **kwargs}

        system_parts: list[str] = []
        chat_messages: list[dict[str, str]] = []
        for msg in messages:
            if msg.get("role") == "system":
                system_parts.append(msg["content"])
            else:
                chat_messages.append(msg)

        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": chat_messages,
            **params,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        if tools:
            payload["tools"] = self._convert_tools(tools)

        headers = {
            "x-api-key": self.api_key or "",
            "anthropic-version": self.API_VERSION,
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                resp = await client.post(
                    f"{self.base_url}/messages",
                    json=payload,
                    headers=headers,
                )
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise ModelError(
                    message=f"API returned {exc.response.status_code}: "
                    f"{exc.response.text[:200]}",
                    model=self.model_name,
                    status_code=exc.response.status_code,
                ) from exc
            except httpx.RequestError as exc:
                raise ModelError(
                    message=f"Request failed: {exc}",
                    model=self.model_name,
                ) from exc

        data = resp.json()
        content_blocks = data.get("content", [])
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []

        for block in content_blocks:
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                tool_calls.append(
                    {
                        "id": block.get("id", ""),
                        "name": block.get("name", ""),
                        "arguments": block.get("input", {}),
                    }
                )

        usage = data.get("usage", {})
        return ModelResponse(
            content="\n".join(text_parts),
            tool_calls=tool_calls,
            raw=data,
            usage={
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": (
                    usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
                ),
            },
            model=self.model_name,
        )

    @staticmethod
    def _convert_tools(
        tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Convert OpenAI-style tool defs to Anthropic format if needed.

        Args:
            tools: Tool definitions (either format accepted).

        Returns:
            Tool definitions in Anthropic format.
        """
        converted: list[dict[str, Any]] = []
        for tool in tools:
            if "function" in tool:
                fn = tool["function"]
                converted.append(
                    {
                        "name": fn.get("name", ""),
                        "description": fn.get("description", ""),
                        "input_schema": fn.get("parameters", {}),
                    }
                )
            elif "input_schema" in tool:
                converted.append(tool)
            else:
                converted.append(
                    {
                        "name": tool.get("name", ""),
                        "description": tool.get("description", ""),
                        "input_schema": tool.get("parameters", {}),
                    }
                )
        return converted
