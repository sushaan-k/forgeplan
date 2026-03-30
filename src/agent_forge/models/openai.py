"""OpenAI-compatible model interface.

Works with OpenAI, Azure OpenAI, and any API that follows the
OpenAI chat completions format (e.g., vLLM, Ollama, Together AI).
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

from agent_forge.exceptions import ModelError
from agent_forge.models.base import BaseModel, ModelResponse


class OpenAIModel(BaseModel):
    """LLM interface for OpenAI-compatible APIs.

    Args:
        model_name: Model identifier (e.g. "gpt-4o", "gpt-4o-mini").
        api_key: API key. Falls back to OPENAI_API_KEY env var.
        base_url: API base URL. Defaults to OpenAI's endpoint.
        default_params: Default generation params (temperature, etc.).
    """

    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        default_params: dict[str, Any] | None = None,
    ) -> None:
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        super().__init__(
            model_name=model_name,
            api_key=resolved_key,
            base_url=base_url or "https://api.openai.com/v1",
            default_params=default_params,
        )

    async def generate(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Call the OpenAI-compatible chat completions endpoint.

        Args:
            messages: Chat messages in OpenAI format.
            tools: Tool definitions in OpenAI function-calling format.
            **kwargs: Override generation parameters.

        Returns:
            Standardized ModelResponse.

        Raises:
            ModelError: On HTTP or API errors.
        """
        params: dict[str, Any] = {**self.default_params, **kwargs}
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            **params,
        }
        if tools:
            payload["tools"] = tools

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                resp = await client.post(
                    f"{self.base_url}/chat/completions",
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

        try:
            data = resp.json()
        except ValueError as exc:
            raise ModelError(
                message=f"Failed to parse OpenAI response JSON: {exc}",
                model=self.model_name,
                status_code=resp.status_code,
            ) from exc
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = message.get("content", "") or ""

        tool_calls_raw = message.get("tool_calls", [])
        tool_calls = []
        for tc in tool_calls_raw:
            fn = tc.get("function", {})
            arguments_raw = fn.get("arguments", "{}")
            try:
                arguments = json.loads(arguments_raw)
            except (TypeError, ValueError) as exc:
                raise ModelError(
                    message=(
                        "Failed to parse tool call arguments for "
                        f"'{fn.get('name', '')}': {exc}"
                    ),
                    model=self.model_name,
                    status_code=resp.status_code,
                ) from exc
            tool_calls.append(
                {
                    "id": tc.get("id", ""),
                    "name": fn.get("name", ""),
                    "arguments": arguments,
                }
            )

        usage = data.get("usage", {})
        return ModelResponse(
            content=content,
            tool_calls=tool_calls,
            raw=data,
            usage={
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
            model=self.model_name,
        )
