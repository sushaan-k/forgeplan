"""Abstract base interface for LLM models.

All model backends (OpenAI, Anthropic, local) implement this interface
so the planner and executor can work with any LLM.
"""

from __future__ import annotations

import abc
from typing import Any

from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field


class ModelResponse(PydanticBaseModel):
    """Standardized response from an LLM call.

    Attributes:
        content: The text content of the response.
        tool_calls: Any tool/function calls the model wants to make.
        raw: The raw response object from the underlying API.
        usage: Token usage statistics.
        model: The model identifier that produced this response.
    """

    content: str = ""
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)
    usage: dict[str, int] = Field(default_factory=dict)
    model: str = ""


class BaseModel(abc.ABC):
    """Abstract interface for LLM model backends.

    Subclasses must implement `generate` to call the underlying API
    and return a standardized ModelResponse.

    Args:
        model_name: Identifier for the model (e.g. "gpt-4o", "claude-sonnet-4-6").
        api_key: Optional API key. If None, reads from environment.
        base_url: Optional base URL override for the API endpoint.
        default_params: Default generation parameters (temperature, etc.).
    """

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        base_url: str | None = None,
        default_params: dict[str, Any] | None = None,
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.default_params: dict[str, Any] = default_params or {}

    @abc.abstractmethod
    async def generate(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Generate a response from the LLM.

        Args:
            messages: Chat messages in OpenAI-compatible format.
            tools: Tool/function definitions available to the model.
            **kwargs: Additional generation parameters.

        Returns:
            Standardized ModelResponse.
        """

    async def generate_text(
        self,
        prompt: str,
        system: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Convenience method: generate plain text from a single prompt.

        Args:
            prompt: The user prompt.
            system: Optional system message.
            **kwargs: Additional generation parameters.

        Returns:
            The text content of the response.
        """
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = await self.generate(messages, **kwargs)
        return response.content
