"""Plain Python function tool wrapper.

Wraps ordinary Python functions (sync or async) as tools that the
executor can invoke. Handles schema generation from type hints and
docstrings, argument validation, and async execution.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import TYPE_CHECKING, Any, get_type_hints

from pydantic import BaseModel

from agent_forge.exceptions import ToolError

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


class ToolParameter(BaseModel):
    """Schema for a single tool parameter.

    Attributes:
        name: Parameter name.
        type: JSON Schema type string.
        description: Human-readable description.
        required: Whether the parameter is required.
    """

    name: str
    type: str = "string"
    description: str = ""
    required: bool = True


class FunctionTool:
    """Wraps a Python callable as an agent-forge tool.

    Extracts parameter schemas from type hints, validates arguments
    at call time, and handles both sync and async functions.

    Args:
        fn: The Python function to wrap.
        name: Override tool name. Defaults to function name.
        description: Override description. Defaults to docstring.
    """

    PYTHON_TYPE_MAP: dict[type, str] = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }

    def __init__(
        self,
        fn: Callable[..., Any],
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        self.fn = fn
        self.name = name or fn.__name__
        self.description = description or (inspect.getdoc(fn) or "")
        self._is_async = asyncio.iscoroutinefunction(fn)
        self._parameters = self._extract_parameters()
        logger.debug(
            "FunctionTool created: %s (%d params)", self.name, len(self._parameters)
        )

    def _extract_parameters(self) -> list[ToolParameter]:
        """Extract parameter schemas from the function signature.

        Returns:
            List of ToolParameter describing each function argument.
        """
        sig = inspect.signature(self.fn)
        try:
            hints = get_type_hints(self.fn)
        except Exception:
            hints = {}

        params: list[ToolParameter] = []
        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            python_type = hints.get(param_name, str)
            origin = getattr(python_type, "__origin__", None)
            if origin is not None:
                python_type = origin

            json_type = self.PYTHON_TYPE_MAP.get(python_type, "string")
            has_default = param.default is not inspect.Parameter.empty

            params.append(
                ToolParameter(
                    name=param_name,
                    type=json_type,
                    description=f"Parameter '{param_name}'",
                    required=not has_default,
                )
            )
        return params

    @property
    def parameters(self) -> list[ToolParameter]:
        """Return the tool's parameter schemas."""
        return list(self._parameters)

    def to_schema(self) -> dict[str, Any]:
        """Generate an OpenAI-compatible tool schema.

        Returns:
            Dict representing the tool in OpenAI function-calling format.
        """
        properties: dict[str, Any] = {}
        required: list[str] = []
        for param in self._parameters:
            properties[param.name] = {
                "type": param.type,
                "description": param.description,
            }
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    async def execute(self, **kwargs: Any) -> Any:
        """Execute the wrapped function with the given arguments.

        Args:
            **kwargs: Arguments to pass to the function.

        Returns:
            The function's return value.

        Raises:
            ToolError: If the function raises an exception.
        """
        logger.debug(
            "Executing tool '%s' with args: %s", self.name, list(kwargs.keys())
        )
        try:
            if self._is_async:
                result = await self.fn(**kwargs)
            else:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(None, lambda: self.fn(**kwargs))
            return result
        except Exception as exc:
            raise ToolError(
                message=str(exc),
                tool_name=self.name,
            ) from exc
