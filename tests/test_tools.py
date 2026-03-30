"""Tests for tool interfaces (FunctionTool and MCPTool)."""

from __future__ import annotations

import pytest

from agent_forge.exceptions import ToolError
from agent_forge.tools.function import FunctionTool


class TestFunctionTool:
    """Tests for wrapping Python functions as tools."""

    def test_wrap_sync_function(self, sync_tool: FunctionTool) -> None:
        """Should wrap a sync function."""
        assert sync_tool.name == "add"
        assert len(sync_tool.parameters) == 2

    def test_wrap_async_function(self, async_tool: FunctionTool) -> None:
        """Should wrap an async function."""
        assert async_tool.name == "multiply"

    @pytest.mark.asyncio
    async def test_execute_sync(self, sync_tool: FunctionTool) -> None:
        """Should execute sync functions correctly."""
        result = await sync_tool.execute(a=3, b=4)
        assert result == 7

    @pytest.mark.asyncio
    async def test_execute_async(self, async_tool: FunctionTool) -> None:
        """Should execute async functions correctly."""
        result = await async_tool.execute(x=5, y=6)
        assert result == 30

    @pytest.mark.asyncio
    async def test_execute_error_wrapping(self) -> None:
        """Should wrap exceptions in ToolError."""

        def failing_fn() -> None:
            raise ValueError("deliberate error")

        tool = FunctionTool(fn=failing_fn, name="fail")
        with pytest.raises(ToolError) as exc_info:
            await tool.execute()
        assert "deliberate error" in str(exc_info.value)

    def test_to_schema(self, sync_tool: FunctionTool) -> None:
        """Should generate OpenAI-compatible tool schema."""
        schema = sync_tool.to_schema()
        assert schema["type"] == "function"
        fn = schema["function"]
        assert fn["name"] == "add"
        assert "a" in fn["parameters"]["properties"]
        assert "b" in fn["parameters"]["properties"]
        assert "a" in fn["parameters"]["required"]

    def test_parameter_extraction_types(self) -> None:
        """Should map Python types to JSON Schema types."""

        def typed_fn(name: str, count: int, rate: float, active: bool) -> None:
            pass

        tool = FunctionTool(fn=typed_fn)
        params = {p.name: p.type for p in tool.parameters}
        assert params["name"] == "string"
        assert params["count"] == "integer"
        assert params["rate"] == "number"
        assert params["active"] == "boolean"

    def test_optional_parameters(self) -> None:
        """Parameters with defaults should not be required."""

        def fn_with_defaults(required: str, optional: str = "default") -> str:
            return required + optional

        tool = FunctionTool(fn=fn_with_defaults)
        required_names = [p.name for p in tool.parameters if p.required]
        optional_names = [p.name for p in tool.parameters if not p.required]
        assert "required" in required_names
        assert "optional" in optional_names

    def test_custom_name_and_description(self) -> None:
        """Should accept custom name and description."""

        def my_fn() -> None:
            pass

        tool = FunctionTool(fn=my_fn, name="custom_name", description="Custom desc")
        assert tool.name == "custom_name"
        assert tool.description == "Custom desc"
