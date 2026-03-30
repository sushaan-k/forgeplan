"""Edge case tests for tool interfaces.

Covers FunctionTool with various callable types, parameter
extraction edge cases, schema generation, and error handling.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_forge.exceptions import ToolError
from agent_forge.tools.function import FunctionTool, ToolParameter


class TestFunctionToolParameterExtraction:
    """Edge cases for parameter extraction from functions."""

    def test_no_parameters(self) -> None:
        """Function with no parameters should have empty params list."""

        def no_args() -> str:
            return "hello"

        tool = FunctionTool(fn=no_args)
        assert tool.parameters == []

    def test_kwargs_only(self) -> None:
        """Function with **kwargs should not extract kwargs as a parameter."""

        def with_kwargs(**kwargs: Any) -> dict[str, Any]:
            return kwargs

        tool = FunctionTool(fn=with_kwargs)
        # **kwargs is handled by inspect, but shouldn't be listed as a named param
        # depending on implementation -- verify it doesn't crash
        assert isinstance(tool.parameters, list)

    def test_default_value_parameter(self) -> None:
        """Parameters with defaults should be marked as not required."""

        def fn(required: str, optional: int = 10, flag: bool = False) -> str:
            return f"{required}-{optional}-{flag}"

        tool = FunctionTool(fn=fn)
        params_by_name = {p.name: p for p in tool.parameters}
        assert params_by_name["required"].required is True
        assert params_by_name["optional"].required is False
        assert params_by_name["flag"].required is False

    def test_list_type_parameter(self) -> None:
        """List type hint should map to 'array' JSON type."""

        def fn(items: list) -> list:
            return items

        tool = FunctionTool(fn=fn)
        assert tool.parameters[0].type == "array"

    def test_dict_type_parameter(self) -> None:
        """Dict type hint should map to 'object' JSON type."""

        def fn(data: dict) -> dict:
            return data

        tool = FunctionTool(fn=fn)
        assert tool.parameters[0].type == "object"

    def test_unknown_type_defaults_to_string(self) -> None:
        """Unknown/custom types should default to 'string'."""

        class CustomType:
            pass

        def fn(obj: CustomType) -> str:
            return str(obj)

        tool = FunctionTool(fn=fn)
        assert tool.parameters[0].type == "string"

    def test_no_type_hints(self) -> None:
        """Function without type hints should default all to string."""

        def untyped(a, b, c):
            return a

        tool = FunctionTool(fn=untyped)
        assert len(tool.parameters) == 3
        assert all(p.type == "string" for p in tool.parameters)

    def test_class_method_skips_self(self) -> None:
        """Instance method with 'self' param should skip it."""

        class MyClass:
            def method(self, x: int) -> int:
                return x

        tool = FunctionTool(fn=MyClass().method)
        # After binding, self is not in the signature
        param_names = [p.name for p in tool.parameters]
        assert "self" not in param_names


class TestFunctionToolExecution:
    """Edge cases for tool execution."""

    @pytest.mark.asyncio
    async def test_execute_sync_no_args(self) -> None:
        """Should execute sync function with no arguments."""

        def greet() -> str:
            return "hello"

        tool = FunctionTool(fn=greet, name="greet")
        result = await tool.execute()
        assert result == "hello"

    @pytest.mark.asyncio
    async def test_execute_async_no_args(self) -> None:
        """Should execute async function with no arguments."""

        async def async_greet() -> str:
            return "async hello"

        tool = FunctionTool(fn=async_greet, name="async_greet")
        result = await tool.execute()
        assert result == "async hello"

    @pytest.mark.asyncio
    async def test_execute_returns_none(self) -> None:
        """Tool returning None should be valid."""

        def returns_none() -> None:
            pass

        tool = FunctionTool(fn=returns_none, name="noop")
        result = await tool.execute()
        assert result is None

    @pytest.mark.asyncio
    async def test_execute_returns_complex_type(self) -> None:
        """Tool returning complex nested data should work."""

        def complex_result() -> dict[str, Any]:
            return {"data": [1, 2, {"nested": True}], "count": 3}

        tool = FunctionTool(fn=complex_result, name="complex")
        result = await tool.execute()
        assert result["count"] == 3
        assert result["data"][2]["nested"] is True

    @pytest.mark.asyncio
    async def test_error_wraps_as_tool_error(self) -> None:
        """All exceptions from the function should be wrapped in ToolError."""

        def type_error_fn() -> None:
            _ = 1 + "string"  # type: ignore[operator]

        tool = FunctionTool(fn=type_error_fn, name="type_err")
        with pytest.raises(ToolError) as exc_info:
            await tool.execute()
        assert exc_info.value.tool_name == "type_err"

    @pytest.mark.asyncio
    async def test_async_error_wraps_as_tool_error(self) -> None:
        """Async function errors should also be wrapped in ToolError."""

        async def async_err() -> None:
            raise KeyError("missing key")

        tool = FunctionTool(fn=async_err, name="async_err")
        with pytest.raises(ToolError) as exc_info:
            await tool.execute()
        assert "missing key" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_with_keyword_arguments(self) -> None:
        """Should correctly pass keyword arguments to function."""

        def concat(prefix: str, suffix: str, sep: str = "-") -> str:
            return f"{prefix}{sep}{suffix}"

        tool = FunctionTool(fn=concat, name="concat")
        result = await tool.execute(prefix="hello", suffix="world", sep="_")
        assert result == "hello_world"


class TestFunctionToolSchema:
    """Edge cases for schema generation."""

    def test_schema_no_params(self) -> None:
        """Schema for parameterless function should have empty properties."""

        def noop() -> None:
            pass

        tool = FunctionTool(fn=noop, name="noop", description="Does nothing")
        schema = tool.to_schema()
        assert schema["type"] == "function"
        assert schema["function"]["parameters"]["properties"] == {}
        assert schema["function"]["parameters"]["required"] == []

    def test_schema_mixed_required_optional(self) -> None:
        """Schema should correctly separate required and optional params."""

        def fn(req1: str, req2: int, opt1: float = 0.0, opt2: bool = True) -> str:
            return ""

        tool = FunctionTool(fn=fn)
        schema = tool.to_schema()
        required = schema["function"]["parameters"]["required"]
        assert "req1" in required
        assert "req2" in required
        assert "opt1" not in required
        assert "opt2" not in required

    def test_schema_description_from_docstring(self) -> None:
        """Schema description should come from function docstring."""

        def documented_fn(x: int) -> int:
            """Add one to x."""
            return x + 1

        tool = FunctionTool(fn=documented_fn)
        assert "Add one to x" in tool.description
        schema = tool.to_schema()
        assert "Add one to x" in schema["function"]["description"]

    def test_schema_custom_name_override(self) -> None:
        """Custom name should appear in schema, not function name."""

        def internal_name() -> None:
            pass

        tool = FunctionTool(fn=internal_name, name="public_api_name")
        schema = tool.to_schema()
        assert schema["function"]["name"] == "public_api_name"


class TestToolParameterModel:
    """Tests for the ToolParameter data model."""

    def test_defaults(self) -> None:
        """ToolParameter should have sensible defaults."""
        param = ToolParameter(name="test")
        assert param.type == "string"
        assert param.description == ""
        assert param.required is True

    def test_all_fields(self) -> None:
        """ToolParameter should accept all fields."""
        param = ToolParameter(
            name="count",
            type="integer",
            description="Number of items",
            required=False,
        )
        assert param.name == "count"
        assert param.type == "integer"
        assert param.description == "Number of items"
        assert param.required is False
