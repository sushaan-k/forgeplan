"""Tests for MCP tool integration."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from agent_forge.exceptions import ToolError
from agent_forge.tools.mcp import MCPTool


class TestMCPTool:
    """Tests for MCP tool client."""

    def test_init(self) -> None:
        """Should initialize with server URL and tool name."""
        tool = MCPTool(
            server_url="http://localhost:3000",
            tool_name="search",
            description="Search tool",
        )
        assert tool.name == "search"
        assert tool.server_url == "http://localhost:3000"
        assert tool.description == "Search tool"

    def test_init_strips_trailing_slash(self) -> None:
        """Should strip trailing slash from server URL."""
        tool = MCPTool(
            server_url="http://localhost:3000/",
            tool_name="test",
        )
        assert tool.server_url == "http://localhost:3000"

    def test_to_schema(self) -> None:
        """Should generate OpenAI-compatible tool schema."""
        tool = MCPTool(
            server_url="http://localhost:3000",
            tool_name="search",
            description="Search the web",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
        )

        schema = tool.to_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "search"
        assert schema["function"]["description"] == "Search the web"
        assert "query" in schema["function"]["parameters"]["properties"]

    @pytest.mark.asyncio
    async def test_execute_success(self) -> None:
        """Should send JSON-RPC request and return result."""
        mock_response = httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "result": {"data": "search results"},
                "id": 1,
            },
            request=httpx.Request("POST", "http://localhost:3000/mcp"),
        )

        tool = MCPTool(
            server_url="http://localhost:3000",
            tool_name="search",
        )

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            result = await tool.execute(query="test")

        assert result == {"data": "search results"}

        # Verify the request payload
        call_args = mock_post.call_args
        payload = call_args.kwargs.get("json", call_args[1].get("json", {}))
        assert payload["method"] == "tools/call"
        assert payload["params"]["name"] == "search"
        assert payload["params"]["arguments"]["query"] == "test"

    @pytest.mark.asyncio
    async def test_execute_error_response(self) -> None:
        """Should raise ToolError when server returns an error."""
        mock_response = httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "error": {"code": -32000, "message": "Tool failed"},
                "id": 1,
            },
            request=httpx.Request("POST", "http://localhost:3000/mcp"),
        )

        tool = MCPTool(
            server_url="http://localhost:3000",
            tool_name="search",
        )

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            with pytest.raises(ToolError) as exc_info:
                await tool.execute(query="test")
            assert "MCP error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_http_error(self) -> None:
        """Should raise ToolError on HTTP errors."""
        tool = MCPTool(
            server_url="http://localhost:3000",
            tool_name="search",
        )

        mock_request = httpx.Request("POST", "http://localhost:3000/mcp")
        mock_response = httpx.Response(500, text="Server Error", request=mock_request)
        error = httpx.HTTPStatusError(
            "Server Error", request=mock_request, response=mock_response
        )

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = error
            with pytest.raises(ToolError):
                await tool.execute(query="test")

    @pytest.mark.asyncio
    async def test_execute_network_error(self) -> None:
        """Should raise ToolError on network errors."""
        tool = MCPTool(
            server_url="http://localhost:3000",
            tool_name="search",
        )

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.ConnectError("Connection refused")
            with pytest.raises(ToolError) as exc_info:
                await tool.execute(query="test")
            assert "Failed to reach MCP server" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_discover_success(self) -> None:
        """Should discover tools from MCP server."""
        mock_response = httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "result": {
                    "tools": [
                        {
                            "name": "search",
                            "description": "Search the web",
                            "inputSchema": {
                                "type": "object",
                                "properties": {"query": {"type": "string"}},
                            },
                        },
                        {
                            "name": "fetch",
                            "description": "Fetch a URL",
                            "inputSchema": {
                                "type": "object",
                                "properties": {"url": {"type": "string"}},
                            },
                        },
                    ]
                },
                "id": 1,
            },
            request=httpx.Request("POST", "http://localhost:3000/mcp"),
        )

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            tools = await MCPTool.discover("http://localhost:3000")

        assert len(tools) == 2
        assert tools[0].name == "search"
        assert tools[1].name == "fetch"
        assert tools[0].server_url == "http://localhost:3000"

    @pytest.mark.asyncio
    async def test_discover_failure(self) -> None:
        """Should raise ToolError when discovery fails."""
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.ConnectError("Connection refused")
            with pytest.raises(ToolError) as exc_info:
                await MCPTool.discover("http://localhost:3000")
            assert "discovery" in str(exc_info.value)
