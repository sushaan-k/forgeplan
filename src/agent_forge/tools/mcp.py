"""MCP (Model Context Protocol) tool integration.

Provides a client for discovering and invoking tools exposed by MCP
servers. Supports stdio and SSE transports.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from agent_forge.exceptions import ToolError

logger = logging.getLogger(__name__)


class MCPTool:
    """Client for a single MCP tool exposed by an MCP server.

    Wraps an MCP server endpoint, handling tool discovery and invocation
    over HTTP/SSE.

    Args:
        server_url: URL of the MCP server.
        tool_name: Name of the specific tool on the server.
        description: Human-readable description of the tool.
        input_schema: JSON Schema describing the tool's input parameters.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        server_url: str,
        tool_name: str,
        description: str = "",
        input_schema: dict[str, Any] | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.server_url = server_url.rstrip("/")
        self.name = tool_name
        self.description = description
        self.input_schema = input_schema or {}
        self.timeout = timeout
        logger.debug("MCPTool created: %s @ %s", self.name, self.server_url)

    def to_schema(self) -> dict[str, Any]:
        """Generate an OpenAI-compatible tool schema.

        Returns:
            Dict representing the tool in OpenAI function-calling format.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }

    async def execute(self, **kwargs: Any) -> Any:
        """Invoke the MCP tool with the given arguments.

        Sends a JSON-RPC style request to the MCP server's tool
        invocation endpoint.

        Args:
            **kwargs: Arguments matching the tool's input_schema.

        Returns:
            The tool's response data.

        Raises:
            ToolError: On network errors or tool execution failures.
        """
        logger.debug(
            "Invoking MCP tool '%s' with args: %s", self.name, list(kwargs.keys())
        )
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": self.name,
                "arguments": kwargs,
            },
            "id": 1,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                resp = await client.post(
                    f"{self.server_url}/mcp",
                    json=payload,
                )
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise ToolError(
                    message=f"MCP server returned HTTP {exc.response.status_code}",
                    tool_name=self.name,
                ) from exc
            except httpx.RequestError as exc:
                raise ToolError(
                    message=f"Failed to reach MCP server: {exc}",
                    tool_name=self.name,
                ) from exc

        data = resp.json()
        if "error" in data:
            raise ToolError(
                message=f"MCP error: {data['error']}",
                tool_name=self.name,
            )
        return data.get("result", data)

    @classmethod
    async def discover(cls, server_url: str, timeout: float = 30.0) -> list[MCPTool]:
        """Discover all tools available on an MCP server.

        Calls the server's tools/list endpoint and creates MCPTool
        instances for each available tool.

        Args:
            server_url: URL of the MCP server.
            timeout: Request timeout in seconds.

        Returns:
            List of MCPTool instances.

        Raises:
            ToolError: If discovery fails.
        """
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "params": {},
            "id": 1,
        }

        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                resp = await client.post(
                    f"{server_url.rstrip('/')}/mcp",
                    json=payload,
                )
                resp.raise_for_status()
            except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                raise ToolError(
                    message=f"MCP discovery failed: {exc}",
                    tool_name="discovery",
                ) from exc

        data = resp.json()
        tools_data = data.get("result", {}).get("tools", [])
        tools: list[MCPTool] = []
        for tool_def in tools_data:
            tools.append(
                cls(
                    server_url=server_url,
                    tool_name=tool_def.get("name", ""),
                    description=tool_def.get("description", ""),
                    input_schema=tool_def.get("inputSchema", {}),
                    timeout=timeout,
                )
            )
        logger.info("Discovered %d tools from %s", len(tools), server_url)
        return tools
