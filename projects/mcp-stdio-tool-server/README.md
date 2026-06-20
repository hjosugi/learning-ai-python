# MCP Stdio Tool Server

A dependency-free MCP-shaped JSON-RPC tool server for learning tool discovery and tool calls.

This is intentionally small. It teaches the protocol shape before adding a real SDK.

## Run A Demo Request

```bash
python3 projects/mcp-stdio-tool-server/server.py --demo
```

## Test

```bash
python3 projects/mcp-stdio-tool-server/test_server.py
```

## What To Learn

- JSON-RPC request/response shape
- `initialize`, `tools/list`, and `tools/call`
- keeping tool behavior deterministic before connecting LLMs
- how an MCP server differs from a normal REST endpoint

