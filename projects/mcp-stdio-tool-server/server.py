from __future__ import annotations

import argparse
import json
import sys
from typing import Any


TOOLS = [
    {
        "name": "summarize_text",
        "description": "Return a short deterministic summary of text.",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    }
]


def handle(request: dict[str, Any]) -> dict[str, Any]:
    method = request.get("method")
    request_id = request.get("id")

    if method == "initialize":
        return {"jsonrpc": "2.0", "id": request_id, "result": {"serverInfo": {"name": "learning-mcp-stdio"}}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": request_id, "result": {"tools": TOOLS}}
    if method == "tools/call":
        params = request.get("params", {})
        if params.get("name") != "summarize_text":
            return error(request_id, -32602, "unknown tool")
        text = params.get("arguments", {}).get("text", "")
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"content": [{"type": "text", "text": summarize(text)}]},
        }
    return error(request_id, -32601, "method not found")


def summarize(text: str) -> str:
    words = text.split()
    if len(words) <= 8:
        return text
    return " ".join(words[:8]) + "..."


def error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def serve() -> None:
    for line in sys.stdin:
        if not line.strip():
            continue
        response = handle(json.loads(line))
        print(json.dumps(response), flush=True)


def demo() -> None:
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": "summarize_text", "arguments": {"text": "MCP connects models to tools with structured calls."}},
    }
    print(json.dumps(handle(request), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    if args.demo:
        demo()
    else:
        serve()


if __name__ == "__main__":
    main()

