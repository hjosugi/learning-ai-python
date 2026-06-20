from server import handle


def test_tools_list() -> None:
    response = handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert response["result"]["tools"][0]["name"] == "summarize_text"


def test_tool_call() -> None:
    response = handle(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "summarize_text",
                "arguments": {"text": "one two three four five six seven eight nine"},
            },
        }
    )
    assert response["result"]["content"][0]["text"] == "one two three four five six seven eight..."


if __name__ == "__main__":
    test_tools_list()
    test_tool_call()
    print("ok")

