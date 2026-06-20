from __future__ import annotations

import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Ticket:
    user: str
    message: str


@dataclass(frozen=True)
class AgentResult:
    category: str
    tool_used: str
    reply: str
    passed_eval: bool


def classify_ticket(ticket: Ticket) -> str:
    text = ticket.message.lower()
    if "refund" in text or "charged" in text:
        return "billing"
    if "login" in text or "password" in text:
        return "account"
    return "general"


def draft_reply(ticket: Ticket, category: str) -> str:
    templates = {
        "billing": "I will check the charge details and confirm the refund path.",
        "account": "I will help you recover access and verify the account safely.",
        "general": "I will route this to the right support path with the details provided.",
    }
    return f"Hi {ticket.user}. {templates[category]}"


def evaluate(ticket: Ticket, result: AgentResult) -> bool:
    if ticket.user not in result.reply:
        return False
    if result.category == "billing" and "refund" not in result.reply.lower():
        return False
    if result.category == "account" and "account" not in result.reply.lower():
        return False
    return True


def run_agent(ticket: Ticket) -> AgentResult:
    category = classify_ticket(ticket)
    reply = draft_reply(ticket, category)
    provisional = AgentResult(
        category=category,
        tool_used="classify_ticket,draft_reply",
        reply=reply,
        passed_eval=False,
    )
    return AgentResult(
        category=provisional.category,
        tool_used=provisional.tool_used,
        reply=provisional.reply,
        passed_eval=evaluate(ticket, provisional),
    )


def main() -> None:
    examples = [
        Ticket(user="Aki", message="I was charged twice and need a refund."),
        Ticket(user="Mina", message="I forgot my password and cannot login."),
        Ticket(user="Ren", message="Where can I find the product roadmap?"),
    ]
    results = [asdict(run_agent(ticket)) for ticket in examples]
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

