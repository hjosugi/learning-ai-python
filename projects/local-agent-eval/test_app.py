from app import Ticket, run_agent


def test_billing_ticket_passes_eval() -> None:
    result = run_agent(Ticket(user="Aki", message="Refund request for a duplicate charge"))
    assert result.category == "billing"
    assert result.passed_eval is True


def test_account_ticket_passes_eval() -> None:
    result = run_agent(Ticket(user="Mina", message="Password reset does not work"))
    assert result.category == "account"
    assert result.passed_eval is True


if __name__ == "__main__":
    test_billing_ticket_passes_eval()
    test_account_ticket_passes_eval()
    print("ok")

