from __future__ import annotations

from agent import Action, OrderOrchestrator
from agent.testing_tools import InMemoryOrderTools

MENU = [
    {
        "item_id": "chicken_ranch_medium",
        "name": "Chicken Ranch Medium",
        "size": "medium",
        "price_piasters": 15000,
        "available": True,
        "allowed_modifiers": ["no_onion"],
    },
    {
        "item_id": "sold_out_pizza",
        "name": "Sold Out Pizza",
        "size": "medium",
        "price_piasters": 12000,
        "available": False,
        "allowed_modifiers": [],
    },
]


def setup() -> tuple[OrderOrchestrator, InMemoryOrderTools, str]:
    tools = InMemoryOrderTools(MENU)
    ids = iter(["request-1", "request-2"])
    agent = OrderOrchestrator(tools, request_id_factory=lambda: next(ids))
    return agent, tools, agent.start_session("session-1")


def add_valid(agent: OrderOrchestrator, session_id: str, quantity: int = 2) -> str:
    result = agent.handle_action(
        session_id,
        Action("add_item", query="Chicken Ranch", quantity=quantity, modifiers=[]),
    )
    assert result.outcome == "item_added"
    return result.data["order"]["items"][0]["line_id"]


def test_add_valid_item() -> None:
    agent, tools, session_id = setup()
    line_id = add_valid(agent, session_id)
    assert tools.orders[session_id]["items"][0]["line_id"] == line_id
    assert tools.orders[session_id]["total_piasters"] == 30000


def test_missing_information_does_not_create_an_order() -> None:
    agent, tools, session_id = setup()
    result = agent.handle_action(session_id, Action("add_item", quantity=1))
    assert result.outcome == "missing_item"
    assert tools.create_calls == []


def test_unsupported_item() -> None:
    agent, _, session_id = setup()
    result = agent.handle_action(session_id, Action("add_item", query="mushroom", quantity=1))
    assert result.outcome == "not_found"


def test_unavailable_item() -> None:
    agent, _, session_id = setup()
    result = agent.handle_action(session_id, Action("add_item", query="Sold Out", quantity=1))
    assert result.outcome == "unavailable"


def test_invalid_modifier() -> None:
    agent, _, session_id = setup()
    result = agent.handle_action(
        session_id,
        Action("add_item", query="Chicken Ranch", quantity=1, modifiers=["extra_cheese"]),
    )
    assert result.outcome == "invalid_modifier"


def test_update_quantity_and_modifiers() -> None:
    agent, tools, session_id = setup()
    line_id = add_valid(agent, session_id)
    result = agent.handle_action(
        session_id, Action("update_item", line_id=line_id, quantity=1, modifiers=["no_onion"])
    )
    assert result.outcome == "item_updated"
    assert tools.orders[session_id]["items"][0]["quantity"] == 1
    assert tools.orders[session_id]["items"][0]["modifiers"] == ["no_onion"]


def test_remove_item() -> None:
    agent, tools, session_id = setup()
    line_id = add_valid(agent, session_id)
    result = agent.handle_action(session_id, Action("remove_item", line_id=line_id))
    assert result.outcome == "item_removed"
    assert tools.orders[session_id]["items"] == []


def test_summary_before_confirmation_does_not_save() -> None:
    agent, tools, session_id = setup()
    add_valid(agent, session_id)
    result = agent.handle_action(session_id, Action("show_summary"))
    assert result.outcome == "summary_ready"
    assert result.data["total_piasters"] == 30000
    assert tools.create_calls == []


def test_confirm_without_summary_never_calls_create_order() -> None:
    agent, tools, session_id = setup()
    add_valid(agent, session_id)
    result = agent.handle_action(session_id, Action("confirm_order"))
    assert result.outcome == "summary_required"
    assert tools.create_calls == []


def test_order_change_after_summary_requires_a_new_summary() -> None:
    agent, tools, session_id = setup()
    line_id = add_valid(agent, session_id)
    agent.handle_action(session_id, Action("show_summary"))
    agent.handle_action(session_id, Action("update_item", line_id=line_id, quantity=1))
    result = agent.handle_action(session_id, Action("confirm_order"))
    assert result.outcome == "summary_required"
    assert tools.create_calls == []


def test_successful_explicit_confirmation() -> None:
    agent, tools, session_id = setup()
    add_valid(agent, session_id)
    agent.handle_action(session_id, Action("show_summary"))
    result = agent.handle_action(session_id, Action("confirm_order"))
    assert result.outcome == "order_saved"
    assert result.data["order_id"] == 1
    assert tools.create_calls == ["request-1"]


def test_price_changed_requires_new_confirmation() -> None:
    agent, tools, session_id = setup()
    add_valid(agent, session_id)
    agent.handle_action(session_id, Action("show_summary"))
    tools.next_create_error = "PRICE_CHANGED"
    tools.price_change_on_next_create = ("chicken_ranch_medium", 16000)
    changed = agent.handle_action(session_id, Action("confirm_order"))
    assert changed.outcome == "price_changed"
    assert changed.data["total_piasters"] == 32000
    saved = agent.handle_action(session_id, Action("confirm_order"))
    assert saved.outcome == "order_saved"
    assert tools.create_calls == ["request-1", "request-1"]


def test_save_retry_reuses_request_id_without_duplicate() -> None:
    agent, tools, session_id = setup()
    add_valid(agent, session_id)
    agent.handle_action(session_id, Action("show_summary"))
    tools.next_create_error = "SAVE_FAILED"
    failed = agent.handle_action(session_id, Action("confirm_order"))
    retried = agent.handle_action(session_id, Action("confirm_order"))
    assert failed.outcome == "save_failed"
    assert retried.outcome == "order_saved"
    assert tools.create_calls == ["request-1", "request-1"]
    assert len(tools.saved) == 1


def test_tool_failure_does_not_crash_orchestration() -> None:
    agent, tools, session_id = setup()
    tools.raise_on = "get_order"
    result = agent.handle_action(session_id, Action("show_summary"))
    assert result.outcome == "tool_failure"
