from __future__ import annotations

from pathlib import Path

import pytest

from agent import Action
from app import RestaurantOrderApplication
from order_db import create_order, get_order, get_saved_order, seed


@pytest.fixture
def application(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> RestaurantOrderApplication:
    database_path = tmp_path / "integration.sqlite3"
    monkeypatch.setenv("DB_PATH", str(database_path))
    return RestaurantOrderApplication(database_path, request_id_factory=lambda: "integration-request-1")


def add_chicken(app: RestaurantOrderApplication, session_id: str, quantity: int = 1) -> str:
    result = app.submit(
        session_id,
        Action(
            "add_item",
            query="chicken ranch medium",
            quantity=quantity,
            modifiers=["no_onion"],
        ),
    )
    assert result.outcome == "item_added"
    return result.data["order"]["items"][0]["line_id"]


def test_real_tools_search_add_and_retrieve_draft(application: RestaurantOrderApplication) -> None:
    session_id = application.start_session("integration-search-add")

    search = application.orchestrator.tools.search_menu("chicken ranch medium")
    assert search["ok"] is True
    assert [item["item_id"] for item in search["data"]] == ["chicken_ranch_medium"]

    line_id = add_chicken(application, session_id, quantity=2)
    draft = get_order(session_id)
    assert draft["ok"] is True
    assert draft["data"]["items"][0]["line_id"] == line_id
    assert draft["data"]["total_piasters"] == 30000


def test_real_tools_update_and_remove(application: RestaurantOrderApplication) -> None:
    session_id = application.start_session("integration-update-remove")
    line_id = add_chicken(application, session_id)

    updated = application.submit(
        session_id,
        Action("update_item", line_id=line_id, quantity=3, modifiers=[]),
    )
    assert updated.outcome == "item_updated"
    assert get_order(session_id)["data"]["items"][0]["quantity"] == 3
    assert get_order(session_id)["data"]["total_piasters"] == 45000

    removed = application.submit(session_id, Action("remove_item", line_id=line_id))
    assert removed.outcome == "item_removed"
    assert get_order(session_id)["data"]["items"] == []


def test_summary_blocks_save_until_explicit_confirmation(application: RestaurantOrderApplication) -> None:
    session_id = application.start_session("integration-confirm")
    add_chicken(application, session_id, quantity=2)

    summary = application.submit(session_id, Action("show_summary"))
    assert summary.outcome == "summary_ready"
    assert summary.data["total_piasters"] == 30000
    assert get_saved_order(1)["ok"] is False

    saved = application.submit(session_id, Action("confirm_order"))
    assert saved.outcome == "order_saved"
    stored = get_saved_order(saved.data["order_id"])
    assert stored["ok"] is True
    assert stored["data"]["status"] == "confirmed"
    assert stored["data"]["total_piasters"] == 30000


def test_price_change_requires_reconfirmation_and_same_request_id(
    application: RestaurantOrderApplication,
) -> None:
    session_id = application.start_session("integration-price-change")
    add_chicken(application, session_id, quantity=2)
    application.submit(session_id, Action("show_summary"))

    seed.set_price("chicken_ranch_medium", 16000)
    changed = application.submit(session_id, Action("confirm_order"))
    assert changed.outcome == "price_changed"
    assert changed.data["total_piasters"] == 32000
    assert application.orchestrator.sessions.get(session_id).pending_request_id == "integration-request-1"

    saved = application.submit(session_id, Action("confirm_order"))
    assert saved.outcome == "order_saved"
    # The real database returns the same record for a repeated request ID.
    retry = create_order(session_id, "integration-request-1")
    assert retry["ok"] is True
    assert retry["data"]["order_id"] == saved.data["order_id"]
    assert retry["data"]["total_piasters"] == 32000


def test_real_tool_validation_reaches_orchestrator_safely(
    application: RestaurantOrderApplication,
) -> None:
    session_id = application.start_session("integration-validation")

    unavailable = application.submit(
        session_id,
        Action("add_item", query="veggie supreme", quantity=1, modifiers=[]),
    )
    invalid_modifier = application.submit(
        session_id,
        Action(
            "add_item",
            query="cola",
            item_id="cola_can",
            quantity=1,
            modifiers=["no_onion"],
        ),
    )

    assert unavailable.outcome == "unavailable"
    assert invalid_modifier.outcome == "invalid_modifier"
    assert get_order(session_id)["data"]["items"] == []
