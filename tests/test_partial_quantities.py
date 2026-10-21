"""Focused language/Action bridge and menu audit with real isolated SQLite tools."""

import json

import pytest

from app import RestaurantOrderApplication
from conversation import ConversationService
from conversation.conversation_agent import UnderstandingError, validate_understanding
from conversation.prompts import SYSTEM_PROMPT
from order_db import seed, tools


def extracted(intent, **changes):
    item = {
        "item_name": None,
        "quantity": None,
        "size": None,
        "modifiers": None,
        "position": None,
        "quantity_delta": None,
        "affected_quantity": None,
    }
    item.update(changes)
    return json.dumps({"intent": intent, "items": [item], "missing_information": []})


class Understanding:
    def __init__(self):
        self.raw = ""

    def complete(self, messages):
        return self.raw


@pytest.fixture
def context(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "partial.sqlite3"))
    application = RestaurantOrderApplication()
    provider = Understanding()
    return ConversationService(application, provider), provider


def start(context, quantity=2):
    service, model = context
    model.raw = extracted(
        "add_item", item_name="تشيكن رانش", size="medium", quantity=quantity, modifiers=[]
    )
    assert service.handle("s", "عايز اتنين تشيكن رانش وسط").results[0].outcome == "item_added"
    return service, model


def variants():
    return {
        (tuple(line["modifiers"]), line["quantity"])
        for line in tools.get_order("s")["data"]["items"]
    }


def split(context):
    service, model = start(context)
    model.raw = extracted("update_item", modifiers=["no_onion"], affected_quantity=1)
    reply = service.handle("s", "واحدة من غير بصل")
    assert [result.outcome for result in reply.results] == ["item_updated", "item_added"]
    assert variants() == {((), 1), (("no_onion",), 1)}
    return service, model


def test_partial_modifier_split(context):
    split(context)


def test_explicit_partial_words_override_model_omission(context):
    service, model = start(context)
    # The real Qwen smoke initially omitted scope and would have changed all units.
    model.raw = extracted("update_item", modifiers=["no_onion"])
    service.handle("s", "واحدة من غير بصل")
    assert variants() == {((), 1), (("no_onion",), 1)}


def test_explicit_removal_overrides_model_generic_clarification(context):
    service, model = start(context)
    model.raw = json.dumps({"intent": "clarification", "items": [], "missing_information": []})
    service.handle("s", "شيل واحدة")
    assert variants() == {((), 1)}


@pytest.mark.parametrize(
    "quantity,message,intent,fields,expected",
    [
        (2, "شيل واحدة", "remove_item", {"quantity": 1}, 1),
        (3, "شيل اتنين", "remove_item", {"quantity": 2}, 1),
        (2, "زود واحدة", "update_item", {"quantity_delta": 1}, 3),
        (2, "نقص واحدة", "update_item", {"quantity_delta": -1}, 1),
        (1, "نقص واحدة", "update_item", {"quantity_delta": -1}, 0),
    ],
)
def test_contextual_quantity(context, quantity, message, intent, fields, expected):
    service, model = start(context, quantity)
    model.raw = extracted(intent, **fields)
    assert service.handle("s", message).results
    lines = tools.get_order("s")["data"]["items"]
    assert (lines[0]["quantity"] if lines else 0) == expected


def test_variant_ambiguity_specific_no_mutation(context):
    service, model = split(context)
    before = tools.get_order("s")["data"]
    model.raw = extracted("remove_item", quantity=1)
    reply = service.handle("s", "شيل واحدة")
    assert not reply.results
    assert all(text in reply.text for text in ["Chicken Ranch", "العادية", "من غير بصل", "ولا"])
    assert tools.get_order("s")["data"] == before


def test_summary_and_confirmation_after_split(context, monkeypatch):
    service, model = start(context)
    service.handle("s", "خلاص")
    calls = []
    save = tools.create_order
    monkeypatch.setattr(tools, "create_order", lambda *args: (calls.append(args), save(*args))[1])
    model.raw = extracted("update_item", modifiers=["no_onion"], affected_quantity=1)
    service.handle("s", "خلي واحدة من غير بصل")
    assert service.handle("s", "أيوه أكد").results[0].outcome == "summary_required"
    assert not calls
    summary = service.handle("s", "خلاص")
    draft = tools.get_order("s")["data"]
    assert len(summary.results[0].data["order"]["items"]) == 2
    assert summary.results[0].data["total_piasters"] == draft["total_piasters"]
    assert "300.00" in summary.text
    assert service.handle("s", "أيوه أكد").completed and len(calls) == 1


def test_invalid_partial_modifier_never_reduces_quantity(context):
    service, model = start(context)
    model.raw = extracted("update_item", modifiers=["extra_cheese"], affected_quantity=1)
    reply = service.handle("s", "خلي واحدة بجبنة")
    assert "مش مسموح" in reply.text
    assert variants() == {((), 2)}


def test_split_add_failure_restores_original_quantity(context, monkeypatch):
    service, model = start(context)
    model.raw = extracted("update_item", modifiers=["no_onion"], affected_quantity=1)
    monkeypatch.setattr(
        tools,
        "add_item",
        lambda *args: {
            "ok": False,
            "data": None,
            "error": {"code": "SAVE_FAILED", "message": "test"},
        },
    )
    reply = service.handle("s", "واحدة من غير بصل")
    assert "رجعت الكمية الأصلية" in reply.text
    assert variants() == {((), 2)}


def test_uncertain_committed_add_is_not_blindly_compensated(context, monkeypatch):
    service, model = start(context)
    model.raw = extracted("update_item", modifiers=["no_onion"], affected_quantity=1)
    real_add = tools.add_item

    def uncertain_add(*args):
        real_add(*args)
        raise RuntimeError("response lost after commit")

    monkeypatch.setattr(tools, "add_item", uncertain_add)
    reply = service.handle("s", "واحدة من غير بصل")
    assert "راجع ملخص" in reply.text
    assert variants() == {((), 1), (("no_onion",), 1)}


def test_partial_split_consumes_merged_draft(context):
    service, model = start(context, 3)
    existing = tools.add_item("s", "chicken_ranch_medium", 1, ["no_onion"])["data"]["items"][1]
    model.raw = extracted("update_item", modifiers=["no_onion"], affected_quantity=1, position=1)
    reply = service.handle("s", "خلي واحدة من الأول من غير بصل")
    assert variants() == {((), 2), (("no_onion",), 2)}
    assert any(
        line["line_id"] == existing["line_id"] for line in reply.results[-1].data["order"]["items"]
    )


@pytest.mark.parametrize("affected", [0, -1, True, "1"])
def test_affected_quantity_validation(affected):
    with pytest.raises(UnderstandingError):
        validate_understanding(
            extracted("update_item", modifiers=["no_onion"], affected_quantity=affected)
        )


def test_seed_schema_and_single_authority(context):
    menu = tools.search_menu("")["data"]
    assert len({item["item_id"] for item in menu}) == len(menu) == len(seed.SEED_MENU)
    for item in menu:
        assert set(item) == {
            "item_id",
            "name",
            "size",
            "price_piasters",
            "available",
            "allowed_modifiers",
        }
        assert item["item_id"] and item["name"] and item["size"]
        assert type(item["price_piasters"]) is int and item["price_piasters"] > 0
        assert type(item["available"]) is bool
        assert isinstance(item["allowed_modifiers"], list)
        assert all(isinstance(mod, str) for mod in item["allowed_modifiers"])
        assert len(set(item["allowed_modifiers"])) == len(item["allowed_modifiers"])
        assert item["name"].casefold() not in SYSTEM_PROMPT.casefold()
        assert item["item_id"] not in SYSTEM_PROMPT
        assert str(item["price_piasters"]) not in SYSTEM_PROMPT


@pytest.mark.parametrize(
    "query,expected",
    [
        ("Chicken Ranch", {"chicken_ranch_small", "chicken_ranch_medium", "chicken_ranch_large"}),
        ("تشيكن رانش", {"chicken_ranch_small", "chicken_ranch_medium", "chicken_ranch_large"}),
        ("تشيكن رانش وسط", {"chicken_ranch_medium"}),
        ("تشيكين رانش وسط", {"chicken_ranch_medium"}),
        ("Chicken Ranch medium", {"chicken_ranch_medium"}),
        ("Chicken رانش وسط", {"chicken_ranch_medium"}),
        ("burger", {"chicken_burger_single", "beef_burger_single"}),
        ("برجر", {"chicken_burger_single", "beef_burger_single"}),
        ("تشيكن برجر", {"chicken_burger_single"}),
    ],
)
def test_natural_database_search(context, query, expected):
    result = tools.search_menu(query)
    assert result["ok"]
    assert {item["item_id"] for item in result["data"]} == expected


def test_partial_summary_uses_current_database_price(context):
    service, model = start(context)
    seed.set_price("chicken_ranch_medium", 17500)
    model.raw = extracted("update_item", modifiers=["no_onion"], affected_quantity=1)
    service.handle("s", "واحدة منهم من غير بصل")
    summary = service.handle("s", "خلاص")
    assert (
        summary.results[0].data["total_piasters"]
        == tools.get_order("s")["data"]["total_piasters"]
        == 35000
    )
    assert "350.00" in summary.text
