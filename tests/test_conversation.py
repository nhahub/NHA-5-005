from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy

import pytest

from app import RestaurantOrderApplication
from conversation import ConversationService
from order_db import tools


def item(name=None, quantity=None, size=None, modifiers=None, position=None, delta=None):
    return {
        "item_name": name,
        "quantity": quantity,
        "size": size,
        "modifiers": modifiers,
        "position": position,
        "quantity_delta": delta,
    }


def output(intent, *items, missing=None):
    return json.dumps(
        {"intent": intent, "items": list(items), "missing_information": missing or []}
    )


class FakeUnderstanding:
    """Scripted structured outputs, no natural-language accuracy claim."""

    def __init__(self):
        self.outputs = {}
        self.calls = []

    def complete(self, messages):
        self.calls.append(deepcopy(messages))
        value = self.outputs[messages[-1]["content"]]
        if isinstance(value, Exception):
            raise value
        return value


@pytest.fixture
def service(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "conversation.sqlite3"))
    app = RestaurantOrderApplication(request_id_factory=lambda: "conversation-request")
    model = FakeUnderstanding()
    return ConversationService(app, model), model


def add(service, model, session="s1", quantity=2, modifiers=None):
    model.outputs["add"] = output(
        "add_item", item("Chicken Ranch", quantity, "medium", modifiers or [])
    )
    return service.handle(session, "add")


@pytest.mark.parametrize(
    "utterance",
    [
        "عايز 2 تشيكن رانش وسط",
        "عايز اتنين تشيكن رانش وسط",
        "عايز ٢ Chicken Ranch medium",
    ],
)
def test_arabic_egyptian_quantity_and_mixed_name(service, utterance):
    conversation, model = service
    model.outputs[utterance] = output("add_item", item("Chicken Ranch", 2, "medium", []))
    reply = conversation.handle("s1", utterance)
    assert reply.results[0].outcome == "item_added"
    draft = tools.get_order("s1")["data"]
    assert draft["items"][0]["item_id"] == "chicken_ranch_medium"
    assert draft["items"][0]["quantity"] == 2
    assert model.calls[-1][-1]["content"] == utterance


def test_missing_information_then_followup(service):
    conversation, model = service
    model.outputs["عايز تشيكن رانش"] = output(
        "add_item", item("Chicken Ranch"), missing=["quantity"]
    )
    assert not conversation.handle("s1", "عايز تشيكن رانش").results
    assert tools.get_order("s1")["data"]["items"] == []
    model.outputs["اتنين وسط"] = output("add_item", item("Chicken Ranch", 2, "medium", []))
    assert conversation.handle("s1", "اتنين وسط").results[0].outcome == "item_added"
    assert any(m["content"] == "عايز تشيكن رانش" for m in model.calls[-1])


def test_update_with_current_context(service):
    conversation, model = service
    add(conversation, model)
    model.outputs["خليهم تلاتة"] = output("update_item", item(quantity=3))
    reply = conversation.handle("s1", "خليهم تلاتة")
    assert reply.results[0].outcome == "item_updated"
    assert tools.get_order("s1")["data"]["items"][0]["quantity"] == 3


def test_remove_one_with_context(service):
    conversation, model = service
    add(conversation, model)
    model.outputs["شيل واحدة"] = output("remove_item", item(quantity=1))
    assert conversation.handle("s1", "شيل واحدة").results[0].outcome == "item_updated"
    assert tools.get_order("s1")["data"]["items"][0]["quantity"] == 1
    model.outputs["شيلها"] = output("remove_item", item())
    assert conversation.handle("s1", "شيلها").results[0].outcome == "item_removed"


def test_ambiguous_reference_clarifies_without_mutating(service):
    conversation, model = service
    add(conversation, model)
    tools.add_item("s1", "cola_can", 1, [])
    before = tools.get_order("s1")["data"]
    model.outputs["شيلها"] = output("remove_item", item())
    assert not conversation.handle("s1", "شيلها").results
    assert tools.get_order("s1")["data"] == before


def test_second_reference_is_a_real_line(service):
    conversation, model = service
    add(conversation, model)
    tools.add_item("s1", "cola_can", 1, [])
    model.outputs["شيل التاني"] = output("remove_item", item(position=2))
    assert conversation.handle("s1", "شيل التاني").results[0].outcome == "item_removed"
    assert [line["item_id"] for line in tools.get_order("s1")["data"]["items"]] == [
        "chicken_ranch_medium"
    ]


def test_done_is_summary_and_explicit_confirmation_is_proper_action(service):
    conversation, model = service
    add(conversation, model)
    summary = conversation.handle("s1", "خلاص")
    assert summary.results[0].outcome == "summary_ready"
    assert "300.00" in summary.text
    assert not tools.get_saved_order(1)["ok"]
    confirmed = conversation.handle("s1", "أيوه أكد")
    assert confirmed.results[0].outcome == "order_saved"
    assert confirmed.completed


def test_confirmation_without_summary_is_rejected_by_orchestrator(service):
    conversation, model = service
    add(conversation, model)
    assert conversation.handle("s1", "أيوه أكد").results[0].outcome == "summary_required"
    assert not tools.get_saved_order(1)["ok"]


@pytest.mark.parametrize(
    "bad",
    [
        "not JSON",
        '{"intent":"hack","items":[],"missing_information":[]}',
        '{"intent":"add_item","items":{},"missing_information":[]}',
        output("add_item", item("Chicken Ranch", 0, "medium", [])),
        output("add_item", item("Chicken Ranch", True, "medium", [])),
        output("add_item", item("Chicken Ranch", "2", "medium", [])),
        output("add_item", item("Chicken Ranch", 1.5, "medium", [])),
        output("add_item", item("Chicken Ranch", 1, "medium", "no_onion")),
        output("add_item", {"item_id": "invented"}),
        output("add_item", item("Chicken Ranch", 1, "medium", []), {"bad": 1}),
        output("add_item", item("Chicken Ranch", 1, "medium", [], position=20)),
        '{"intent":"question","intent":"add_item","items":[],"missing_information":[]}',
    ],
)
def test_malformed_unknown_and_invalid_outputs_never_execute(service, bad):
    conversation, model = service
    model.outputs["bad"] = bad
    assert not conversation.handle("s1", "bad").results
    assert tools.get_order("s1")["data"]["items"] == []
    assert not tools.get_saved_order(1)["ok"]


def test_session_history_is_separate(service):
    conversation, model = service
    add(conversation, model, "s1")
    model.outputs["separate"] = output("question")
    conversation.handle("s2", "separate")
    assert not any(m["content"] == "add" for m in model.calls[-1])
    assert tools.get_order("s2")["data"]["items"] == []
    assert conversation.history.for_session("s1") is not conversation.history.for_session("s2")


def test_import_triggers_no_provider_or_audio_initialization():
    code = "import sys; import conversation, app, voice; assert 'groq' not in sys.modules; assert 'pygame' not in sys.modules; assert 'speech_recognition' not in sys.modules; assert 'edge_tts' not in sys.modules"
    result = subprocess.run(
        [sys.executable, "-B", "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""


def test_menu_questions_and_availability_are_tool_grounded(service):
    conversation, model = service
    from order_db import seed

    seed.set_price("chicken_ranch_medium", 17000)
    model.outputs["البيتزا بكام؟"] = output("question", item("Chicken Ranch", size="medium"))
    reply = conversation.handle("s1", "البيتزا بكام؟")
    assert "170.00" in reply.text and "Chicken Ranch" in reply.text
    model.outputs["sizes"] = output("question", item("Chicken Ranch"))
    reply = conversation.handle("s1", "sizes")
    assert all(size in reply.text for size in ["small", "medium", "large"])
    model.outputs["unavailable"] = output("add_item", item("Veggie Supreme", 1, "medium", []))
    assert conversation.handle("s1", "unavailable").results[0].outcome == "unavailable"


def test_model_cannot_invent_confirmation(service):
    conversation, model = service
    add(conversation, model)
    conversation.handle("s1", "خلاص")
    model.outputs["لا متأكدش"] = output("confirm_order")
    assert not conversation.handle("s1", "لا متأكدش").results
    assert not tools.get_saved_order(1)["ok"]


def test_merged_lines_and_stale_history(service):
    conversation, model = service
    add(conversation, model, modifiers=["no_onion"])
    tools.add_item("s1", "chicken_ranch_medium", 1, [])
    model.outputs["change second"] = output("update_item", item(modifiers=["no_onion"], position=2))
    result = conversation.handle("s1", "change second").results[0]
    assert len(result.data["order"]["items"]) == 1
    assert result.data["order"]["items"][0]["quantity"] == 3
    model.outputs["شيل التاني"] = output("remove_item", item(position=2))
    assert not conversation.handle("s1", "شيل التاني").results
    model.outputs["زود واحد"] = output("update_item", item(delta=1))
    conversation.handle("s1", "زود واحد")
    assert tools.get_order("s1")["data"]["items"][0]["quantity"] == 4


def test_change_invalidates_summary_end_to_end(service):
    conversation, model = service
    add(conversation, model)
    conversation.handle("s1", "خلاص")
    model.outputs["change"] = output("update_item", item(quantity=1))
    conversation.handle("s1", "change")
    assert conversation.handle("s1", "أيوه أكد").results[0].outcome == "summary_required"
    assert not tools.get_saved_order(1)["ok"]


def test_text_end_to_end_real_sqlite(service):
    conversation, model = service
    utterance = "عايز 2 تشيكن رانش وسط"
    model.outputs[utterance] = output("add_item", item("Chicken Ranch", 2, "medium", []))
    conversation.handle("s1", utterance)
    model.outputs["خليهم تلاتة ومن غير بصل"] = output(
        "update_item", item(quantity=3, modifiers=["no_onion"])
    )
    conversation.handle("s1", "خليهم تلاتة ومن غير بصل")
    summary = conversation.handle("s1", "خلاص")
    assert "450.00" in summary.text
    assert (
        summary.results[0].data["total_piasters"] == tools.get_order("s1")["data"]["total_piasters"]
    )
    assert not tools.get_saved_order(1)["ok"]
    reply = conversation.handle("s1", "أيوه أكد")
    saved = tools.get_saved_order(reply.results[0].data["order_id"])["data"]
    assert saved["status"] == "confirmed" and saved["total_piasters"] == 45000
    assert saved["items"][0]["modifiers"] == ["no_onion"]
    assert str(saved["order_id"]) in reply.text


def test_price_changed_end_to_end_needs_fresh_confirmation(service):
    conversation, model = service
    from order_db import seed

    add(conversation, model)
    conversation.handle("s1", "خلاص")
    seed.set_price("chicken_ranch_medium", 16000)
    changed = conversation.handle("s1", "أيوه أكد")
    assert changed.results[0].outcome == "price_changed" and "320.00" in changed.text
    assert not tools.get_saved_order(1)["ok"]
    state = conversation.application.orchestrator.sessions.get("s1")
    request_id = state.pending_request_id
    saved = conversation.handle("s1", "أيوه أكد")
    assert state.pending_request_id == request_id
    assert saved.completed and tools.get_saved_order(1)["data"]["total_piasters"] == 32000


def test_provider_failure_is_safe(service):
    conversation, model = service
    model.outputs["error"] = RuntimeError("do not echo provider internals")
    reply = conversation.handle("s1", "error")
    assert not reply.results and "internals" not in reply.text


def test_unknown_item_and_invalid_modifier_remain_database_decisions(service):
    conversation, model = service
    model.outputs["unknown"] = output("add_item", item("Nonexistent dish", 1, "medium", []))
    assert conversation.handle("s1", "unknown").results[0].outcome == "not_found"
    model.outputs["invalid"] = output(
        "add_item", item("Chicken Ranch", 1, "medium", ["extra_gold"])
    )
    assert conversation.handle("s1", "invalid").results[0].outcome == "invalid_modifier"
    assert tools.get_order("s1")["data"]["items"] == []


def test_real_create_order_not_called_before_confirmation(service, monkeypatch):
    conversation, model = service
    calls = []
    original = tools.create_order

    def spy(session_id, request_id):
        calls.append((session_id, request_id))
        return original(session_id, request_id)

    monkeypatch.setattr(tools, "create_order", spy)
    add(conversation, model)
    conversation.handle("s1", "خلاص")
    assert calls == []
    conversation.handle("s1", "أيوه أكد")
    assert len(calls) == 1
    assert tools.get_saved_order(1)["data"]["status"] == "confirmed"


def test_groq_adapter_uses_environment_and_isolated_sdk(monkeypatch):
    from types import SimpleNamespace

    from conversation import GroqUnderstandingProvider

    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=output("question")))]
        )

    def client(**kwargs):
        assert kwargs["api_key"] == "test-placeholder"
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    monkeypatch.setenv("GROQ_API_KEY", "test-placeholder")
    monkeypatch.setenv("LLM_MODEL", "configured-test-model")
    monkeypatch.setitem(sys.modules, "groq", SimpleNamespace(Groq=client))
    provider = GroqUnderstandingProvider()
    assert calls == []
    provider.complete([{"role": "user", "content": "test"}])
    assert calls[0]["model"] == "configured-test-model"
    assert calls[0]["response_format"] == {"type": "json_object"}
