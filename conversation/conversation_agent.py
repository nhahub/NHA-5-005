"""Session-scoped understanding, validated Action bridge and grounded replies."""

from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlparse

from agent import Action, ConversationResult
from app import RestaurantOrderApplication

from .conversation_history import SessionConversationHistory
from .prompts import SYSTEM_PROMPT


class UnderstandingProvider(Protocol):
    """Small provider boundary; deterministic tests supply scripted outputs."""

    def complete(self, messages: list[dict[str, str]]) -> str: ...


class GroqUnderstandingProvider:
    """Optional live runtime adapter; SDK/client initialization is deferred."""

    def __init__(self) -> None:
        self._client: Any = None

    def complete(self, messages: list[dict[str, str]]) -> str:
        if self._client is None:
            from groq import Groq

            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                raise RuntimeError("GROQ_API_KEY is required for live understanding")
            self._client = Groq(api_key=api_key, timeout=30, max_retries=1)
        response = self._client.chat.completions.create(
            model=os.environ.get("LLM_MODEL", "openai/gpt-oss-20b"),
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0,
        )
        return response.choices[0].message.content or ""


class LocalUnderstandingProvider:
    """Loopback-only llama.cpp/Ollama HTTP boundary; never loads model weights."""

    def __init__(self, provider: str = "llamacpp") -> None:
        if provider not in {"llamacpp", "ollama"}:
            raise ValueError("Unsupported local LLM provider")
        self.provider = provider
        variable = "LLAMA_CPP_BASE_URL" if provider == "llamacpp" else "OLLAMA_BASE_URL"
        default = "http://127.0.0.1:8080" if provider == "llamacpp" else "http://127.0.0.1:11434"
        self.base_url = os.getenv(variable, default).rstrip("/")
        parsed = urlparse(self.base_url)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Local LLM endpoint must be an HTTP loopback URL")

    def complete(self, messages: list[dict[str, str]]) -> str:
        # Qwen's chat template accepts one initial system message, not several.
        # Preserve all instructions and grounded draft context without changing callers.
        messages = [dict(message) for message in messages]
        while len(messages) > 1 and messages[0]["role"] == messages[1]["role"] == "system":
            messages[0]["content"] += "\n\n" + messages.pop(1)["content"]
        if self.provider == "llamacpp":
            endpoint = "/v1/chat/completions"
            payload = {
                "model": os.getenv("LLM_MODEL", "qwen35-4b-local"),
                "messages": messages,
                "response_format": {"type": "json_object"},
                "temperature": 0,
                "max_tokens": 768,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        else:
            endpoint = "/api/chat"
            payload = {
                "model": os.getenv("OLLAMA_MODEL", "qwen35-4b-local:latest"),
                "messages": messages,
                "format": "json",
                "think": False,
                "stream": False,
                "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 768},
            }
        request = urllib.request.Request(
            self.base_url + endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        # Do not send localhost traffic or conversation text through environment proxies.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=60) as response:
            data = json.load(response)
        content = (
            data["choices"][0]["message"]["content"]
            if self.provider == "llamacpp"
            else data["message"]["content"]
        )
        if not isinstance(content, str):
            raise TypeError("Local model returned non-text content")
        return content


def configured_provider() -> UnderstandingProvider:
    """Same validated Conversation pipeline for every provider; local by default."""
    provider = os.getenv("LLM_PROVIDER", "llamacpp").lower()
    if provider == "groq":
        return GroqUnderstandingProvider()
    return LocalUnderstandingProvider(provider)


class UnderstandingError(ValueError):
    """Untrusted model output cannot safely become an Action."""


class CustomerClarification(UnderstandingError):
    """A known reference/validation problem with a grounded customer-facing message."""


INTENTS = {
    "add_item",
    "update_item",
    "remove_item",
    "show_summary",
    "confirm_order",
    "question",
    "clarification",
}
ITEM_FIELDS = {"item_name", "quantity", "size", "modifiers", "position", "quantity_delta"}
MODIFIER_LABELS = {"no_onion": "من غير بصل"}
DONE = {"خلاص", "كده خلاص", "بس كده", "that's all", "done", "summary"}
CONFIRM = {
    "أيوه أكد",
    "ايوه اكد",
    "أيوة أكد",
    "ايوة اكد",
    "أكد",
    "اكد",
    "confirm",
    "yes confirm",
    "أيوه",
    "ايوه",
    "أيوة",
    "ايوة",
    "yes",
    "نعم",
}


def validate_understanding(raw: str) -> dict[str, Any]:
    """Strict structural validation before any order mutation."""

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise UnderstandingError("duplicate field")
            result[key] = value
        return result

    try:
        data = json.loads(raw, object_pairs_hook=unique_object)
    except (ValueError, TypeError) as exc:
        raise UnderstandingError("malformed JSON") from exc
    if not isinstance(data, dict) or set(data) != {"intent", "items", "missing_information"}:
        raise UnderstandingError("invalid top-level fields")
    intent = data["intent"]
    if not isinstance(intent, str) or intent not in INTENTS:
        raise UnderstandingError("unknown intent")
    if not isinstance(data["missing_information"], list) or not all(
        isinstance(value, str) for value in data["missing_information"]
    ):
        raise UnderstandingError("invalid missing information")
    if not isinstance(data["items"], list) or len(data["items"]) > 20:
        raise UnderstandingError("invalid items")
    if intent in {"show_summary", "confirm_order", "clarification"} and data["items"]:
        raise UnderstandingError("unexpected items")
    if intent in {"add_item", "update_item", "remove_item"} and not data["items"]:
        raise UnderstandingError("missing items")
    for item in data["items"]:
        if not isinstance(item, dict) or set(item) not in (
            ITEM_FIELDS,
            ITEM_FIELDS | {"affected_quantity"},
        ):
            raise UnderstandingError("invalid item fields")
        affected = item.get("affected_quantity")
        if affected is not None and (
            type(affected) is not int
            or affected < 1
            or intent != "update_item"
            or item["quantity"] is not None
            or item["quantity_delta"] is not None
            or item["modifiers"] is None
        ):
            raise UnderstandingError("invalid affected quantity")
        for field in ("item_name", "size"):
            if item[field] is not None and (
                not isinstance(item[field], str) or not item[field].strip()
            ):
                raise UnderstandingError("invalid text")
        for field in ("quantity", "position"):
            if item[field] is not None and (type(item[field]) is not int or item[field] < 1):
                raise UnderstandingError("invalid positive integer")
        delta = item["quantity_delta"]
        if delta is not None and (type(delta) is not int or delta == 0 or intent != "update_item"):
            raise UnderstandingError("invalid quantity delta")
        if delta is not None and item["quantity"] is not None:
            raise UnderstandingError("conflicting quantities")
        if intent == "add_item" and item["position"] is not None:
            raise UnderstandingError("unexpected add reference")
        if intent == "remove_item" and item["modifiers"] is not None:
            raise UnderstandingError("cannot remove and change modifiers together")
        mods = item["modifiers"]
        if mods is not None and (
            not isinstance(mods, list)
            or not all(isinstance(mod, str) and mod.strip() for mod in mods)
        ):
            raise UnderstandingError("invalid modifiers")
    return data


@dataclass(frozen=True)
class ConversationReply:
    text: str
    results: tuple[ConversationResult, ...] = ()

    @property
    def completed(self) -> bool:
        return any(result.outcome in {"order_saved", "already_saved"} for result in self.results)


class ConversationService:
    def __init__(self, application: RestaurantOrderApplication, provider: UnderstandingProvider):
        self.application = application
        self.provider = provider
        self.history = SessionConversationHistory()

    def handle(self, session_id: str, message: str) -> ConversationReply:
        """Understand one utterance, resolve against tools and render their results."""
        if not isinstance(message, str) or not message.strip():
            return ConversationReply("مسمعتش الطلب، ممكن تقول تاني؟")
        history = self.history.for_session(session_id)
        history.add_user_message(message)
        try:
            draft = self.application.current_order(session_id)
            if not draft["ok"]:
                return self._remember(history, ConversationReply("مش قادر أجيب الطلب، حاول تاني."))
            control = message.strip().casefold().rstrip(".!؟?")
            if control in DONE or control in CONFIRM:
                intent = "show_summary" if control in DONE else "confirm_order"
                understood = {"intent": intent, "items": [], "missing_information": []}
            else:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "system",
                        "content": "CURRENT DATABASE DRAFT: " + self._describe_draft(draft["data"]),
                    },
                ]
                understood = validate_understanding(
                    self.provider.complete(messages + history.get_messages())
                )
                # A model classification alone is not evidence of customer affirmation.
                if understood["intent"] == "confirm_order":
                    return self._remember(history, ConversationReply("للتأكيد قول: أيوه أكد."))
                # Explicit short quantity wording is stronger evidence than a model's
                # omitted scope field. This extracts language only, never a menu fact.
                contextual = self._contextual_quantity_request(message)
                if contextual is not None:
                    understood = contextual
            if understood["missing_information"] or understood["intent"] == "clarification":
                return self._remember(
                    history, ConversationReply("ممكن توضح الصنف والكمية والتعديل؟")
                )
            if understood["intent"] == "question":
                reply = self._menu_question(understood)
            else:
                actions = self._actions(understood, draft["data"])
                reply = self._execute(session_id, actions, draft["data"])
            return self._remember(history, reply)
        except CustomerClarification as exc:
            return self._remember(history, ConversationReply(str(exc)))
        except UnderstandingError:
            return self._remember(
                history, ConversationReply("مش واضح قصدك، حدد الصنف والكمية والتعديل.")
            )
        except Exception:  # noqa: BLE001 - contain provider/tool failures without logging secrets
            return self._remember(history, ConversationReply("حصلت مشكلة في الخدمة، حاول تاني."))

    def _actions(self, data: dict[str, Any], draft: dict[str, Any]) -> list[Action]:
        intent = data["intent"]
        if intent in {"show_summary", "confirm_order"}:
            return [Action(intent)]
        actions = []
        used_lines: set[str] = set()
        for item in data["items"]:
            if intent == "add_item":
                if item["item_name"] is None or item["quantity"] is None:
                    raise UnderstandingError("missing add information")
                actions.append(
                    Action(
                        "add_item",
                        query=self._query(item),
                        quantity=item["quantity"],
                        modifiers=item["modifiers"] or [],
                    )
                )
                continue
            line = self._resolve_line(item, draft)
            if line["line_id"] in used_lines:
                raise UnderstandingError("duplicate line references")
            used_lines.add(line["line_id"])
            if item["size"] is not None:
                details = self.application.item_details(line["item_id"])
                if not details["ok"] or details["data"]["size"] != item["size"]:
                    # Canonical update_item cannot replace item_id/size; never ignore it.
                    raise UnderstandingError("size replacement needs remove/add clarification")
            if intent == "remove_item":
                quantity = item["quantity"]
                if quantity is None or quantity == line["quantity"]:
                    actions.append(Action("remove_item", line_id=line["line_id"]))
                elif quantity < line["quantity"]:
                    actions.append(
                        Action(
                            "update_item",
                            line_id=line["line_id"],
                            quantity=line["quantity"] - quantity,
                        )
                    )
                else:
                    raise UnderstandingError("removal exceeds line quantity")
            else:
                affected = item.get("affected_quantity")
                if affected is not None:
                    if affected > line["quantity"]:
                        raise CustomerClarification("الكمية المطلوبة أكبر من اللي في الطلب.")
                    details = self.application.item_details(line["item_id"])
                    if not details["ok"]:
                        raise CustomerClarification("مش قادر أجيب بيانات الصنف، حاول تاني.")
                    if not details["data"]["available"]:
                        raise CustomerClarification("الصنف ده مش متاح دلوقتي.")
                    if set(item["modifiers"]) - set(details["data"]["allowed_modifiers"]):
                        raise CustomerClarification("التعديل ده مش مسموح للصنف.")
                    if affected < line["quantity"] and set(item["modifiers"]) != set(
                        line["modifiers"]
                    ):
                        actions.extend(
                            [
                                Action(
                                    "update_item",
                                    line_id=line["line_id"],
                                    quantity=line["quantity"] - affected,
                                ),
                                Action(
                                    "add_item",
                                    item_id=line["item_id"],
                                    quantity=affected,
                                    modifiers=item["modifiers"],
                                ),
                            ]
                        )
                        continue
                    actions.append(
                        Action("update_item", line_id=line["line_id"], modifiers=item["modifiers"])
                    )
                    continue
                quantity = item["quantity"]
                if item["quantity_delta"] is not None:
                    quantity = line["quantity"] + item["quantity_delta"]
                    if quantity == 0:
                        actions.append(Action("remove_item", line_id=line["line_id"]))
                        continue
                    if quantity < 0:
                        raise UnderstandingError("invalid resulting quantity")
                if quantity is None and item["modifiers"] is None:
                    raise UnderstandingError("empty update")
                actions.append(
                    Action(
                        "update_item",
                        line_id=line["line_id"],
                        quantity=quantity,
                        modifiers=item["modifiers"],
                    )
                )
        return actions

    @staticmethod
    def _contextual_quantity_request(message: str) -> dict[str, Any] | None:
        text = re.sub(r"[\u064b-\u065f\u0670\u0640]", "", message).strip().rstrip(".!؟?")
        text = " ".join(text.translate(str.maketrans({"ة": "ه", "أ": "ا", "إ": "ا"})).split())
        numbers = {"واحد": 1, "واحده": 1, "اتنين": 2, "اثنين": 2, "تلاته": 3, "ثلاثه": 3}
        number_pattern = r"(واحده|واحد|اتنين|اثنين|تلاته|ثلاثه|\d+)"
        change = re.fullmatch(r"(شيل|زود|نقص)\s+" + number_pattern + r"(?:\s+منهم)?", text)
        modifier = re.fullmatch(
            r"(?:خلي\s+)?" + number_pattern + r"(?:\s+منهم)?\s+(من غير بصل|من غير صوص|بجبنه)",
            text,
        )
        if not change and not modifier:
            return None
        spoken_number = change[2] if change else modifier[1]
        quantity = numbers.get(spoken_number)
        if quantity is None:
            quantity = int(spoken_number)
        if quantity < 1:
            raise UnderstandingError("invalid spoken quantity")
        item = {
            "item_name": None,
            "quantity": None,
            "size": None,
            "modifiers": None,
            "position": None,
            "quantity_delta": None,
            "affected_quantity": None,
        }
        intent = "update_item"
        if modifier:
            item["affected_quantity"] = quantity
            item["modifiers"] = [
                {"من غير بصل": "no_onion", "من غير صوص": "no_sauce", "بجبنه": "extra_cheese"}[
                    modifier[2]
                ]
            ]
        elif change[1] == "شيل":
            intent = "remove_item"
            item["quantity"] = quantity
        else:
            item["quantity_delta"] = quantity if change[1] == "زود" else -quantity
        return {"intent": intent, "items": [item], "missing_information": []}

    def _execute(self, session_id: str, actions: list[Action], draft: dict[str, Any]):
        """Execute semantic decomposition through the existing application only.

        Splits are two canonical operations, not an atomic DB transaction. If the
        add fails and the draft is still exactly the returned reduced draft, restore
        the original quantity. Never blindly retry an uncertain committed write.
        """
        results = []
        latest = draft
        for index, action in enumerate(actions):
            if action.line_id and not any(
                row["line_id"] == action.line_id for row in latest["items"]
            ):
                return ConversationReply(
                    "سطر الطلب اتغير، راجع الملخص وحدد التعديل تاني.", tuple(results)
                )
            result = self.application.submit(session_id, action)
            results.append(result)
            if result.requires_customer_input:
                if action.name == "add_item" and action.item_id and index:
                    previous = actions[index - 1]
                    original = next(
                        (row for row in draft["items"] if row["line_id"] == previous.line_id), None
                    )
                    current = self.application.current_order(session_id)
                    if original and current["ok"] and current["data"]["items"] == latest["items"]:
                        restore = self.application.submit(
                            session_id,
                            Action(
                                "update_item",
                                line_id=previous.line_id,
                                quantity=original["quantity"],
                                modifiers=original["modifiers"],
                            ),
                        )
                        results.append(restore)
                        if restore.outcome == "item_updated":
                            return ConversationReply(
                                "التعديل ما تمش؛ رجعت الكمية الأصلية. حاول تاني.", tuple(results)
                            )
                    return ConversationReply(
                        "التعديل ما اكتملش. راجع ملخص الطلب قبل المتابعة.", tuple(results)
                    )
                break
            latest = result.data.get("order", latest)
        if (
            len(actions) == 2
            and actions[1].name == "add_item"
            and actions[1].item_id
            and all(not result.requires_customer_input for result in results)
        ):
            return ConversationReply(
                "تمام، عدلت الكمية المطلوبة، والباقي زي ما هو.", tuple(results)
            )
        return ConversationReply("\n".join(self._render(r) for r in results), tuple(results))

    def _resolve_line(self, item: dict[str, Any], draft: dict[str, Any]) -> dict[str, Any]:
        lines = draft["items"]
        position = item["position"]
        if position is not None:
            if position > len(lines):
                raise UnderstandingError("impossible position")
            lines = [lines[position - 1]]
        if item["item_name"]:
            found = self.application.search_menu(self._query(item))
            if not found["ok"]:
                raise UnderstandingError("menu lookup failed")
            ids = {entry["item_id"] for entry in found["data"]}
            lines = [line for line in lines if line["item_id"] in ids]
        if len(lines) != 1:
            if len(lines) > 1:
                raise CustomerClarification(self._reference_question(lines))
            raise CustomerClarification("مش لاقي الصنف المقصود في الطلب الحالي.")
        return lines[0]

    def _reference_question(self, lines):
        options = []
        for line in lines:
            details = self.application.item_details(line["item_id"])
            if not details["ok"]:
                raise UnderstandingError("cannot describe candidates")
            item = details["data"]
            mods = [MODIFIER_LABELS.get(mod, mod) for mod in line["modifiers"]]
            variant = "اللي " + " و".join(mods) if mods else "العادية"
            options.append(f"{item['name']} {item['size'] or ''} {variant}")
        return "تقصد " + " ولا ".join(options) + "؟"

    @staticmethod
    def _query(item: dict[str, Any]) -> str:
        return " ".join(value for value in (item["item_name"], item["size"]) if value)

    def _menu_question(self, data: dict[str, Any]) -> ConversationReply:
        queries = [self._query(item) for item in data["items"]] or [""]
        answers = []
        for query in queries:
            result = self.application.search_menu(query)
            if not result["ok"]:
                return ConversationReply("مش قادر أجيب المنيو دلوقتي، حاول تاني.")
            for item in result["data"]:
                state = "متاح" if item["available"] else "غير متاح"
                answers.append(
                    f"{item['name']} {item['size'] or ''}: "
                    f"{self._money(item['price_piasters'])} جنيه، {state}"
                )
        return ConversationReply("\n".join(answers) or "مش لاقي الصنف ده في المنيو.")

    def _describe_draft(self, order: dict[str, Any]) -> str:
        rows = []
        for index, line in enumerate(order["items"], 1):
            details = self.application.item_details(line["item_id"])
            if not details["ok"]:
                raise UnderstandingError("cannot describe current line")
            rows.append(
                {
                    "position": index,
                    "name": details["data"]["name"],
                    "size": details["data"]["size"],
                    "quantity": line["quantity"],
                    "modifiers": line["modifiers"],
                }
            )
        return json.dumps(rows, ensure_ascii=False)

    def _render(self, result: ConversationResult) -> str:
        outcome = result.outcome
        if outcome in {"summary_ready", "price_changed"}:
            order = result.data["order"]
            parts = []
            for line in order["items"]:
                details = self.application.item_details(line["item_id"])
                # Even if a name lookup fails, show the full tool-returned draft
                # and total; a summary must not become confirmable without display.
                item = (
                    details["data"]
                    if details["ok"]
                    else {
                        "name": line["item_id"],
                        "size": None,
                    }
                )
                parts.append(
                    f"{line['quantity']} {item['name']} {item['size'] or ''} "
                    + (
                        "، ".join(MODIFIER_LABELS.get(mod, mod) for mod in line["modifiers"])
                        if line["modifiers"]
                        else "العادية"
                    )
                )
            prefix = "السعر اتغير. " if outcome == "price_changed" else ""
            return (
                prefix
                + "طلبك: "
                + "؛ ".join(parts)
                + ". الإجمالي "
                + self._money(order["total_piasters"])
                + " جنيه. تأكد الطلب؟"
            )
        if outcome in {"order_saved", "already_saved"}:
            return f"طلبك اتأكد، رقم الطلب {result.data['order_id']}."
        responses = {
            "item_added": "تمام، ضفت الصنف للطلب.",
            "item_updated": "تمام، عدلت الطلب.",
            "item_removed": "تمام، شلت الصنف من الطلب.",
            "unavailable": "الصنف ده مش متاح دلوقتي.",
            "not_found": "مش لاقي الصنف أو السطر ده.",
            "invalid_modifier": "التعديل ده مش مسموح للصنف.",
            "invalid_quantity": "محتاج كمية صحيحة.",
            "empty_order": "الطلب فاضي، تحب تطلب إيه؟",
            "summary_required": "محتاج أعرض ملخص الطلب الأول. قول خلاص لعرضه.",
            "save_failed": "الحفظ ما تمش، حاول تأكد تاني.",
            "tool_failure": "حصلت مشكلة في الخدمة، حاول تاني.",
        }
        if outcome == "choose_item":
            options = [f"{entry['name']} {entry['size'] or ''}" for entry in result.data["matches"]]
            return "تقصد أنهي صنف؟ " + "؛ ".join(options)
        return responses.get(outcome, "ممكن توضح طلبك؟")

    @staticmethod
    def _money(piasters: int) -> str:
        # Formatting only: no totals or item prices are computed here.
        return f"{piasters // 100}.{piasters % 100:02d}"

    @staticmethod
    def _remember(history: Any, reply: ConversationReply) -> ConversationReply:
        history.add_assistant_message(reply.text)
        return reply
