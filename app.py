"""Order composition plus the single text/voice MVP runtime entry point."""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from os import PathLike
from typing import TYPE_CHECKING

from agent import Action, ConversationResult, OrderOrchestrator
from order_db import init_db, tools

if TYPE_CHECKING:
    from conversation import ConversationService


class RestaurantOrderApplication:
    """Reusable composition of the orchestration layer and real database tools."""

    def __init__(
        self,
        db_path: str | PathLike[str] | None = None,
        request_id_factory: Callable[[], str] | None = None,
    ) -> None:
        if db_path is not None:
            os.environ["DB_PATH"] = os.fspath(db_path)
        # ``init_db`` is idempotent and seeds the deterministic demo menu when empty.
        init_db()
        self.orchestrator = OrderOrchestrator(tools, request_id_factory=request_id_factory)

    def start_session(self, session_id: str | None = None) -> str:
        """Start or resume the stable session used for the order draft."""

        return self.orchestrator.start_session(session_id)

    def submit(self, session_id: str, action: Action) -> ConversationResult:
        """Submit one structured action from the future conversation layer."""

        return self.orchestrator.handle_action(session_id, action)

    def current_order(self, session_id: str):
        """Read the authoritative draft for conversation reference resolution."""
        return tools.get_order(session_id)

    def search_menu(self, query: str):
        """Ground a customer's menu question through the canonical tool."""
        return tools.search_menu(query)

    def item_details(self, item_id: str):
        """Read menu names/sizes for rendering, never from model memory."""
        return tools.get_item_details(item_id)


class VoiceRuntime:
    """Transport between speech/text functions and the conversation service."""

    def __init__(
        self,
        conversation: ConversationService,
        listen: Callable[[], str],
        speak: Callable[[str], None],
    ):
        self.conversation = conversation
        self.listen = listen
        self.speak = speak

    def run(self, session_id: str, max_empty: int = 3) -> int | None:
        self.speak("أهلا بيك، تحب تطلب إيه؟")
        empty = 0
        while empty < max_empty:
            message = self.listen()
            if message.strip().casefold() in {"exit", "quit", "خروج", "إنهاء", "انهاء"}:
                return None
            if not message.strip():
                empty += 1
                self.speak("مسمعتش حاجة، ممكن تعيد؟")
                continue
            empty = 0
            reply = self.conversation.handle(session_id, message)
            self.speak(reply.text)
            if reply.completed:
                return reply.results[-1].data["order_id"]
        return None


def main() -> None:
    """One entry point for the live text/voice MVP and optional audio smoke tests."""
    from conversation import ConversationService, configured_provider

    parser = argparse.ArgumentParser(description="Restaurant ordering MVP")
    parser.add_argument("--mode", choices=["text", "voice"], default="text")
    parser.add_argument("--db-path", default=None, help="Defaults to DB_PATH or restaurant.db")
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--voice-smoke", choices=["stt", "tts"], default=None)
    args = parser.parse_args()
    # The existing DB loader reads .env before providers/VoiceConfig are imported.
    if args.voice_smoke:
        from voice import listen, speak

        if args.voice_smoke == "stt":
            print(listen())
        else:
            speak("أهلا بيك، تحب تطلب إيه؟")
        return
    application = RestaurantOrderApplication(args.db_path)
    conversation = ConversationService(application, configured_provider())
    session_id = application.start_session(args.session_id)
    if args.mode == "voice":
        from voice import listen, speak

        runtime = VoiceRuntime(conversation, listen, speak)
    else:
        runtime = VoiceRuntime(conversation, lambda: input("Customer> "), print)
    try:
        runtime.run(session_id)
    except (KeyboardInterrupt, EOFError):
        print("Session ended.")


if __name__ == "__main__":
    main()
