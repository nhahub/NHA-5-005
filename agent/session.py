"""Small in-process store for orchestration state keyed by stable session ID."""

from __future__ import annotations

from uuid import uuid4

from .models import SessionState


class SessionStore:
    """Keeps confirmation state separate from the database-owned Order Draft.

    Production hosts can replace this with a durable implementation while keeping
    the same ``get`` and ``start`` contract.
    """

    def __init__(self) -> None:
        self._states: dict[str, SessionState] = {}

    def start(self, session_id: str | None = None) -> str:
        stable_id = session_id or str(uuid4())
        if not stable_id.strip():
            raise ValueError("session_id must not be blank")
        self._states.setdefault(stable_id, SessionState())
        return stable_id

    def get(self, session_id: str) -> SessionState:
        if not session_id or not session_id.strip():
            raise ValueError("session_id must not be blank")
        self._states.setdefault(session_id, SessionState())
        return self._states[session_id]
