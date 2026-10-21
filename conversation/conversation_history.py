"""Chat history per session; database drafts remain authoritative order state."""


class ConversationHistory:
    def __init__(self):
        self.messages = []

    def add_user_message(self, message):
        self.messages.append({"role": "user", "content": message})

    def add_assistant_message(self, message):
        self.messages.append({"role": "assistant", "content": message})

    def get_messages(self):
        return [dict(message) for message in self.messages]


class SessionConversationHistory:
    """Independent histories keyed by the application's stable session ID."""

    def __init__(self):
        self._histories = {}

    def for_session(self, session_id):
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must not be blank")
        return self._histories.setdefault(session_id, ConversationHistory())
