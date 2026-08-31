"""
In-memory session store for multi-turn conversation state.

Each session keeps the raw message history (what actually gets replayed to
the model as conversation turns -- this is what gives us "what about
Canada?" follow-up behavior for free, since the model sees the prior
question). We cap history length so an old session can't grow unbounded or
bleed into unrelated topics indefinitely, per the assignment's requirement
that the agent "not carry unrelated details indefinitely."
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

MAX_TURNS_KEPT = 12  # user+assistant messages kept in context (6 exchanges)


@dataclass
class Session:
    session_id: str
    messages: list[dict] = field(default_factory=list)  # [{role, content}]
    last_order_id: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def add(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        if len(self.messages) > MAX_TURNS_KEPT:
            self.messages = self.messages[-MAX_TURNS_KEPT:]
        self.updated_at = time.time()


class SessionStore:
    def __init__(self):
        self._sessions: dict[str, Session] = {}

    def get_or_create(self, session_id: str) -> Session:
        if session_id not in self._sessions:
            self._sessions[session_id] = Session(session_id=session_id)
        return self._sessions[session_id]

    def reset(self, session_id: str):
        self._sessions.pop(session_id, None)
