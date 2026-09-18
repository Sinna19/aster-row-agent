"""
These tests stub out the OpenAI-compatible client (used to call Groq) so the
tool-use plumbing (does the agent actually call order_lookup, feed back a
sanitized result, and never leak raw internal fields into the messages sent
to the model) can be verified without a live API key or network access.
"""
import json
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.agent import Agent, new_session_id

KB_DIR = os.path.join(os.path.dirname(__file__), "..", "knowledge-base")
ORDERS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "orders.json")


class FakeEmbeddingModel:
    def encode(self, texts, **kwargs):
        return np.zeros((len(texts), 4), dtype=np.float32)


def _tool_call(call_id, name, arguments: dict):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


class FakeCompletions:
    """Simulates: first call -> tool_call for order_lookup; second call -> final text."""

    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            msg = SimpleNamespace(
                content=None,
                tool_calls=[_tool_call("call_1", "order_lookup", {"order_id": "ORD-1007"})],
            )
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)])
        else:
            # Ensure the sanitized tool result (not raw order data) was fed back
            last_msg = kwargs["messages"][-1]
            assert last_msg["role"] == "tool"
            assert "ava.morgan" not in last_msg["content"]  # no PII leaked into model context
            msg = SimpleNamespace(
                content="Your order ORD-1007 shipped with UPS, arriving August 22, 2026. (Source: order lookup)",
                tool_calls=None,
            )
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


class FakeChat:
    def __init__(self):
        self.completions = FakeCompletions()


class FakeGroqClient:
    def __init__(self, *a, **kw):
        self.chat = FakeChat()


def test_agent_calls_order_lookup_and_never_leaks_pii_to_model(monkeypatch):
    import app.agent as agent_mod

    monkeypatch.setattr(agent_mod, "OpenAI", FakeGroqClient)
    agent = Agent(KB_DIR, ORDERS_PATH, log_path=None, embedding_model=FakeEmbeddingModel())
    session_id = new_session_id()
    result = agent.respond(session_id, "Where is ORD-1007 and when should it arrive?")

    assert "UPS" in result["response"]
    assert result["trace"]["tool_calls"][0]["tool"] == "order_lookup"
    assert result["trace"]["tool_calls"][0]["arguments"]["order_id"] == "ORD-1007"
    # sanitized result recorded in the trace must not contain PII either
    tool_result = result["trace"]["tool_calls"][0]["result"]
    assert "email" not in tool_result
    assert "customer" not in tool_result
