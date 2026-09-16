"""
Gradio entrypoint for deploying the Aster & Row support agent as a
Hugging Face Space.

This is a thin UI wrapper -- it does not change any agent/retrieval/tool
logic in app/. It just:
  1. Builds one Agent instance at process startup (KB indexing happens once).
  2. Gives each browser session its own session_id via gr.State, so
     Agent.sessions (per-session history) works the same as it does in cli.py.
  3. Renders sources/handoff info inline in the chat reply, mirroring what
     app/cli.py prints to the terminal.
"""
from __future__ import annotations

import os
import uuid

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

from app.agent import Agent

ROOT = os.path.dirname(os.path.abspath(__file__))
KB_DIR = os.path.join(ROOT, "knowledge-base")
ORDERS_PATH = os.path.join(ROOT, "data", "orders.json")
LOG_PATH = os.path.join(ROOT, "logs", "trace.jsonl")

# Built once at startup (TF-IDF indexing over 14 short docs is well under a
# second -- see README's storage/architecture notes). Shared across all
# visitors; per-visitor state lives only in Agent.sessions, keyed by session_id.
agent = Agent(KB_DIR, ORDERS_PATH, log_path=LOG_PATH)

if not os.environ.get("GROQ_API_KEY"):
    print("WARNING: GROQ_API_KEY is not set. Set it as a Space secret.")


def _new_session_id() -> str:
    return str(uuid.uuid4())


def chat_fn(message: str, history: list, session_id: str):
    result = agent.respond(session_id, message)
    reply = result["response"]

    extras = []
    if result["sources"]:
        extras.append(f"*Sources: {', '.join(sorted(set(result['sources'])))}*")
    if result["handoff"]:
        extras.append("*[Recommending human handoff]*")
    if extras:
        reply = reply + "\n\n" + "\n".join(extras)

    return reply


demo = gr.ChatInterface(
    fn=chat_fn,
    additional_inputs=[gr.State(_new_session_id)],
    title="Aster & Row Support Agent",
    description=(
        "A reliability-focused RAG + tool-calling customer support agent. "
        "Answers are grounded only in retrieved policy documents (with citations), "
        "order status comes from a real deterministic tool (never invented), "
        "and the agent explicitly flags conflicting or non-authoritative sources "
        "rather than guessing. See the README tab / repo for the full architecture "
        "and a 9-entry bug diary from building this."
    ),
    examples=[
        ["How long can I return a backpack?"],
        ["Where is ORD-1007 and when will it arrive?"],
        ["Do you ship internationally?"],
        ["Should I put the entire Breeze Tumbler in the dishwasher?"],
    ],
)

if __name__ == "__main__":
    # Render (and most non-HF hosts) assign a port via $PORT and expect the
    # app to bind on 0.0.0.0, not Gradio's default localhost:7860.
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
