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

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from app.agent import Agent

ROOT = os.path.dirname(os.path.abspath(__file__))
KB_DIR = os.path.join(ROOT, "knowledge-base")
ORDERS_PATH = os.path.join(ROOT, "data", "orders.json")


def _load_streamlit_secrets() -> None:
    """Allow local .env development and Streamlit Cloud's secrets panel."""
    for key in ("GROQ_API_KEY", "ASTER_ROW_MODEL"):
        if not os.environ.get(key) and key in st.secrets:
            os.environ[key] = str(st.secrets[key])


@st.cache_resource(show_spinner="Loading the hybrid retrieval model…")
def get_agent() -> Agent:
    # Community Cloud's filesystem is ephemeral, so trace logging is disabled.
    return Agent(KB_DIR, ORDERS_PATH, log_path=None)


def main() -> None:
    st.set_page_config(page_title="Aster & Row Support", page_icon="🎒")
    _load_streamlit_secrets()

    st.title("🎒 Aster & Row Support")
    st.caption("Policy-grounded answers, hybrid retrieval, and privacy-safe order lookup.")

    if not os.environ.get("GROQ_API_KEY"):
        st.error("This app needs a GROQ_API_KEY. Add it in the app's Streamlit Cloud secrets.")
        st.code('GROQ_API_KEY = "gsk_..."', language="toml")
        st.stop()

    if "agent_session_id" not in st.session_state:
        st.session_state.agent_session_id = str(uuid.uuid4())
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    with st.sidebar:
        st.subheader("Try a question")
        st.caption("Each browser session has its own conversation history.")
        if st.button("Start a new conversation", use_container_width=True):
            st.session_state.agent_session_id = str(uuid.uuid4())
            st.session_state.chat_messages = []
            st.rerun()

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask about returns, shipping, products, or an order…")
    if not prompt:
        return

    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Checking Aster & Row support information…"):
            result = get_agent().respond(st.session_state.agent_session_id, prompt)
        response = result["response"]
        st.markdown(response)
        if result["sources"]:
            st.caption("Sources: " + ", ".join(sorted(set(result["sources"]))))
        if result["handoff"]:
            st.warning("A human support specialist is recommended for this request.")

    st.session_state.chat_messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
