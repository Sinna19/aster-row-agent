"""
Minimal CLI for the Aster & Row support agent.

Usage:
    python -m app.cli
    python -m app.cli --debug     # print retrieval/tool trace after each turn

Type 'exit' to quit, 'reset' to start a new session.
"""
from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv

load_dotenv()

from app.agent import Agent, new_session_id

KB_DIR = os.path.join(os.path.dirname(__file__), "..", "knowledge-base")
ORDERS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "orders.json")
LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "logs", "trace.jsonl")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="print the trace after each response")
    args = parser.parse_args()

    if not os.environ.get("GROQ_API_KEY"):
        print("WARNING: GROQ_API_KEY is not set. Set it in your environment or .env file.")

    agent = Agent(KB_DIR, ORDERS_PATH, log_path=LOG_PATH)
    session_id = new_session_id()
    print("Aster & Row support agent. Type 'exit' to quit, 'reset' for a new session.\n")

    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            break
        if user_input.lower() == "reset":
            session_id = new_session_id()
            print("(new session started)\n")
            continue

        result = agent.respond(session_id, user_input)
        print(f"\nagent> {result['response']}")
        if result["sources"]:
            print(f"       sources: {', '.join(sorted(set(result['sources'])))}")
        if result["handoff"]:
            print("       [recommending human handoff]")
        if args.debug:
            print("\n--- trace ---")
            print(result["trace"])
        print()


if __name__ == "__main__":
    main()
