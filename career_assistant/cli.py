"""Command-line interface. Asks for the employee's full name before anything else."""
from __future__ import annotations

import sys

from .agent import Agent, sign_in
from .llm import AnthropicClient, LLMError
from .store import DataStore

MAX_NAME_ATTEMPTS = 3
EXIT_WORDS = {"exit", "quit", "bye", "q"}


def main(argv=None) -> int:
    try:
        client = AnthropicClient()
    except LLMError as exc:
        print(f"Setup problem: {exc}", file=sys.stderr)
        return 2
    store = DataStore()

    print("=" * 62)
    print(" Career Skills Assistant")
    print("=" * 62)
    try:
        session = None
        for _ in range(MAX_NAME_ATTEMPTS):
            session = sign_in(store, input("Before we start, please enter your full name: "))
            if session:
                break
            print("Sorry, I couldn't find an employee with that full name. Please try again.")
        if not session:
            print("Too many unsuccessful attempts. Goodbye.")
            return 1

        agent = Agent(store, client, session)
        print(f"\nHello, {session.name}! Ask me about your career skills, e.g.\n"
              f"  \"What skills should I develop for my next career opportunity?\"\n"
              f"(type 'exit' to quit)")
        while True:
            text = input("\nYou: ")
            if text.strip().lower() in EXIT_WORDS:
                print("\nAssistant: Goodbye, and good luck with your career journey!")
                return 0
            try:
                print("\nAssistant: " + agent.handle(text))
            except LLMError as exc:
                print(f"\nAssistant: Sorry, I couldn't reach the model ({exc}). Please try again.")
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
