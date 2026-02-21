"""
PyMOL AI Agent — core conversation loop.

Usage:
    python agent.py              # start in guided mode
    python agent.py --expert     # start in expert mode

Runtime commands (type at the prompt):
    expert / guided   switch mode mid-session
    quit / exit       end session
"""

import re
import sys
import argparse

import anthropic

from pymol_interface import get_session_state, execute_command, close_session

from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 2048
SYSTEM_PROMPT_FILE = "system_prompt.txt"

# ---------------------------------------------------------------------------
# Command parsing
# ---------------------------------------------------------------------------

_CMD_RE = re.compile(r"<pymol>(.*?)</pymol>", re.DOTALL)


def extract_commands(text: str) -> list[str]:
    """Pull every <pymol>...</pymol> block out of the LLM response."""
    return [m.strip() for m in _CMD_RE.findall(text) if m.strip()]


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def run_agent(start_mode: str = "guided") -> None:
    client = anthropic.Anthropic()
    conversation_history: list[dict] = []
    mode = start_mode
    pending_outputs: list[str] = []   # command outputs to feed back next turn

    with open(SYSTEM_PROMPT_FILE) as f:
        system_prompt = f.read()

    print(f"PyMOL Agent ready (mode: {mode}).")
    print("Type 'guided' or 'expert' to switch modes, 'quit' to exit.\n")

    while True:
        # --- get user input ---
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "bye"):
            print("Goodbye.")
            break

        if user_input.lower() in ("guided", "expert"):
            mode = user_input.lower()
            print(f"Switched to {mode} mode.\n")
            continue

        # --- build context to append to user message ---
        context_parts = [
            f"\n\nCurrent PyMOL session state:\n{get_session_state()}",
            f"Current mode: {mode}",
        ]
        if pending_outputs:
            context_parts.append(
                "Outputs from commands executed last turn:\n" + "\n".join(pending_outputs)
            )
            pending_outputs = []

        conversation_history.append({
            "role": "user",
            "content": user_input + "\n".join(context_parts),
        })

        # --- call the LLM ---
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=system_prompt,
                messages=conversation_history,
            )
        except anthropic.APIError as e:
            print(f"[ERROR] API call failed: {e}\n")
            conversation_history.pop()   # drop the failed turn
            continue

        reply = response.content[0].text
        conversation_history.append({"role": "assistant", "content": reply})

        print(f"\nAgent: {reply}\n")

        # --- execute commands ---
        commands = extract_commands(reply)
        for cmd in commands:
            print(f"[CMD] {cmd}")
            try:
                output = execute_command(cmd)
            except Exception as e:
                output = f"ERROR: {e}"
                print(f"      ! {output}")
                pending_outputs.append(f"  Command {cmd!r} failed: {e}")
                continue

            if output:
                print(f"      → {output}")
                pending_outputs.append(f"  {cmd!r} → {output}")

        if commands:
            print()

    close_session()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyMOL AI Agent")
    parser.add_argument(
        "--expert",
        action="store_true",
        help="Start in expert mode (minimal explanations)",
    )
    args = parser.parse_args()

    run_agent(start_mode="expert" if args.expert else "guided")
