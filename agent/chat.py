"""Chat with the Oslo load agent in the terminal.

The agent loop:
  1. Send the question, the instructions and the tool descriptions to the LLM.
  2. If the LLM asks for a tool, run the Python function and send the result back.
  3. Repeat until the LLM answers in text.

Run from the repo root:  python agent/chat.py
Needs in .env: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from openai import OpenAI

from tools import TOOL_SCHEMAS, TOOLS

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
INSTRUCTIONS = (Path(__file__).parent / "instructions.md").read_text(encoding="utf-8")
MAX_TOOL_ROUNDS = 5  # stop runaway loops


def run_tool(call):
    """Run one tool call from the LLM and return the result as a JSON string."""
    args = json.loads(call.arguments or "{}")
    print(f"  [tool] {call.name}({', '.join(f'{k}={v!r}' for k, v in args.items())})")
    func = TOOLS.get(call.name)
    if func is None:
        result = {"error": f"Unknown tool {call.name}"}
    else:
        try:
            result = func(**args)
        except Exception as exc:  # the LLM should hear about failures, not crash the chat
            result = {"error": f"{type(exc).__name__}: {exc}"}
    if "error" in result:
        print(f"  [tool] error: {result['error']}")
    return json.dumps(result, ensure_ascii=False)


def instructions():
    """The LLM has no clock, so we tell it today's date on every turn."""
    now = datetime.now(ZoneInfo("Europe/Oslo"))
    return f"{INSTRUCTIONS}\nCurrent date and time in Oslo: {now:%A %Y-%m-%d %H:%M}."


def ask(client, question, previous_id):
    """One user turn: may involve several tool rounds."""
    extra = {"previous_response_id": previous_id} if previous_id else {}  # Azure rejects None
    response = client.responses.create(
        model=DEPLOYMENT, instructions=instructions(), tools=TOOL_SCHEMAS,
        input=question, **extra,
    )
    for _ in range(MAX_TOOL_ROUNDS):
        calls = [item for item in response.output if item.type == "function_call"]
        if not calls:
            break
        outputs = [{"type": "function_call_output", "call_id": c.call_id, "output": run_tool(c)}
                   for c in calls]
        response = client.responses.create(
            model=DEPLOYMENT, instructions=instructions(), tools=TOOL_SCHEMAS,
            input=outputs, previous_response_id=response.id,
        )
    return response


def main():
    missing = [n for n, v in [("AZURE_OPENAI_ENDPOINT", ENDPOINT), ("AZURE_OPENAI_API_KEY", API_KEY),
                              ("AZURE_OPENAI_DEPLOYMENT", DEPLOYMENT)] if not v]
    if missing:
        sys.exit(f"Missing in .env: {', '.join(missing)}")

    client = OpenAI(base_url=f"{ENDPOINT}/openai/v1/", api_key=API_KEY)
    print("Oslo load agent. Ask about electricity consumption in NO1. Type 'exit' to quit.\n")
    previous_id = None
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"exit", "quit", "q"}:
            break
        if not question:
            continue
        response = ask(client, question, previous_id)
        previous_id = response.id
        print(f"\nAgent: {response.output_text}\n")


if __name__ == "__main__":
    main()
