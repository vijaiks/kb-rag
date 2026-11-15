"""Agentic layer on top of the RAG pipeline, with a human in the loop.

Roles:
  Orchestrator (Claude, tool-use loop) - plans, calls tools, decides when done.
  Researcher   (tool: search_kb)       - hybrid retrieve + rerank from LanceDB.
  Critic       (tool: verify_answer)   - second LLM pass that checks every [S#]
                                         claim against the retrieved chunks (ack / rework).
  Human        (tool: ask_human, and approval gate on every action) - you.

Run:  uv run python -m agents.loop "Compare LTE and NR HARQ, verify it, and save a note"
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown

from common import chunks_table, load_config
from llm.client import LLMClient
from query.ask import retrieve

console = Console()
MAX_TURNS = 12

TOOLS = [
    {
        "name": "search_kb",
        "description": "Search the local telecom knowledge base. Returns the best chunks with "
        "source tags [S#] (file, pages, section). Call again with a refined query if results are weak.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "technology": {"type": "string", "description": "Optional filter: NR, LTE, O-RAN, 6G"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "verify_answer",
        "description": "Ask the critic agent to check a drafted answer: is every claim backed by "
        "the cited chunk? Returns verdict 'ack' or 'rework' with reasons. Always verify before "
        "presenting a final answer or taking an action based on it.",
        "input_schema": {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        },
    },
    {
        "name": "ask_human",
        "description": "Ask the human a clarifying question and wait for their typed reply.",
        "input_schema": {
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
        },
    },
    {
        "name": "save_note",
        "description": "ACTION (needs human approval): save a markdown note into data/notes/. "
        "Use for verified summaries the human may want to keep.",
        "input_schema": {
            "type": "object",
            "properties": {"filename": {"type": "string"}, "markdown": {"type": "string"}},
            "required": ["filename", "markdown"],
        },
    },
]

ORCH_SYSTEM = """You are the orchestrator of a telecom knowledge-base assistant for a RAN software architect.
Work in steps: search_kb (possibly several refined queries) -> draft an answer with [S#] citations ->
verify_answer -> if 'rework', fix the answer using only supported claims and verify again ->
present the final answer. Only claims from retrieved chunks may be stated as fact; everything else is
clearly marked as your own reasoning. Use ask_human when the request is ambiguous. Propose save_note
only when the human asked for a note or the answer is clearly worth keeping. Be concise."""

CRITIC_SYSTEM = """You are a strict verifier. You get an ANSWER with [S#] citation tags and the SOURCES.
Check every factual claim: is it supported by the chunk it cites? Reply as JSON only:
{"verdict": "ack" | "rework", "unsupported_claims": ["..."], "notes": "..."}
'ack' only if every claim is supported. Citing the wrong source counts as unsupported."""


class Session:
    def __init__(self):
        self.cfg = load_config()
        self.tbl = chunks_table(self.cfg)
        self.llm = LLMClient(self.cfg)
        self.last_chunks: list[dict] = []

    # ---- tool implementations -------------------------------------------
    def search_kb(self, query: str, technology: str | None = None) -> str:
        where = f"technology LIKE '%{technology}%'" if technology else None
        self.last_chunks = retrieve(self.cfg, self.tbl, query, where)
        if not self.last_chunks:
            return "No chunks found."
        out = []
        for i, c in enumerate(self.last_chunks, 1):
            out.append(
                f"[S{i}] {c['file_path']} pp.{c['page_start']}-{c['page_end']} "
                f"{c['section_path'] or ''}\n{c['text'][:1200]}"
            )
        return "\n\n".join(out)

    def verify_answer(self, answer: str) -> str:
        if not self.last_chunks:
            return json.dumps({"verdict": "rework", "notes": "No sources retrieved yet."})
        src = "\n\n".join(
            f"[S{i}] {c['file_path']}\n{c['text'][:1200]}" for i, c in enumerate(self.last_chunks, 1)
        )
        console.print("[dim]critic checking the draft...[/dim]")
        return self.llm.complete(CRITIC_SYSTEM, f"ANSWER:\n{answer}\n\nSOURCES:\n{src}", max_tokens=800)

    def ask_human(self, question: str) -> str:
        console.print(f"\n[bold yellow]Agent asks:[/bold yellow] {question}")
        return input("your reply> ").strip() or "(no answer)"

    def save_note(self, filename: str, markdown: str) -> str:
        # human-in-the-loop gate: every action must be approved explicitly
        safe = Path(filename).name
        if not safe.endswith(".md"):
            safe += ".md"
        console.print(f"\n[bold red]ACTION REQUEST[/bold red]: save note data/notes/{safe}")
        console.print(Markdown(markdown[:1500]))
        if input("approve? [y/N]> ").strip().lower() != "y":
            return "Human DECLINED the action. Do not retry it."
        p = Path(self.cfg["data_dir"]) / "notes"
        p.mkdir(parents=True, exist_ok=True)
        (p / safe).write_text(markdown, encoding="utf-8")
        return f"Saved to {p / safe}"

    def run_tool(self, name: str, args: dict) -> str:
        try:
            return getattr(self, name)(**args)
        except Exception as e:  # noqa: BLE001
            return f"Tool error: {e}"


def main():
    if len(sys.argv) < 2:
        print('usage: uv run python -m agents.loop "your request"')
        return 1
    task = " ".join(sys.argv[1:])
    s = Session()
    messages = [{"role": "user", "content": task}]
    for turn in range(MAX_TURNS):
        resp = s.llm.client.messages.create(
            model=s.cfg["llm"]["model"],
            max_tokens=s.cfg["llm"]["max_tokens"],
            system=ORCH_SYSTEM,
            tools=TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason != "tool_use":
            for b in resp.content:
                if getattr(b, "text", None):
                    console.print(Markdown(b.text))
            return 0
        results = []
        for b in resp.content:
            if b.type == "tool_use":
                console.print(f"[dim]-> {b.name}({json.dumps(b.input)[:120]})[/dim]")
                results.append(
                    {"type": "tool_result", "tool_use_id": b.id, "content": s.run_tool(b.name, b.input)}
                )
        messages.append({"role": "user", "content": results})
    console.print("[red]Stopped: reached the turn limit.[/red]")
    return 1


if __name__ == "__main__":
    sys.exit(main())
