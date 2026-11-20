"""Interactive terminal chat: models load ONCE, answers stream, follow-up questions work.

Run:  uv run python -m query.chat      (or: uv run kb chat)
Commands inside the chat:
  :tech NR      filter next questions to a technology tag (:tech off to clear)
  :mode kb|assistant   kb = answer only from the KB; assistant = KB first, general knowledge allowed
  :new          start a new conversation (clears memory)
  :chunks       toggle printing the retrieved chunks
  :fast         toggle fast mode (skip the reranker, shorter answers)
  :quit         exit
"""
from __future__ import annotations

import time

try:
    import readline  # noqa: F401 - importing it gives input() line editing and arrow-key history
except ImportError:  # pragma: no cover - not available on Windows
    pass

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

from query.engine import MODES, ChatEngine

console = Console()


def handle_command(q: str, eng: ChatEngine, state: dict) -> bool:
    """Run a ':command'. Returns True if the line was a command."""
    cmd, _, arg = q.partition(" ")
    arg = arg.strip()
    if cmd == ":tech":
        state["tech"] = None if arg in ("", "off") else arg
        console.print(f"[dim]technology filter: {state['tech'] or 'off'}[/dim]")
    elif cmd == ":mode":
        if arg in MODES:
            eng.set_mode(arg)
        console.print(f"[dim]mode: {eng.mode} (options: {', '.join(MODES)})[/dim]")
    elif cmd == ":new":
        eng.reset()
        console.print("[dim]new conversation[/dim]")
    elif cmd == ":chunks":
        state["chunks"] = not state["chunks"]
        console.print(f"[dim]show chunks: {state['chunks']}[/dim]")
    elif cmd == ":fast":
        eng.set_fast(not eng.fast)
        console.print(f"[dim]fast mode: {eng.fast}[/dim]")
    else:
        return False
    return True


def main():
    t0 = time.time()
    console.print("[dim]loading models (once per session)...[/dim]")
    eng = ChatEngine()
    eng.warm_up()
    console.print(f"[dim]ready in {time.time() - t0:.1f} s - ask away (:quit to exit)[/dim]")

    state = {"tech": None, "chunks": False}
    while True:
        try:
            q = input("\nkb> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q in (":quit", ":q", "exit"):
            break
        if q.startswith(":"):
            if not handle_command(q, eng, state):
                console.print("[dim]unknown command[/dim]")
            continue

        text, sources, t1 = "", [], time.time()
        try:
            with Live(console=console, refresh_per_second=8, vertical_overflow="visible") as live:
                for ev in eng.reply(q, state["tech"]):
                    if ev["type"] == "sources":
                        sources = ev["sources"]
                        if state["chunks"]:
                            for s in sources:
                                live.console.print(
                                    f"[dim][{s['tag']}] {s['file']} p.{s['page_start']} "
                                    f"{s['section'][:60]}[/dim]")
                    elif ev["type"] == "token":
                        text += ev["text"]
                        live.update(Markdown(text))
        except KeyboardInterrupt:
            console.print("[dim]interrupted (this turn is not kept in memory)[/dim]")
            continue
        except Exception as e:  # noqa: BLE001 - keep the session alive on API/network errors
            console.print(f"[red]error: {e}[/red]")
            continue
        console.print(f"[dim]{time.time() - t1:.1f} s[/dim]")
        for s in sources:
            console.print(f"[dim][{s['tag']}] {s['file']} p.{s['page_start']}-{s['page_end']}[/dim]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
