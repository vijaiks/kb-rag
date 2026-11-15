"""Interactive chat: models load ONCE, then every question costs only search + rerank + LLM.

Run:  uv run python -m query.chat
Commands inside the chat:
  :tech NR      filter next questions to a technology tag (:tech off to clear)
  :chunks       toggle printing the retrieved chunks
  :fast         toggle fast mode (skip the reranker, shorter answers)
  :quit         exit
"""
from __future__ import annotations

import time

from rich.console import Console
from rich.markdown import Markdown

from common import chunks_table, embed_texts, load_config
from llm.client import LLMClient
from query.ask import SYSTEM, build_prompt, retrieve

console = Console()


def main():
    cfg = load_config()
    t0 = time.time()
    console.print("[dim]loading models (once per session)...[/dim]")
    tbl = chunks_table(cfg)
    embed_texts(cfg, ["warm up"])          # loads BGE-M3 onto the GPU
    llm = LLMClient(cfg)
    console.print(f"[dim]ready in {time.time() - t0:.1f} s - ask away (:quit to exit)[/dim]")

    tech, show_chunks, fast = None, False, False
    while True:
        try:
            q = input("\nkb> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q in (":quit", ":q", "exit"):
            break
        if q.startswith(":tech"):
            arg = q.split(maxsplit=1)[1] if " " in q else "off"
            tech = None if arg == "off" else arg
            console.print(f"[dim]technology filter: {tech or 'off'}[/dim]")
            continue
        if q == ":chunks":
            show_chunks = not show_chunks
            console.print(f"[dim]show chunks: {show_chunks}[/dim]")
            continue
        if q == ":fast":
            fast = not fast
            cfg["retrieval"]["reranker"] = None if fast else "BAAI/bge-reranker-v2-m3"
            console.print(f"[dim]fast mode: {fast}[/dim]")
            continue

        t1 = time.time()
        where = f"technology LIKE '%{tech}%'" if tech else None
        chunks = retrieve(cfg, tbl, q, where)
        t_ret = time.time() - t1
        if not chunks:
            console.print("no chunks retrieved")
            continue
        if show_chunks:
            for i, c in enumerate(chunks, 1):
                console.print(f"[dim][S{i}] {c['file_path']} p.{c['page_start']} {c['section_path'][:60]}[/dim]")
        t2 = time.time()
        answer = llm.complete(SYSTEM, build_prompt(q, chunks), max_tokens=700 if fast else None)
        console.print(Markdown(answer))
        console.print(f"[dim]retrieval {t_ret:.1f} s - llm {time.time() - t2:.1f} s[/dim]")
        for i, c in enumerate(chunks, 1):
            console.print(f"[dim][S{i}] {c['file_path']} p.{c['page_start']}-{c['page_end']}[/dim]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
