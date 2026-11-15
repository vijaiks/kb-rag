"""MCP server exposing the telecom knowledge base to the Claude desktop app.

Claude calls these tools during a normal chat; retrieval (BGE-M3 + LanceDB +
reranker) runs locally on this machine. Nothing but the chunks Claude asked
for leaves the laptop.

IMPORTANT: this server speaks the MCP protocol on stdout. Never print() here -
anything on stdout corrupts the protocol. Diagnostics go to stderr.

Run manually to test:  uv run python -m mcp_server.server
(then it waits on stdin; Ctrl+C to exit)
"""
from __future__ import annotations

import sys

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("kb-rag")

_state: dict = {}


def _session():
    """Lazy init: the first tool call loads models (10-20 s once), later calls are fast."""
    if not _state:
        print("kb-rag: loading models...", file=sys.stderr)
        from common import chunks_table, embed_texts, load_config

        cfg = load_config()
        _state["cfg"] = cfg
        _state["tbl"] = chunks_table(cfg)
        embed_texts(cfg, ["warm up"])
        print("kb-rag: ready", file=sys.stderr)
    return _state["cfg"], _state["tbl"]


@mcp.tool()
def search_kb(query: str, technology: str = "") -> str:
    """Search the user's personal telecom/cloud/coding knowledge base (curated PDFs:
    5G NR, LTE, 6G, O-RAN, PHY/MAC/RLC protocol internals, Kubernetes, DevOps,
    system design, AI/ML). Returns the most relevant text chunks, each tagged with
    source file, pages and section. Use this whenever the user asks what their
    knowledge base / library / saved documents say about a topic, or wants an
    answer grounded in their own collection. Cite file and page in your answer.
    technology: optional filter - NR, LTE, O-RAN, 6G, Kubernetes, etc."""
    from query.ask import retrieve

    cfg, tbl = _session()
    where = f"technology LIKE '%{technology}%'" if technology else None
    chunks = retrieve(cfg, tbl, query, where)
    if not chunks:
        return "No matching chunks in the knowledge base."
    out = []
    for i, c in enumerate(chunks, 1):
        pages = f"p.{c['page_start']}" if c["page_start"] == c["page_end"] else f"pp.{c['page_start']}-{c['page_end']}"
        out.append(
            f"[S{i}] {c['file_path']}, {pages}"
            + (f", {c['section_path']}" if c["section_path"] else "")
            + f" (tech: {c['technology'] or 'n/a'})\n{c['text']}"
        )
    return "\n\n---\n\n".join(out)


@mcp.tool()
def kb_stats() -> str:
    """How big the knowledge base index is and what it covers: chunk count and
    the number of indexed chunks per technology tag."""
    _, tbl = _session()
    total = tbl.count_rows()
    rows = tbl.search().select(["technology"]).limit(100000).to_list()
    from collections import Counter

    c = Counter((r["technology"] or "untagged").split("; ")[0] for r in rows)
    lines = [f"Total chunks: {total}"] + [f"- {k}: {v}" for k, v in c.most_common(15)]
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run(transport="stdio")
