"""Phase 0 query: hybrid retrieve (vector + BM25, RRF) -> rerank -> answer with citations.

Run:  uv run python -m query.ask "How does 2-step RACH differ from 4-step RACH?"
      uv run python -m query.ask --show-chunks "..."     # print retrieved chunks too
      uv run python -m query.ask --tech NR "..."          # metadata filter
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict

from rich.console import Console
from rich.markdown import Markdown

from common import chunks_table, embed_texts, load_config, tech_filter
from llm.client import LLMClient

console = Console()

SYSTEM = """You are a telecom and software architecture assistant answering ONLY from the provided sources.
Rules:
- Every factual claim must end with a citation tag like [S2]. Use several tags when several sources agree.
- If the sources do not contain the answer, say "Not found in the knowledge base" and stop. Do not use outside knowledge.
- Prefer the source with the most specific technical detail. When sources disagree (e.g. different 3GPP releases), say so and cite both.
- Be concise and use plain language; expand acronyms once."""


def rrf(rank_lists: list[list[str]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = defaultdict(float)
    for ranks in rank_lists:
        for i, cid in enumerate(ranks):
            scores[cid] += 1.0 / (k + i + 1)
    return scores


def keyword_hits(tbl, question: str, where: str | None, n: int) -> list[dict]:
    """BM25 hits, or [] when the keyword index is missing (e.g. the first ingest was
    interrupted before it was built) - retrieval then degrades to vector-only.
    The warning goes to stderr: the MCP server uses stdout for its protocol."""
    fq = tbl.search(question, query_type="fts").limit(n)
    if where:
        fq = fq.where(where, prefilter=True)
    try:
        return fq.to_list()
    except Exception as e:  # noqa: BLE001
        print(
            f"warning: keyword search unavailable, using vector search only ({e}). "
            "Fix: uv run python -m ingest.pipeline --rebuild-fts",
            file=sys.stderr,
        )
        return []


def retrieve(cfg, tbl, question: str, where: str | None):
    r = cfg["retrieval"]
    n = r["top_k_candidates"]
    qv = embed_texts(cfg, [question])[0].tolist()
    vq = tbl.search(qv).limit(n)
    if where:
        vq = vq.where(where, prefilter=True)
    vhits = vq.to_list()
    fhits = keyword_hits(tbl, question, where, n) if r["hybrid"] else []
    by_id = {h["chunk_id"]: h for h in vhits + fhits}
    fused = rrf([[h["chunk_id"] for h in vhits], [h["chunk_id"] for h in fhits]]) if r["hybrid"] else {
        h["chunk_id"]: 1 / (i + 1) for i, h in enumerate(vhits)
    }
    cands = [by_id[c] for c, _ in sorted(fused.items(), key=lambda x: -x[1])]
    return rerank(cfg, question, cands)


_reranker = None


def rerank(cfg, question, cands):
    global _reranker
    r = cfg["retrieval"]
    if r.get("reranker") and cands:
        if _reranker is None:
            from sentence_transformers import CrossEncoder

            _reranker = CrossEncoder(r["reranker"], device=cfg["embedding"]["device"])
        scores = _reranker.predict([(question, c["text"]) for c in cands])
        cands = [c for _, c in sorted(zip(scores, cands), key=lambda x: -x[0])]
    out, per_file = [], defaultdict(int)
    for c in cands:
        if per_file[c["file_path"]] >= r["max_per_file"]:
            continue
        per_file[c["file_path"]] += 1
        out.append(c)
        if len(out) >= r["top_k_final"]:
            break
    return out


def build_prompt(question, chunks):
    blocks = []
    for i, c in enumerate(chunks, 1):
        pages = f"p.{c['page_start']}" if c["page_start"] == c["page_end"] else f"pp.{c['page_start']}-{c['page_end']}"
        blocks.append(f"[S{i}] {c['file_path']}, {pages}, {c['section_path'] or 'n/a'}\n{c['text']}")
    return "Sources:\n\n" + "\n\n---\n\n".join(blocks) + f"\n\nQuestion: {question}"


def ask(question: str, tech: str | None = None, show_chunks: bool = False):
    cfg = load_config()
    tbl = chunks_table(cfg)
    where = tech_filter(tech)
    chunks = retrieve(cfg, tbl, question, where)
    if not chunks:
        console.print("No chunks retrieved."); return
    if show_chunks:
        for i, c in enumerate(chunks, 1):
            console.rule(f"S{i} {c['file_path']} p.{c['page_start']}")
            console.print(c["text"][:700])
    answer = LLMClient(cfg).complete(SYSTEM, build_prompt(question, chunks))
    console.print(Markdown(answer))
    console.rule("Sources")
    for i, c in enumerate(chunks, 1):
        console.print(f"[S{i}] {c['file_path']}  p.{c['page_start']}-{c['page_end']}  {c['section_path']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="+")
    ap.add_argument("--tech", help="filter on technology tag, e.g. NR, LTE, O-RAN")
    ap.add_argument("--show-chunks", action="store_true")
    a = ap.parse_args()
    ask(" ".join(a.question), a.tech, a.show_chunks)


if __name__ == "__main__":
    sys.exit(main())
