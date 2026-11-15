# kb-rag — a local RAG system over a telecom engineer's knowledge base

Ask questions against your own library of PDFs — 5G NR, LTE, O-RAN, protocol internals,
Kubernetes, system design — and get answers **cited to file and page**, or an honest
*"not found in the knowledge base"* instead of a hallucination.

Everything except the final answer generation runs **locally**: parsing, embeddings,
vector search and reranking happen on your machine (GPU-accelerated). Only the ~8
retrieved text chunks per question are sent to the LLM. The knowledge base also plugs
into the Claude desktop app as an MCP tool, turning a normal Claude chat into a
personal assistant over your documents.

Built and tested on WSL2 (Ubuntu, aarch64) with an NVIDIA GPU; any Linux with
CUDA — or plain CPU — works.

## Architecture

```
                   OFFLINE (ingestion, local)
 PDFs/DOCX/PNG ──> scan+hash ──> parse (Docling) ──> chunk by heading
      │               skip unchanged      │              ~500 tokens
 _kb_catalog.csv ─────────────────────────┘                  │
  (what to index, tags)                              embed (BGE-M3, GPU)
                                                             │
                                             ┌───────────────▼───────────────┐
                                             │   LanceDB on local disk       │
                                             │ vectors + BM25 + metadata     │
                                             └───────────────┬───────────────┘
                   ONLINE (per question)                     │
 question ──> embed ──> vector + keyword search ──> RRF fuse ──> rerank
                                                             │  (cross-encoder)
                                            top 8 chunks, max 3 per file
                                                             │
                                      LLM answers ONLY from chunks, cites [S#]
```

Key design choices:

- **Hybrid retrieval.** Dense vectors catch meaning ("random access" finds RACH);
  BM25 catches exact tokens embeddings blur (CORESET, TS 38.213, RNTI). Fused with
  reciprocal rank fusion, then reranked by a cross-encoder.
- **Idempotent ingestion.** Every file is tracked by SHA-256 in a SQLite manifest.
  Re-running ingestion is always safe: unchanged files are skipped, changed files are
  re-indexed, deleted files can be pruned. Crash mid-run? Just run again.
- **Grounding contract.** The system prompt forces citations per claim and an explicit
  refusal when the sources don't contain the answer.
- **Catalog-driven scope.** A CSV decides which files are indexed and carries tags
  (technology, protocol layer, 3GPP release) that become queryable metadata.

## Stack

| Layer | Tool |
|---|---|
| Env/packages | [uv](https://docs.astral.sh/uv/), Python 3.12+ |
| Parsing + chunking | [Docling](https://github.com/docling-project/docling) (layout-aware, OCR, tables) |
| Embeddings | [BGE-M3](https://huggingface.co/BAAI/bge-m3) via sentence-transformers, local GPU/CPU |
| Vector + keyword store | [LanceDB](https://lancedb.com) (embedded, file-based, no server) |
| Reranker | [bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) cross-encoder |
| Answering LLM | Claude (Opus/Sonnet/Haiku) via the Anthropic API, behind a vendor-neutral wrapper |
| Assistant integration | [MCP](https://modelcontextprotocol.io) server for the Claude desktop app |

## Quick start

```bash
git clone https://github.com/vijaiks/kb-rag && cd kb-rag
cp config.example.yaml config.yaml   # then edit: source_dir = your document folder
uv sync
uv run python check_gpu.py                 # verify torch sees your GPU (CPU also fine)
export ANTHROPIC_API_KEY="sk-ant-your-key"   # put the real key in ~/.bashrc
uv run python -m ingest.pipeline --all     # index everything (resumable)
uv run python -m query.ask "How does 2-step RACH differ from 4-step RACH?"
```

## How to use the framework

**One-shot questions** (fresh process each time, ~15 s model load included):

```bash
uv run python -m query.ask "What is CORESET 0 and how does the UE find it?"
uv run python -m query.ask --tech LTE "Explain the MAC scheduler"
uv run python -m query.ask --show-chunks "..."   # inspect what retrieval returned
```

**Interactive chat** (models load once, then ~seconds per question):

```bash
uv run python -m query.chat
kb> what does CSI-RS measure?
kb> :tech NR          # filter to a technology; :tech off to clear
kb> :chunks           # toggle showing retrieved chunks
kb> :fast             # skip reranker + shorter answers for quick lookups
```

**Agent mode** (research → verify → act, human approves every action):

```bash
uv run python -m agents.loop "Compare HARQ in LTE and NR, verify it, and save a note"
```

An orchestrator agent searches the KB, a critic agent checks every cited claim against
the retrieved chunks, and actions (like saving a note) wait for your explicit `y`.

**Claude desktop assistant (MCP):** register `mcp_server/server.py` in the Claude
desktop app's `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "kb-rag": {
      "command": "wsl.exe",
      "args": ["bash", "-lc", "cd ~/kb-rag && exec uv run python -m mcp_server.server"]
    }
  }
}
```

Restart the app, then chat normally: *"What does my knowledge base say about SRS-based
beamforming? Cite files and pages."* Claude calls the local `search_kb` tool; retrieval
never leaves your machine.

**Debugging retrieval:** when an answer looks wrong, run the same question with
`--show-chunks`. If the right file isn't in the 8 chunks, it's a retrieval problem
(tune chunking/top-k/filters); if it is, it's a prompting problem. The LLM only ever
sees those 8 chunks.

## Updating the knowledge base (add / replace / remove PDFs)

The manifest + content hashes make maintenance a matter of editing files and re-running
one command. All cases:

**Add new PDFs** — drop them into the source folder, add a catalog row (or regenerate
the catalog), then:

```bash
uv run python -m ingest.pipeline --all
```

Only the new files are parsed and embedded; everything else is skipped by hash.
Takes seconds per file.

**Replace an outdated PDF with a newer version** — overwrite the file (same name) or
delete old + add new. On the next ingestion run the changed hash is detected, the old
chunks for that file are deleted from LanceDB, and the new content is indexed. No
duplicate or stale chunks remain.

**Remove outdated PDFs** — two options:
- *Soft remove* (file stays on disk, leaves the index): set `index = no` in
  `_kb_catalog.csv` for that row, then run the prune step below.
- *Hard remove*: delete the file from the source folder, then prune.

```bash
uv run python -m ingest.prune          # removes chunks + manifest rows for files
                                       # that are deleted or catalog-excluded
```

**Verify the index state** at any time:

```bash
uv run python -m ingest.pipeline --all   # prints {done/skipped/empty} + total chunks
uv run python -c "from common import *; m=Manifest(load_config()['data_dir']); \
print(*m.all(), sep='\n')"               # per-file status
```

**Changing the embedding model** is the one maintenance action that requires a full
rebuild (old and new vectors can't be mixed): delete the `data/lancedb` folder and the
manifest, update `config.yaml`, re-run ingestion. Each chunk stores `embed_model`, so a
mismatch is detectable.

**Suggested routine:** keep dropping files into your source folder as you collect them;
run `ingest.pipeline --all` weekly (or add a cron/scheduled task). The run is cheap -
it only ever processes what changed.

## Project layout

```
common.py            config, catalog, SHA-256 manifest, embedder, LanceDB schema
ingest/pipeline.py   scan -> dedupe -> parse -> chunk -> embed -> store (+ FTS index)
ingest/prune.py      remove chunks for deleted/excluded files
query/ask.py         hybrid retrieve -> RRF -> rerank -> cited answer (CLI)
query/chat.py        interactive REPL, models loaded once
agents/loop.py       orchestrator + critic + human-approved actions
llm/client.py        vendor-neutral LLM interface (Anthropic implementation)
mcp_server/server.py MCP server exposing search_kb / kb_stats to the Claude app
config.yaml          paths, models, chunk sizes, retrieval settings
```

## Roadmap

- Vision captions for image-only pages (slides, screenshots) at ingestion time
- Golden question set + Ragas evaluation harness
- 3GPP / O-RAN specification collection with clause-based chunking
- Streamlit chat UI

## License

MIT (code only - bring your own documents).
