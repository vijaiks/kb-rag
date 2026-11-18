"""Phase 0 ingestion: scan -> dedupe/diff -> parse (Docling) -> chunk -> embed (BGE-M3) -> LanceDB.

Run:  uv run python -m ingest.pipeline            # folders listed in config phase0_folders
      uv run python -m ingest.pipeline --all      # every file with index == yes in the catalog
      uv run python -m ingest.pipeline --rebuild-fts
Re-running skips files whose hash is unchanged, so it is safe to run after adding files.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from rich.console import Console
from rich.progress import track

from common import (
    Manifest,
    chunks_table,
    embed_texts,
    load_catalog,
    load_config,
    make_chunk_id,
    sha256,
    sql_str,
)

console = Console()


_converter = None
_chunker = None


def parse_and_chunk(cfg: dict, path: Path):
    """Docling converts the file; HybridChunker splits it on headings within a token budget.

    The converter and chunker are created once and reused: Docling's layout and
    table models take seconds to load, so a fresh converter per file wastes minutes
    over a few hundred files."""
    global _converter, _chunker
    if _converter is None:
        from docling.chunking import HybridChunker
        from docling.document_converter import DocumentConverter

        _converter = DocumentConverter()
        _chunker = HybridChunker(
            tokenizer=cfg["embedding"]["model"], max_tokens=cfg["chunking"]["target_tokens"], merge_peers=True
        )
    doc = _converter.convert(str(path)).document
    chunker = _chunker
    out = []
    for ch in chunker.chunk(dl_doc=doc):
        pages = [p.page_no for item in ch.meta.doc_items for p in item.prov] or [0]
        labels = {item.label for item in ch.meta.doc_items}
        ctype = "table" if any("table" in str(l).lower() for l in labels) else "text"
        out.append(
            {
                "text": chunker.contextualize(chunk=ch),  # heading path + text, what gets embedded
                "page_start": min(pages),
                "page_end": max(pages),
                "section_path": " > ".join(ch.meta.headings or []),
                "content_type": ctype,
            }
        )
    return out


def ingest_file(cfg, tbl, man, rel: str, meta: dict, src: Path):
    h = sha256(src)
    prev = man.get(rel)
    if prev and prev[0] == h and prev[1] == "done":
        return "skipped"
    # Do all the work that can fail (parse + embed) BEFORE touching the index, so a
    # failure on a changed file keeps its previous, still-searchable chunks.
    try:
        chunks = parse_and_chunk(cfg, src)
        vecs = embed_texts(cfg, [c["text"] for c in chunks]) if chunks else []
    except Exception as e:  # noqa: BLE001
        man.set(rel, h, 0, "error", f"{e}\n{traceback.format_exc()[-800:]}")
        kept = " (previous version kept in the index)" if prev else ""
        console.print(f"[red]ingest failed[/] {rel}: {e}{kept}")
        return "error"
    if prev:  # changed file: new chunks are ready, now replace the old ones
        tbl.delete(f"file_path = {sql_str(rel)}")
    if not chunks:
        man.set(rel, h, 0, "empty")
        return "empty"
    rows = [
        dict(
            chunk_id=make_chunk_id(rel, h, i),
            vector=v.tolist(),
            file_hash=h,
            file_path=rel,
            title=meta.get("title", Path(rel).stem),
            domain=meta.get("domain", ""),
            technology=meta.get("technology", ""),
            layer=meta.get("layer", ""),
            release=meta.get("release", ""),
            embed_model=cfg["embedding"]["model"],
            **c,
        )
        for i, (c, v) in enumerate(zip(chunks, vecs))
    ]
    tbl.add(rows)
    man.set(rel, h, len(rows), "done")
    return "done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="ignore phase0_folders, index every catalog row")
    ap.add_argument("--rebuild-fts", action="store_true", help="only rebuild the keyword index")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config()
    tbl = chunks_table(cfg, create=True)
    if args.rebuild_fts:
        tbl.create_fts_index("text", replace=True)
        console.print("FTS index rebuilt")
        return

    src_root = Path(cfg["source_dir"])
    catalog = load_catalog(cfg)
    if catalog is None:
        # No catalog configured: index every supported file under source_dir.
        skip = set(cfg.get("exclude_names") or [])
        catalog = {
            str(p.relative_to(src_root)).replace("\\", "/"): {}
            for p in sorted(src_root.rglob("*"))
            if p.is_file() and p.suffix.lower() in (".pdf", ".docx", ".png") and p.name not in skip
        }
        console.print(f"no catalog_csv - indexing all {len(catalog)} supported files under {src_root}")
    folders = set(cfg.get("phase0_folders") or [])
    todo = [
        (rel, meta)
        for rel, meta in catalog.items()
        if (args.all or not folders or rel.split("/")[0] in folders)
        and rel.lower().endswith((".pdf", ".docx", ".png"))
    ]
    if args.limit:
        todo = todo[: args.limit]
    console.print(f"{len(todo)} files to consider; data in {cfg['data_dir']}")

    man = Manifest(cfg["data_dir"])
    counts: dict[str, int] = {}
    try:
        for rel, meta in track(todo, description="ingesting"):
            src = src_root / rel
            if not src.exists():
                console.print(f"[yellow]missing[/] {rel}")
                continue
            r = ingest_file(cfg, tbl, man, rel, meta, src)
            counts[r] = counts.get(r, 0) + 1
    finally:
        # Build the keyword index even if the run was interrupted (Ctrl+C, crash), so the
        # chunks added so far are searchable. LanceDB cannot index an empty table.
        if tbl.count_rows():
            tbl.create_fts_index("text", replace=True)
    console.print(f"result: {counts}; total chunks in index: {tbl.count_rows()}")


if __name__ == "__main__":
    sys.exit(main())
