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

from common import Manifest, chunks_table, embed_texts, load_catalog, load_config, sha256

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
    if prev:  # changed file: drop old chunks first
        tbl.delete(f"file_path = '{rel}'")
    try:
        chunks = parse_and_chunk(cfg, src)
    except Exception as e:  # noqa: BLE001
        man.set(rel, h, 0, "error", f"{e}\n{traceback.format_exc()[-800:]}")
        console.print(f"[red]parse failed[/] {rel}: {e}")
        return "error"
    if not chunks:
        man.set(rel, h, 0, "empty")
        return "empty"
    vecs = embed_texts(cfg, [c["text"] for c in chunks])
    rows = [
        dict(
            chunk_id=f"{h[:12]}:{i:04d}",
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
    for rel, meta in track(todo, description="ingesting"):
        src = src_root / rel
        if not src.exists():
            console.print(f"[yellow]missing[/] {rel}")
            continue
        r = ingest_file(cfg, tbl, man, rel, meta, src)
        counts[r] = counts.get(r, 0) + 1

    tbl.create_fts_index("text", replace=True)
    console.print(f"result: {counts}; total chunks in index: {tbl.count_rows()}")


if __name__ == "__main__":
    sys.exit(main())
