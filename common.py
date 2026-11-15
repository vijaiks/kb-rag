"""Shared helpers: config, catalog, manifest, embedding model, LanceDB."""
from __future__ import annotations

import csv
import hashlib
import os
import sqlite3
from pathlib import Path

import lancedb
import yaml

ROOT = Path(__file__).resolve().parent


def load_config(path: str | None = None) -> dict:
    cfg_path = Path(path or ROOT / "config.yaml")
    if not cfg_path.exists():
        raise SystemExit("No config.yaml - create it from the template: cp config.example.yaml config.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    for key in ("data_dir", "source_dir", "catalog_csv"):
        if cfg.get(key):
            cfg[key] = str(Path(os.path.expanduser(cfg[key])))
    Path(cfg["data_dir"]).mkdir(parents=True, exist_ok=True)
    return cfg


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_catalog(cfg: dict) -> dict[str, dict] | None:
    """Rows of the catalog CSV keyed by relative path (forward slashes), index == yes only.

    Returns None when no catalog_csv is configured or the file does not exist:
    the caller then indexes every supported file under source_dir."""
    path = cfg.get("catalog_csv")
    if not path or not Path(path).exists():
        return None
    out = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("index", "yes").strip().lower() == "yes":
                out[r["path"].replace("\\", "/")] = r
    return out


class Manifest:
    """SQLite table of every indexed file: path, hash, chunk count. Makes ingestion resumable."""

    def __init__(self, data_dir: str):
        self.con = sqlite3.connect(Path(data_dir) / "manifest.sqlite")
        self.con.execute(
            "CREATE TABLE IF NOT EXISTS files (path TEXT PRIMARY KEY, hash TEXT, chunks INT, "
            "status TEXT, error TEXT, updated TEXT DEFAULT CURRENT_TIMESTAMP)"
        )

    def get(self, path: str):
        return self.con.execute("SELECT hash, status FROM files WHERE path=?", (path,)).fetchone()

    def set(self, path: str, h: str, chunks: int, status: str, error: str = ""):
        self.con.execute(
            "INSERT OR REPLACE INTO files (path, hash, chunks, status, error, updated) "
            "VALUES (?,?,?,?,?,CURRENT_TIMESTAMP)",
            (path, h, chunks, status, error[:500]),
        )
        self.con.commit()

    def all(self):
        return self.con.execute("SELECT path, hash, chunks, status FROM files").fetchall()


_embedder = None


def embedder(cfg: dict):
    """BGE-M3 via sentence-transformers, loaded once, on the GPU."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer(cfg["embedding"]["model"], device=cfg["embedding"]["device"])
    return _embedder


def embed_texts(cfg: dict, texts: list[str]):
    return embedder(cfg).encode(
        texts, batch_size=cfg["embedding"]["batch_size"], normalize_embeddings=True, show_progress_bar=False
    )


def chunks_table(cfg: dict, create: bool = False):
    db = lancedb.connect(str(Path(cfg["data_dir"]) / "lancedb"))
    if "chunks" in db.table_names():
        return db.open_table("chunks")
    if not create:
        raise SystemExit("No index yet. Run: uv run python -m ingest.pipeline")
    import pyarrow as pa

    schema = pa.schema(
        [
            ("chunk_id", pa.string()),
            ("text", pa.string()),
            ("vector", pa.list_(pa.float32(), cfg["embedding"]["dim"])),
            ("file_hash", pa.string()),
            ("file_path", pa.string()),
            ("title", pa.string()),
            ("page_start", pa.int32()),
            ("page_end", pa.int32()),
            ("section_path", pa.string()),
            ("content_type", pa.string()),
            ("domain", pa.string()),
            ("technology", pa.string()),
            ("layer", pa.string()),
            ("release", pa.string()),
            ("embed_model", pa.string()),
        ]
    )
    return db.create_table("chunks", schema=schema)
