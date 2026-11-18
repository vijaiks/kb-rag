"""Failure paths: a failed re-ingest keeps old chunks; a missing keyword index doesn't break search."""
import tempfile
from pathlib import Path

import lancedb
import numpy as np

from common import Manifest, sql_str
from ingest import pipeline
from query import ask

CFG = {
    "embedding": {"model": "test-model"},
    "retrieval": {
        "top_k_candidates": 5,
        "top_k_final": 5,
        "max_per_file": 5,
        "hybrid": True,
        "reranker": None,
    },
}


def make_table(rows):
    return lancedb.connect(tempfile.mkdtemp()).create_table("chunks", data=rows)


def old_chunk_row():
    return {
        "chunk_id": "old:0000",
        "text": "old CORESET text",
        "vector": [1.0, 0.0],
        "file_hash": "oldhash",
        "file_path": "a/doc.pdf",
        "title": "doc",
        "page_start": 1,
        "page_end": 1,
        "section_path": "",
        "content_type": "text",
        "domain": "",
        "technology": "NR",
        "layer": "",
        "release": "",
        "embed_model": "test-model",
    }


def changed_file(tmp_path: Path):
    """A manifest that says a/doc.pdf was indexed with an older hash, plus the new file on disk."""
    src = tmp_path / "doc.pdf"
    src.write_bytes(b"new content")
    man = Manifest(str(tmp_path))
    man.set("a/doc.pdf", "oldhash", 1, "done")
    return src, man


def test_failed_parse_keeps_previous_chunks(tmp_path, monkeypatch):
    tbl = make_table([old_chunk_row()])
    src, man = changed_file(tmp_path)

    def boom(cfg, path):
        raise RuntimeError("docling choked")

    monkeypatch.setattr(pipeline, "parse_and_chunk", boom)
    assert pipeline.ingest_file(CFG, tbl, man, "a/doc.pdf", {}, src) == "error"
    assert tbl.count_rows(f"file_path = {sql_str('a/doc.pdf')}") == 1  # old version still searchable
    assert man.get("a/doc.pdf")[1] == "error"


def test_failed_embedding_keeps_previous_chunks(tmp_path, monkeypatch):
    tbl = make_table([old_chunk_row()])
    src, man = changed_file(tmp_path)
    monkeypatch.setattr(pipeline, "parse_and_chunk", lambda cfg, path: [{"text": "x"}])

    def oom(cfg, texts):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(pipeline, "embed_texts", oom)
    assert pipeline.ingest_file(CFG, tbl, man, "a/doc.pdf", {}, src) == "error"
    assert tbl.count_rows() == 1


def test_successful_reingest_replaces_chunks(tmp_path, monkeypatch):
    tbl = make_table([old_chunk_row()])
    src, man = changed_file(tmp_path)
    new = {"text": "new text", "page_start": 2, "page_end": 2, "section_path": "", "content_type": "text"}
    monkeypatch.setattr(pipeline, "parse_and_chunk", lambda cfg, path: [new])
    monkeypatch.setattr(pipeline, "embed_texts", lambda cfg, texts: np.array([[0.0, 1.0]], dtype=np.float32))

    assert pipeline.ingest_file(CFG, tbl, man, "a/doc.pdf", {}, src) == "done"
    assert [r["text"] for r in tbl.search().limit(10).to_list()] == ["new text"]


def test_retrieve_without_keyword_index_falls_back_to_vectors(monkeypatch, capsys):
    tbl = make_table([old_chunk_row()])  # note: no create_fts_index
    monkeypatch.setattr(ask, "embed_texts", lambda cfg, texts: np.array([[1.0, 0.0]], dtype=np.float32))

    hits = ask.retrieve(CFG, tbl, "CORESET", None)
    assert [h["chunk_id"] for h in hits] == ["old:0000"]
    assert "keyword search unavailable" in capsys.readouterr().err


def test_retrieve_skips_keyword_search_when_hybrid_off(monkeypatch):
    tbl = make_table([old_chunk_row()])
    monkeypatch.setattr(ask, "embed_texts", lambda cfg, texts: np.array([[1.0, 0.0]], dtype=np.float32))
    monkeypatch.setattr(ask, "keyword_hits", lambda *a: (_ for _ in ()).throw(AssertionError("called")))

    cfg = {**CFG, "retrieval": {**CFG["retrieval"], "hybrid": False}}
    assert [h["chunk_id"] for h in ask.retrieve(cfg, tbl, "CORESET", None)] == ["old:0000"]
