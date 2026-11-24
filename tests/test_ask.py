"""query/ask.py: RRF fusion, retrieval wiring, reranking, prompt building, CLI entry points."""
import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

from query import ask


def chunk(cid, path="a.pdf", ps=1, pe=1, section="Sec", text=None):
    return {"chunk_id": cid, "file_path": path, "page_start": ps, "page_end": pe,
            "section_path": section, "text": text or f"text {cid}"}


def cfg(**retrieval):
    r = {"top_k_candidates": 5, "top_k_final": 3, "max_per_file": 2, "hybrid": True, "reranker": None}
    return {"retrieval": {**r, **retrieval}, "embedding": {"device": "cpu"}}


class FakeQuery:
    def __init__(self, hits, log):
        self.hits, self.log = hits, log

    def limit(self, n):
        self.log.append(("limit", n))
        return self

    def where(self, clause, prefilter=False):
        self.log.append(("where", clause, prefilter))
        return self

    def to_list(self):
        return self.hits


class FakeTable:
    def __init__(self, vector_hits, fts_hits):
        self.vector_hits, self.fts_hits, self.log = vector_hits, fts_hits, []

    def search(self, q, query_type=None):
        self.log.append(("search", "fts" if query_type == "fts" else "vector"))
        return FakeQuery(self.fts_hits if query_type == "fts" else self.vector_hits, self.log)


@pytest.fixture(autouse=True)
def fake_embed(monkeypatch):
    monkeypatch.setattr(ask, "embed_texts", lambda c, texts: np.array([[0.1, 0.2]] * len(texts)))
    monkeypatch.setattr(ask, "_reranker", None)


# ---- rrf --------------------------------------------------------------------
def test_rrf_scores_and_fusion():
    s = ask.rrf([["a", "b"], ["b", "c"]], k=60)
    assert s["a"] == pytest.approx(1 / 61)
    assert s["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert max(s, key=s.get) == "b"           # appearing in both lists wins
    assert ask.rrf([[], []]) == {}


# ---- retrieve ---------------------------------------------------------------
def test_retrieve_fuses_vector_and_keyword_hits():
    tbl = FakeTable([chunk("v1"), chunk("both", path="b.pdf")], [chunk("both", path="b.pdf"), chunk("k1", path="c.pdf")])
    out = ask.retrieve(cfg(), tbl, "q", None)
    assert out[0]["chunk_id"] == "both"
    assert {c["chunk_id"] for c in out} == {"both", "v1", "k1"}


def test_retrieve_applies_prefiltered_where_to_both_searches():
    tbl = FakeTable([chunk("v1")], [chunk("k1")])
    ask.retrieve(cfg(), tbl, "q", "technology LIKE '%NR%'")
    assert tbl.log.count(("where", "technology LIKE '%NR%'", True)) == 2


def test_retrieve_vector_only_when_hybrid_off():
    tbl = FakeTable([chunk("v1"), chunk("v2")], [chunk("k1")])
    out = ask.retrieve(cfg(hybrid=False), tbl, "q", None)
    assert [c["chunk_id"] for c in out] == ["v1", "v2"]
    assert ("search", "fts") not in tbl.log


def test_retrieve_empty():
    assert ask.retrieve(cfg(), FakeTable([], []), "q", None) == []


def test_keyword_hits_failure_returns_empty_and_warns_on_stderr(capsys):
    tbl = MagicMock()
    tbl.search.return_value.limit.return_value.to_list.side_effect = RuntimeError("no index")
    assert ask.keyword_hits(tbl, "q", None, 5) == []
    cap = capsys.readouterr()
    assert "keyword search unavailable" in cap.err and cap.out == ""


# ---- rerank -----------------------------------------------------------------
def test_rerank_caps_per_file_and_top_k():
    cands = [chunk(f"a{i}", path="a.pdf") for i in range(4)] + [chunk("b0", path="b.pdf"), chunk("c0", path="c.pdf")]
    out = ask.rerank(cfg(top_k_final=4, max_per_file=2), "q", cands)
    assert [c["chunk_id"] for c in out] == ["a0", "a1", "b0", "c0"]


def test_rerank_with_cross_encoder_reorders_and_loads_once(monkeypatch):
    loads = []

    class FakeCE:
        def __init__(self, name, device):
            loads.append((name, device))

        def predict(self, pairs):
            return [0.1, 0.9, 0.5]

    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(CrossEncoder=FakeCE))
    c = cfg(reranker="rr-model", max_per_file=9)
    cands = [chunk("x"), chunk("y"), chunk("z")]
    assert [h["chunk_id"] for h in ask.rerank(c, "q", cands)] == ["y", "z", "x"]
    ask.rerank(c, "q", cands)
    assert loads == [("rr-model", "cpu")]


# ---- build_prompt -----------------------------------------------------------
def test_build_prompt_formats_sources_and_pages():
    p = ask.build_prompt("why?", [chunk("1", ps=3, pe=3), chunk("2", path="b.pdf", ps=4, pe=6, section="")])
    assert "[S1] a.pdf, p.3, Sec\ntext 1" in p
    assert "[S2] b.pdf, pp.4-6, n/a\ntext 2" in p
    assert p.startswith("Sources:") and p.endswith("Question: why?")


# ---- ask() / main() ---------------------------------------------------------
@pytest.fixture
def wired(monkeypatch):
    llm = MagicMock()
    llm.complete.return_value = "answer [S1]"
    monkeypatch.setattr(ask, "load_config", lambda: cfg())
    monkeypatch.setattr(ask, "chunks_table", lambda c: "tbl")
    monkeypatch.setattr(ask, "LLMClient", lambda c: llm)
    monkeypatch.setattr(ask, "console", MagicMock())
    return llm


def test_ask_answers_with_sources_and_tech_filter(wired, monkeypatch):
    seen = {}
    monkeypatch.setattr(ask, "retrieve", lambda c, t, q, where: seen.update(q=q, where=where) or [chunk("1")])
    ask.ask("what is X?", tech="NR")
    assert seen == {"q": "what is X?", "where": "technology LIKE '%NR%'"}
    system, prompt = wired.complete.call_args.args
    assert system == ask.SYSTEM and "Question: what is X?" in prompt
    printed = " ".join(str(c.args[0]) for c in ask.console.print.call_args_list)
    assert "[S1] a.pdf" in printed


def test_ask_no_chunks_skips_llm(wired, monkeypatch):
    monkeypatch.setattr(ask, "retrieve", lambda *a: [])
    ask.ask("q")
    wired.complete.assert_not_called()
    assert "No chunks" in ask.console.print.call_args.args[0]


def test_main_parses_arguments(monkeypatch):
    got = {}
    monkeypatch.setattr(ask, "ask", lambda q, tech, show: got.update(q=q, tech=tech, show=show))
    monkeypatch.setattr(sys, "argv", ["ask", "--tech", "LTE", "--show-chunks", "how", "does", "HARQ", "work"])
    ask.main()
    assert got == {"q": "how does HARQ work", "tech": "LTE", "show": True}
