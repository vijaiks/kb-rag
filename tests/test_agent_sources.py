"""The agent's [S#] tags must stay stable across searches and the critic must see all sources."""
import json

import pytest

from agents import loop


def chunk(cid: str, path: str) -> dict:
    return {
        "chunk_id": cid,
        "file_path": path,
        "page_start": 1,
        "page_end": 1,
        "section_path": "",
        "text": f"text-{cid}",
    }


@pytest.fixture
def session(monkeypatch):
    monkeypatch.setattr(loop, "load_config", dict)
    monkeypatch.setattr(loop, "chunks_table", lambda cfg: None)
    monkeypatch.setattr(loop, "LLMClient", lambda cfg: None)
    return loop.Session()


def test_tags_are_stable_across_searches(session, monkeypatch):
    batches = iter([[chunk("a", "A.pdf"), chunk("b", "B.pdf")], [chunk("c", "C.pdf"), chunk("a", "A.pdf")]])
    monkeypatch.setattr(loop, "retrieve", lambda *args, **kwargs: next(batches))

    first = session.search_kb("q1")
    second = session.search_kb("q2")

    assert first.index("[S1] A.pdf") < first.index("[S2] B.pdf")
    assert "[S3] C.pdf" in second
    assert "[S1] A.pdf" in second  # re-retrieved chunk keeps its original number
    assert [c["chunk_id"] for c in session.sources] == ["a", "b", "c"]


def test_verify_requires_sources(session):
    assert json.loads(session.verify_answer("anything"))["verdict"] == "rework"


def test_verify_sends_all_sources_to_critic(session, monkeypatch):
    batches = iter([[chunk("a", "A.pdf")], [chunk("b", "B.pdf")]])
    monkeypatch.setattr(loop, "retrieve", lambda *args, **kwargs: next(batches))
    session.search_kb("q1")
    session.search_kb("q2")

    seen = {}

    class FakeLLM:
        def complete(self, system, user, max_tokens=None):
            seen["user"] = user
            return '{"verdict": "ack"}'

    session.llm = FakeLLM()
    session.verify_answer("claim [S1] [S2]")
    assert "[S1] A.pdf" in seen["user"] and "[S2] B.pdf" in seen["user"]
