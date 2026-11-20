"""ChatEngine (multi-turn, streaming) and the local web handler, with a fake LLM and retriever."""
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from unittest.mock import MagicMock

from app.web import make_handler
from llm.client import LLMClient
from query.engine import ASSISTANT_SYSTEM, NOT_FOUND, ChatEngine

CHUNK = {"file_path": "a.pdf", "page_start": 1, "page_end": 2, "section_path": "S", "text": "txt"}


class FakeLLM:
    def __init__(self):
        self.streams, self.completes = [], []

    def complete(self, system, user, max_tokens=None):
        self.completes.append(user)
        return "standalone query"

    def stream(self, system, user, max_tokens=None):
        self.streams.append((system, user, max_tokens))
        yield from ["Hel", "lo [S1]"]


def make_engine(chunks=(CHUNK,), **kw):
    cfg = {"retrieval": {"reranker": "rr"}}
    retr = MagicMock(return_value=list(chunks))
    return ChatEngine(cfg=cfg, llm=FakeLLM(), tbl=object(), retriever=retr, **kw), retr


def run(eng, q, tech=None):
    return list(eng.reply(q, tech))


def test_first_turn_streams_and_skips_condense():
    eng, retr = make_engine()
    events = run(eng, "what is RACH?")
    assert "".join(e["text"] for e in events if e["type"] == "token") == "Hello [S1]"
    assert eng.llm.completes == []
    assert retr.call_args.args[2] == "what is RACH?"
    assert next(e for e in events if e["type"] == "sources")["sources"][0]["tag"] == "S1"
    assert [m["role"] for m in eng.history] == ["user", "assistant"]


def test_followup_is_condensed_and_history_has_no_chunks():
    eng, retr = make_engine()
    run(eng, "what is RACH?")
    run(eng, "and in LTE?")
    assert retr.call_args.args[2] == "standalone query"
    _, msgs, _ = eng.llm.streams[1]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert msgs[0]["content"] == "what is RACH?"          # plain text, not the sourced prompt
    assert "Sources:" in msgs[2]["content"] and "and in LTE?" in msgs[2]["content"]


def test_history_trimmed():
    eng, _ = make_engine(max_turns=2)
    for i in range(5):
        run(eng, f"q{i}")
    assert len(eng.history) == 4 and eng.history[0]["content"] == "q3"


def test_kb_mode_without_chunks_is_refused_without_llm_call():
    eng, _ = make_engine(chunks=())
    assert "".join(e["text"] for e in run(eng, "x") if e["type"] == "token") == NOT_FOUND
    assert eng.llm.streams == []


def test_assistant_mode_without_chunks_still_answers():
    eng, _ = make_engine(chunks=())
    eng.set_mode("assistant")
    run(eng, "x")
    system, msgs, _ = eng.llm.streams[0]
    assert system.startswith(ASSISTANT_SYSTEM) and msgs[-1]["content"] == "x"


def test_fast_toggles_reranker_and_tokens_and_reset():
    eng, _ = make_engine()
    eng.set_fast(True)
    assert eng.cfg["retrieval"]["reranker"] is None
    run(eng, "q")
    assert eng.llm.streams[0][2] == 700
    eng.set_fast(False)
    assert eng.cfg["retrieval"]["reranker"] == "rr"
    eng.reset()
    assert eng.history == []


def test_abandoned_stream_is_not_remembered():
    eng, _ = make_engine()
    gen = eng.reply("q")
    for ev in gen:
        if ev["type"] == "token":
            break
    gen.close()
    assert eng.history == []


def test_llm_stream_passes_messages(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    c = LLMClient({"llm": {"provider": "anthropic", "model": "m", "max_tokens": 10}})
    c.client = MagicMock()
    cm = c.client.messages.stream.return_value.__enter__.return_value
    cm.text_stream = iter(["a", "b"])
    msgs = [{"role": "user", "content": "hi"}]
    assert list(c.stream("sys", msgs)) == ["a", "b"]
    assert c.client.messages.stream.call_args.kwargs["messages"] == msgs


def _post(port, path, body, host=None, ctype="application/json"):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(),
                                 {"Content-Type": ctype, **({"Host": host} if host else {})})
    return urllib.request.urlopen(req)


def test_web_chat_stream_and_guards():
    eng, _ = make_engine()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), None)
    port = srv.server_address[1]
    srv.RequestHandlerClass = make_handler(eng, port)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        out = _post(port, "/chat", {"message": "hi", "mode": "assistant"}).read().decode()
        events = [json.loads(x[6:]) for x in out.split("\n\n") if x]
        assert events[-1] == {"type": "done"}
        assert "".join(e["text"] for e in events if e["type"] == "token") == "Hello [S1]"
        assert eng.mode == "assistant"
        assert _post(port, "/reset", {}).status == 204 and eng.history == []
        for kw in ({"host": "evil.com"}, {"ctype": "text/plain"}):
            try:
                _post(port, "/chat", {"message": "hi"}, **kw)
                raise AssertionError("expected rejection")
            except urllib.error.HTTPError as e:
                assert e.code in (400, 403)
        assert b"<textarea" in urllib.request.urlopen(f"http://127.0.0.1:{port}/").read()
    finally:
        srv.shutdown()
