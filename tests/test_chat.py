"""query/chat.py: ':' commands and the REPL loop, driven by a fake engine and scripted input."""
import io

import pytest
from rich.console import Console

from query import chat


class FakeEngine:
    def __init__(self, events=None, fail=None):
        self.mode, self.fast, self.resets, self.calls = "kb", False, 0, []
        self.events = events if events is not None else [
            {"type": "sources", "sources": [{"tag": "S1", "file": "a.pdf", "page_start": 1,
                                             "page_end": 2, "section": "Intro"}]},
            {"type": "token", "text": "Hello "}, {"type": "token", "text": "world [S1]"},
        ]
        self.fail = fail

    def warm_up(self):
        pass

    def set_mode(self, m):
        self.mode = m

    def set_fast(self, f):
        self.fast = f

    def reset(self):
        self.resets += 1

    def reply(self, q, tech):
        self.calls.append((q, tech))
        if self.fail:
            raise self.fail
        yield from self.events


@pytest.fixture
def out(monkeypatch):
    buf = io.StringIO()
    monkeypatch.setattr(chat, "console", Console(file=buf, width=120, force_terminal=False))
    return buf


def run_session(monkeypatch, eng, lines):
    monkeypatch.setattr(chat, "ChatEngine", lambda: eng)
    it = iter(lines)

    def fake_input(prompt=""):
        try:
            return next(it)
        except StopIteration:
            raise EOFError

    monkeypatch.setattr("builtins.input", fake_input)
    return chat.main()


# ---- handle_command -----------------------------------------------------------
def test_commands(out):
    eng, state = FakeEngine(), {"tech": None, "chunks": False}
    assert chat.handle_command(":tech NR", eng, state) and state["tech"] == "NR"
    chat.handle_command(":tech off", eng, state)
    assert state["tech"] is None
    chat.handle_command(":tech", eng, state)
    assert state["tech"] is None
    chat.handle_command(":mode assistant", eng, state)
    assert eng.mode == "assistant"
    chat.handle_command(":mode bogus", eng, state)
    assert eng.mode == "assistant"                       # invalid value is ignored
    chat.handle_command(":new", eng, state)
    assert eng.resets == 1
    chat.handle_command(":chunks", eng, state)
    assert state["chunks"] is True
    chat.handle_command(":fast", eng, state)
    assert eng.fast is True
    chat.handle_command(":fast", eng, state)
    assert eng.fast is False
    assert chat.handle_command(":nope", eng, state) is False


# ---- main loop ----------------------------------------------------------------
def test_question_is_streamed_with_sources(monkeypatch, out):
    eng = FakeEngine()
    assert run_session(monkeypatch, eng, ["what is X?", ":quit"]) == 0
    text = out.getvalue()
    assert eng.calls == [("what is X?", None)]
    assert "Hello world [S1]" in text and "[S1] a.pdf p.1-2" in text


def test_tech_filter_is_passed_and_commands_do_not_query(monkeypatch, out):
    eng = FakeEngine()
    run_session(monkeypatch, eng, [":tech NR", "", ":mode assistant", ":bogus", "q", "exit"])
    assert eng.calls == [("q", "NR")]
    assert "unknown command" in out.getvalue()


def test_show_chunks_prints_retrieved_sources(monkeypatch, out):
    run_session(monkeypatch, FakeEngine(), [":chunks", "q", ":q"])
    assert "[S1] a.pdf p.1 Intro" in out.getvalue()


def test_engine_error_keeps_session_alive(monkeypatch, out):
    eng = FakeEngine(fail=RuntimeError("api down"))
    run_session(monkeypatch, eng, ["q1", "q2"])
    assert len(eng.calls) == 2 and "error: api down" in out.getvalue()


def test_ctrl_c_during_answer_is_not_fatal(monkeypatch, out):
    eng = FakeEngine(fail=KeyboardInterrupt())
    run_session(monkeypatch, eng, ["q1", "q2"])
    assert len(eng.calls) == 2 and "interrupted" in out.getvalue()


def test_eof_and_ctrl_c_at_prompt_exit_cleanly(monkeypatch, out):
    assert run_session(monkeypatch, FakeEngine(), []) == 0
    monkeypatch.setattr("builtins.input", lambda p="": (_ for _ in ()).throw(KeyboardInterrupt))
    assert chat.main() == 0
