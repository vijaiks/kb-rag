"""`uv run kb ingest [--all]`, `kb ask "..."`, `kb chat` (terminal), `kb chat-web` (browser)."""
import sys


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("ingest", "ask", "chat", "chat-web"):
        print("usage: kb ingest [--all|--rebuild-fts] | kb ask [--tech NR] [--show-chunks] QUESTION | kb chat | kb chat-web [--port N]")
        return 1
    cmd, sys.argv = sys.argv[1], [sys.argv[0]] + sys.argv[2:]
    if cmd == "ingest":
        from ingest.pipeline import main as m
    elif cmd == "chat":
        from query.chat import main as m
    elif cmd == "chat-web":
        from app.web import main as m
    else:
        from query.ask import main as m
    return m()
