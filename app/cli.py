"""`uv run kb ingest [--all]` and `uv run kb ask "..."`."""
import sys


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("ingest", "ask"):
        print("usage: kb ingest [--all|--rebuild-fts] | kb ask [--tech NR] [--show-chunks] QUESTION")
        return 1
    cmd, sys.argv = sys.argv[1], [sys.argv[0]] + sys.argv[2:]
    if cmd == "ingest":
        from ingest.pipeline import main as m
    else:
        from query.ask import main as m
    return m()
