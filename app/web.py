"""Local web chat (Claude-style prompt box) over the same ChatEngine as the terminal chat.

Run:  uv run python -m app.web [--port 8765]     (or: uv run kb chat-web)
Binds to 127.0.0.1 only; nothing is exposed to the network. Single user, one conversation.
Stdlib only: POST /chat streams server-sent events, POST /reset clears the conversation.
"""
from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PAGE = (Path(__file__).parent / "static" / "chat.html").read_bytes
MAX_BODY = 1 << 20


def make_handler(engine, port: int):
    lock = threading.Lock()  # the engine holds one conversation: serialize requests
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep the terminal quiet
            pass

        def _host_ok(self) -> bool:  # blocks DNS-rebinding from other sites
            if self.headers.get("Host") in allowed_hosts:
                return True
            self.send_error(403, "bad host")
            return False

        def _json_body(self) -> dict | None:
            ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip()
            try:
                n = int(self.headers.get("Content-Length") or 0)
                if ctype != "application/json" or not 0 < n <= MAX_BODY:
                    raise ValueError
                body = json.loads(self.rfile.read(n))
                return body if isinstance(body, dict) else None
            except ValueError:
                self.send_error(400, "expected a JSON object")
                return None

        def do_GET(self):
            if not self._host_ok():
                return
            if self.path not in ("/", "/index.html"):
                return self.send_error(404)
            data = PAGE()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            if not self._host_ok():
                return
            body = self._json_body()
            if body is None:
                return
            if self.path == "/reset":
                with lock:
                    engine.reset()
                self.send_response(204)
                self.end_headers()
            elif self.path == "/chat":
                self._chat(body)
            else:
                self.send_error(404)

        def _chat(self, body: dict):
            message = str(body.get("message") or "").strip()
            if not message:
                return self.send_error(400, "empty message")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            def send(ev: dict):
                self.wfile.write(b"data: " + json.dumps(ev).encode() + b"\n\n")
                self.wfile.flush()

            with lock:
                try:
                    if body.get("mode") in ("kb", "assistant"):
                        engine.set_mode(body["mode"])
                    if "fast" in body:
                        engine.set_fast(bool(body["fast"]))
                    for ev in engine.reply(message, str(body.get("tech") or "") or None):
                        if ev["type"] != "query":
                            send(ev)
                    send({"type": "done"})
                except (BrokenPipeError, ConnectionResetError):
                    pass  # browser closed the tab mid-answer; the turn is not kept
                except Exception as e:  # noqa: BLE001
                    try:
                        send({"type": "error", "text": str(e)})
                    except OSError:
                        pass

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    from query.engine import ChatEngine

    print("loading models (once)...")
    engine = ChatEngine()
    engine.warm_up()
    server = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(engine, a.port))
    print(f"chat ready: http://localhost:{a.port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
