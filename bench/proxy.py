"""A counting pass-through in front of Ollama.

The system records no token usage, and in the full arm most model calls happen
inside Celery worker processes the driver cannot see into. Pointing both arms
at this proxy lets the harness read prompt and completion tokens off every
response without touching the system. Tasks run one at a time, so everything
counted between reset() and snapshot() belongs to one task.
"""

import http.client
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict
from urllib.parse import urlparse

from bench.config import OLLAMA_URL, PROXY_PORT, TASK_TIMEOUT_S

_SKIP_HEADERS = {"host", "content-length", "connection", "accept-encoding", "transfer-encoding"}


class UsageProxy:
    def __init__(self, upstream: str = OLLAMA_URL, port: int = PROXY_PORT):
        target = urlparse(upstream)
        self._host, self._upstream_port = target.hostname, target.port or 80
        self._lock = threading.Lock()
        self._counts = {"prompt_tokens": 0, "completion_tokens": 0, "llm_calls": 0}
        self._open = set()
        self._server = ThreadingHTTPServer(("127.0.0.1", port), self._handler())
        self._server.daemon_threads = True
        self.port = self._server.server_address[1]

    def start(self) -> None:
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def reset(self) -> None:
        with self._lock:
            self._counts = {key: 0 for key in self._counts}

    def snapshot(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._counts)

    def abort(self, settle_s: float = 30.0) -> None:
        """Drop every in-flight upstream request.

        After a task is killed, its model calls would otherwise keep generating
        and steal time and tokens from the next task.
        """
        with self._lock:
            connections = list(self._open)
        for connection in connections:
            try:
                # shutdown(), not close(): closing a socket another thread is
                # blocked reading neither wakes that thread nor tells the
                # server, so the generation would carry on. shutdown() does both.
                if connection.sock:
                    connection.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        deadline = time.monotonic() + settle_s
        while time.monotonic() < deadline:
            with self._lock:
                if not self._open:
                    return
            time.sleep(0.2)

    def in_flight(self) -> int:
        with self._lock:
            return len(self._open)

    def _handler(self):
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # keep the benchmark output readable
                pass

            def _forward(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else None
                headers = {k: v for k, v in self.headers.items() if k.lower() not in _SKIP_HEADERS}
                upstream = http.client.HTTPConnection(
                    proxy._host, proxy._upstream_port, timeout=TASK_TIMEOUT_S + 60
                )
                with proxy._lock:
                    proxy._open.add(upstream)
                try:
                    upstream.request(self.command, self.path, body=body, headers=headers)
                    response = upstream.getresponse()
                    status, payload = response.status, response.read()
                    content_type = response.getheader("Content-Type", "application/json")
                except Exception as exc:  # upstream died or was aborted
                    status, content_type = 502, "application/json"
                    payload = json.dumps(
                        {"error": {"message": f"proxy: upstream request failed: {exc}"}}
                    ).encode()
                finally:
                    with proxy._lock:
                        proxy._open.discard(upstream)
                    upstream.close()

                if status == 200 and self.path.endswith("/chat/completions"):
                    proxy._tally(payload)
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # the caller was killed; nothing left to tell it

            do_GET = do_POST = _forward

        return Handler

    def _tally(self, payload: bytes) -> None:
        try:
            usage = json.loads(payload).get("usage") or {}
        except ValueError:
            usage = {}
        with self._lock:
            self._counts["llm_calls"] += 1
            self._counts["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
            self._counts["completion_tokens"] += int(usage.get("completion_tokens") or 0)
