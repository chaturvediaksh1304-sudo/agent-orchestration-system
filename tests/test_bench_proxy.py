"""The proxy is where token counts come from and where a dead run is cut off.

The abort test exists because of a real bug: after a failed run, a model call
that never finished kept generating and the next task queued behind it. Closing
the socket was not enough to stop it; the upstream has to see the disconnect.
"""

import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from bench.proxy import UsageProxy


class _Upstream:
    """A stand-in model server: answers normally, or hangs until hung up on."""

    def __init__(self):
        self.started = threading.Event()
        self.client_gone = threading.Event()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                if b"hang" in body:
                    outer.started.set()
                    # A generation that never ends: block until the caller disconnects.
                    try:
                        while self.connection.recv(1024):
                            pass
                    except OSError:
                        pass
                    outer.client_gone.set()
                    return
                payload = json.dumps(
                    {"choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 7}}
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def upstream():
    server = _Upstream()
    yield server
    server.close()


@pytest.fixture
def proxy(upstream):
    instance = UsageProxy(upstream=upstream.url, port=0)
    instance.start()
    yield instance
    instance.stop()


def _post(proxy, body, timeout=10):
    request = urllib.request.Request(
        f"http://127.0.0.1:{proxy.port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(request, timeout=timeout)


def test_tokens_are_counted_from_the_response(proxy):
    _post(proxy, {"messages": []}).read()
    _post(proxy, {"messages": []}).read()

    assert proxy.snapshot() == {"prompt_tokens": 22, "completion_tokens": 14, "llm_calls": 2}


def test_reset_starts_the_next_task_from_zero(proxy):
    _post(proxy, {"messages": []}).read()

    proxy.reset()

    assert proxy.snapshot() == {"prompt_tokens": 0, "completion_tokens": 0, "llm_calls": 0}


def test_abort_makes_the_upstream_see_the_disconnect(proxy, upstream):
    """Otherwise a runaway generation outlives the run that asked for it."""

    def stuck_call():
        try:
            _post(proxy, {"prompt": "hang"}, timeout=30).read()
        except Exception:
            pass

    threading.Thread(target=stuck_call, daemon=True).start()
    assert upstream.started.wait(5), "the call never reached the upstream"

    proxy.abort(settle_s=5)

    assert upstream.client_gone.wait(5), "the upstream was never told to stop"
    assert proxy.in_flight() == 0


def test_abort_with_nothing_in_flight_is_harmless(proxy):
    proxy.abort(settle_s=1)

    assert proxy.in_flight() == 0
