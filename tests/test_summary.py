"""Tests for the optional Ollama summary client.

CI must not talk to a real Ollama, so we spin up a tiny HTTPServer in a
thread (same pattern as tests/test_models_download.py) and point the
client at it. Unreachable-host cases use a closed local port.
"""

from __future__ import annotations

import json
import socket
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from localcaption import summary


class _FakeOllamaHandler(BaseHTTPRequestHandler):
    """Serves POST /api/generate with a canned Ollama-style JSON body."""

    response_text = "## TL;DR\n\nIt was about widgets.\n"
    status = 200
    raw_body: bytes | None = None  # send this instead of wrapping response_text
    last_body: bytes | None = None
    last_path: str | None = None
    incomplete = False  # lie about Content-Length so the client sees a short body

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        _FakeOllamaHandler.last_body = self.rfile.read(length)
        _FakeOllamaHandler.last_path = self.path

        if self.status != 200:
            err = b'{"error": "nope"}'
            self.send_response(self.status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(err)))
            self.end_headers()
            self.wfile.write(err)
            return

        if self.raw_body is not None:
            body = self.raw_body
        else:
            body = json.dumps(
                {
                    "model": "llama3.1:8b",
                    "response": self.response_text,
                    "done": True,
                }
            ).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        claimed = len(body) + 1000 if self.incomplete else len(body)
        self.send_header("Content-Length", str(claimed))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args, **kwargs):  # silence test noise
        pass


@pytest.fixture
def fake_ollama():
    """HTTPServer mimicking Ollama's /api/generate for one test."""
    _FakeOllamaHandler.response_text = "## TL;DR\n\nIt was about widgets.\n"
    _FakeOllamaHandler.status = 200
    _FakeOllamaHandler.raw_body = None
    _FakeOllamaHandler.last_body = None
    _FakeOllamaHandler.last_path = None
    _FakeOllamaHandler.incomplete = False

    server = HTTPServer(("127.0.0.1", 0), _FakeOllamaHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    endpoint = f"http://127.0.0.1:{port}/api/generate"
    yield endpoint, _FakeOllamaHandler

    server.shutdown()
    server.server_close()


def _closed_endpoint() -> str:
    """An http URL whose port is not listening."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return f"http://127.0.0.1:{port}/api/generate"


# ──────────────────────────────────────────────────────────────────────
# Prompt
# ──────────────────────────────────────────────────────────────────────


def test_default_prompt_has_expected_sections() -> None:
    text = summary.load_prompt()
    lower = text.lower()
    assert "tl;dr" in lower
    assert "key points" in lower
    assert "notable quotes" in lower
    assert "action items" in lower
    assert "{transcript}" in text


def test_load_prompt_from_path(tmp_path: Path) -> None:
    path = tmp_path / "custom.txt"
    path.write_text("Custom: {transcript}\n", encoding="utf-8")
    assert summary.load_prompt(path) == "Custom: {transcript}\n"


def test_build_prompt_substitutes_placeholder() -> None:
    built = summary._build_prompt("Go:\n{transcript}\n", "hello")
    assert built == "Go:\nhello\n"


def test_build_prompt_appends_when_no_placeholder() -> None:
    built = summary._build_prompt("Summarize this.", "hello world")
    assert "Summarize this." in built
    assert "hello world" in built


# ──────────────────────────────────────────────────────────────────────
# HTTP client
# ──────────────────────────────────────────────────────────────────────


def test_generate_posts_to_ollama(fake_ollama) -> None:
    endpoint, handler = fake_ollama
    text = summary.generate(
        "the widgets lecture",
        model="llama3.1:8b",
        prompt="Sum up:\n{transcript}",
        endpoint=endpoint,
        timeout=5,
    )
    assert text is not None
    assert "widgets" in text
    assert handler.last_path == "/api/generate"

    sent = json.loads(handler.last_body)
    assert sent["model"] == "llama3.1:8b"
    assert sent["stream"] is False
    assert "the widgets lecture" in sent["prompt"]
    assert sent["prompt"].startswith("Sum up:")


def test_write_summary_creates_markdown(tmp_path: Path, fake_ollama) -> None:
    endpoint, _handler = fake_ollama
    txt = tmp_path / "talk.txt"
    txt.write_text("We shipped the feature.\n", encoding="utf-8")

    out = summary.write_summary(
        txt, model="mistral", endpoint=endpoint, timeout=5
    )
    assert out == tmp_path / "talk.summary.md"
    assert out.is_file()
    assert "widgets" in out.read_text(encoding="utf-8")


def test_write_summary_uses_custom_prompt(tmp_path: Path, fake_ollama) -> None:
    endpoint, handler = fake_ollama
    txt = tmp_path / "talk.txt"
    txt.write_text("alpha beta", encoding="utf-8")
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("ONLY THIS: {transcript}", encoding="utf-8")

    summary.write_summary(
        txt, prompt_path=prompt, endpoint=endpoint, timeout=5
    )
    sent = json.loads(handler.last_body)
    assert sent["prompt"] == "ONLY THIS: alpha beta"


def test_unreachable_ollama_returns_none_and_warns(capsys) -> None:
    result = summary.generate(
        "hello",
        endpoint=_closed_endpoint(),
        timeout=0.5,
    )
    assert result is None
    err = capsys.readouterr().err
    assert "not reachable" in err
    assert "Summary skipped" in err


def test_http_error_returns_none_and_warns(fake_ollama, capsys) -> None:
    endpoint, handler = fake_ollama
    handler.status = 404
    result = summary.generate("hello", endpoint=endpoint, timeout=5)
    assert result is None
    err = capsys.readouterr().err
    assert "HTTP 404" in err
    assert "Summary skipped" in err


def test_empty_response_returns_none(fake_ollama, capsys) -> None:
    endpoint, handler = fake_ollama
    handler.response_text = "   "
    result = summary.generate("hello", endpoint=endpoint, timeout=5)
    assert result is None
    assert "empty summary" in capsys.readouterr().err


def test_ollama_error_field_returns_none(fake_ollama, capsys) -> None:
    endpoint, handler = fake_ollama
    handler.raw_body = json.dumps({"error": "model not found"}).encode()
    result = summary.generate("hello", endpoint=endpoint, timeout=5)
    assert result is None
    assert "model not found" in capsys.readouterr().err


def test_invalid_json_returns_none(fake_ollama, capsys) -> None:
    endpoint, handler = fake_ollama
    handler.raw_body = b"not-json"
    result = summary.generate("hello", endpoint=endpoint, timeout=5)
    assert result is None
    assert "Summary skipped" in capsys.readouterr().err


def test_incomplete_http_body_returns_none(fake_ollama, capsys) -> None:
    endpoint, handler = fake_ollama
    handler.incomplete = True
    result = summary.generate("hello", endpoint=endpoint, timeout=5)
    assert result is None
    err = capsys.readouterr().err
    assert "Summary skipped" in err


def test_non_string_response_returns_none(fake_ollama, capsys) -> None:
    endpoint, handler = fake_ollama
    handler.raw_body = json.dumps({"response": {"nested": True}}).encode()
    result = summary.generate("hello", endpoint=endpoint, timeout=5)
    assert result is None
    assert "empty summary" in capsys.readouterr().err


def test_generate_ignores_http_proxy(fake_ollama, monkeypatch) -> None:
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:1")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    urllib.request._opener = None
    endpoint, _handler = fake_ollama
    text = summary.generate("hello", endpoint=endpoint, timeout=5)
    assert text is not None
    assert "widgets" in text


def test_urlerror_timeout_says_timed_out(monkeypatch, capsys) -> None:
    def boom(*args, **kwargs):
        raise urllib.error.URLError(TimeoutError("timed out"))

    monkeypatch.setattr(summary._OPENER, "open", boom)
    result = summary.generate("hello", timeout=1)
    assert result is None
    err = capsys.readouterr().err
    assert "timed out" in err
    assert "not reachable" not in err


def test_missing_transcript_skips(tmp_path: Path, capsys) -> None:
    result = summary.write_summary(tmp_path / "missing.txt")
    assert result is None
    assert "transcript not found" in capsys.readouterr().err


def test_empty_transcript_skips(tmp_path: Path, capsys) -> None:
    txt = tmp_path / "empty.txt"
    txt.write_text("   \n", encoding="utf-8")
    result = summary.write_summary(txt)
    assert result is None
    assert "empty" in capsys.readouterr().err


def test_missing_prompt_file_skips(tmp_path: Path, capsys) -> None:
    txt = tmp_path / "talk.txt"
    txt.write_text("hello", encoding="utf-8")
    result = summary.write_summary(
        txt, prompt_path=tmp_path / "no-such-prompt.txt"
    )
    assert result is None
    assert "could not read summary prompt" in capsys.readouterr().err


def test_write_summary_incomplete_body_skips(
    tmp_path: Path, fake_ollama, capsys
) -> None:
    endpoint, handler = fake_ollama
    handler.incomplete = True
    txt = tmp_path / "talk.txt"
    txt.write_text("hello", encoding="utf-8")
    result = summary.write_summary(txt, endpoint=endpoint, timeout=5)
    assert result is None
    assert not (tmp_path / "talk.summary.md").exists()
    assert "Summary skipped" in capsys.readouterr().err
