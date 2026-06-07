"""Integration test for `models.download_model` against a local HTTP server.

We don't want CI to hit the real Hugging Face servers (flaky, slow, and
rude), but we DO want to exercise the actual urllib + progress + atomic-rename
code path. So we spin up a tiny HTTPServer in a thread, monkeypatch the
registry's HF_BASE_URL to point at it, and run the real downloader.
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from localcaption import models
from localcaption.errors import DependencyError

# ──────────────────────────────────────────────────────────────────────
# Tiny HTTP server fixture
# ──────────────────────────────────────────────────────────────────────


FAKE_PAYLOAD = b"fake-ggml-data-" * 1000  # ~15 KB; enough to test progress + I/O


class _FakeWhisperHandler(BaseHTTPRequestHandler):
    """Serves any /ggml-*.bin path with the same fake payload."""

    truncate_after: int | None = None  # set per-test to simulate truncation
    fail_with_status: int | None = None  # set per-test to simulate HTTP error

    def do_GET(self):  # noqa: N802 (BaseHTTPRequestHandler convention)
        if self.fail_with_status is not None:
            self.send_response(self.fail_with_status)
            self.end_headers()
            return

        if not self.path.endswith(".bin"):
            self.send_response(404)
            self.end_headers()
            return

        self.send_response(200)
        body = FAKE_PAYLOAD
        if self.truncate_after is not None:
            # Lie about Content-Length to simulate a truncated download.
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body[: self.truncate_after])
        else:
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def log_message(self, *args, **kwargs):  # silence test noise
        pass


@pytest.fixture
def fake_hf_server(monkeypatch):
    """Spin up an HTTPServer that mimics huggingface.co for the duration of one test."""
    # Reset class-level mutable defaults
    _FakeWhisperHandler.truncate_after = None
    _FakeWhisperHandler.fail_with_status = None

    server = HTTPServer(("127.0.0.1", 0), _FakeWhisperHandler)
    port = server.server_address[1]

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    # Redirect the registry's URL base to the fake server.
    monkeypatch.setattr(models, "HF_BASE_URL", f"http://127.0.0.1:{port}")

    yield _FakeWhisperHandler

    server.shutdown()
    server.server_close()


# ──────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────


def _empty_whisper_tree(root: Path) -> Path:
    whisper = root / "whisper.cpp"
    (whisper / "models").mkdir(parents=True)
    return whisper


def test_download_writes_file_atomically(tmp_path, fake_hf_server):
    whisper = _empty_whisper_tree(tmp_path)

    result = models.download_model("base.en", whisper)

    assert result.is_file()
    assert result.read_bytes() == FAKE_PAYLOAD
    # Atomic-rename invariant: no .part file left lying around
    assert not result.with_suffix(".bin.part").exists()


def test_download_progress_callback_is_invoked(tmp_path, fake_hf_server):
    whisper = _empty_whisper_tree(tmp_path)
    calls: list[tuple[int, int]] = []

    models.download_model(
        "base.en", whisper, on_progress=lambda d, t: calls.append((d, t))
    )

    assert calls, "progress callback was never called"
    final_downloaded, final_total = calls[-1]
    assert final_downloaded == final_total == len(FAKE_PAYLOAD)


def test_download_skips_when_already_present(tmp_path, fake_hf_server):
    whisper = _empty_whisper_tree(tmp_path)
    target = models.model_path(whisper, "base.en")
    target.write_bytes(b"existing-content")

    # Should be a no-op (no overwrite, no exception)
    result = models.download_model("base.en", whisper)
    assert result.read_bytes() == b"existing-content"


def test_download_force_overwrites(tmp_path, fake_hf_server):
    whisper = _empty_whisper_tree(tmp_path)
    target = models.model_path(whisper, "base.en")
    target.write_bytes(b"stale")

    models.download_model("base.en", whisper, force=True)
    assert target.read_bytes() == FAKE_PAYLOAD


def test_download_truncated_response_raises(tmp_path, fake_hf_server):
    _FakeWhisperHandler = fake_hf_server  # alias for clarity
    _FakeWhisperHandler.truncate_after = 100  # send only 100 bytes despite 15 KB header

    whisper = _empty_whisper_tree(tmp_path)
    with pytest.raises(DependencyError) as exc_info:
        models.download_model("base.en", whisper)
    assert "Truncated" in str(exc_info.value)
    # The .part file must have been cleaned up
    assert not models.model_path(whisper, "base.en").exists()
    assert not models.model_path(whisper, "base.en").with_suffix(".bin.part").exists()


def test_download_http_error_is_friendly(tmp_path, fake_hf_server):
    fake_hf_server.fail_with_status = 503

    whisper = _empty_whisper_tree(tmp_path)
    with pytest.raises(DependencyError) as exc_info:
        models.download_model("base.en", whisper)
    msg = str(exc_info.value)
    assert "Download failed" in msg
    assert "503" in msg
    # No leftover artefacts on disk
    assert not models.model_path(whisper, "base.en").exists()
