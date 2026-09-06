"""Optional post-transcript summary via a local Ollama instance.

After whisper.cpp writes ``<id>.txt``, this module POSTs the transcript to
``http://localhost:11434/api/generate`` and writes ``<id>.summary.md`` next
to it. Failures (Ollama down, bad model, timeout) are warnings, not errors:
the transcript pipeline still succeeds.

HTTP is stdlib-only (``urllib.request``). No httpx/requests.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from importlib.resources import files
from pathlib import Path

from . import _logging as log

DEFAULT_MODEL = "llama3.1:8b"
DEFAULT_ENDPOINT = "http://localhost:11434/api/generate"
DEFAULT_TIMEOUT = 120.0

# Ignore HTTP(S)_PROXY. Ollama is local; a proxy would either fail the
# request or send the transcript off-machine.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def load_prompt(path: Path | None = None) -> str:
    """Return the prompt template from *path*, or the built-in default."""
    if path is not None:
        return Path(path).read_text(encoding="utf-8")
    return files("localcaption").joinpath("summary_prompt.txt").read_text(encoding="utf-8")


def _build_prompt(template: str, transcript: str) -> str:
    if "{transcript}" in template:
        return template.replace("{transcript}", transcript)
    return f"{template.rstrip()}\n\n---\n\n{transcript}"


def generate(
    transcript: str,
    *,
    model: str = DEFAULT_MODEL,
    prompt: str | None = None,
    endpoint: str = DEFAULT_ENDPOINT,
    timeout: float = DEFAULT_TIMEOUT,
) -> str | None:
    """POST *transcript* to Ollama. Return the summary text, or None on failure."""
    try:
        template = prompt if prompt is not None else load_prompt()
        body = json.dumps(
            {
                "model": model,
                "prompt": _build_prompt(template, transcript),
                "stream": False,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=body,
            headers={"Content-Type": "application/json", "User-Agent": "localcaption"},
            method="POST",
        )
        with _OPENER.open(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = f"HTTP {exc.code}"
        try:
            err_body = exc.read().decode("utf-8", errors="replace").strip()
            if err_body:
                detail = f"HTTP {exc.code}: {err_body[:200]}"
        except OSError:
            pass
        log.warn(f"Ollama request failed ({detail}). Summary skipped.")
        return None
    except urllib.error.URLError as exc:
        reason = exc.reason
        if isinstance(reason, TimeoutError):
            log.warn(f"Ollama request timed out after {timeout:.0f}s. Summary skipped.")
            return None
        log.warn(
            f"Ollama is not reachable at {endpoint} ({reason}). "
            "Is Ollama running? Summary skipped."
        )
        return None
    except TimeoutError:
        log.warn(f"Ollama request timed out after {timeout:.0f}s. Summary skipped.")
        return None
    except Exception as exc:
        log.warn(f"Ollama summary failed ({exc}). Summary skipped.")
        return None

    if not isinstance(payload, dict):
        log.warn("Ollama returned a non-object JSON body. Summary skipped.")
        return None

    error = payload.get("error")
    raw = payload.get("response")
    text = raw.strip() if isinstance(raw, str) else ""
    if error and not text:
        log.warn(f"Ollama error: {error}. Summary skipped.")
        return None
    if not text:
        log.warn("Ollama returned an empty summary. Summary skipped.")
        return None
    return text


def write_summary(
    txt_path: Path,
    *,
    model: str = DEFAULT_MODEL,
    prompt_path: Path | None = None,
    endpoint: str = DEFAULT_ENDPOINT,
    timeout: float = DEFAULT_TIMEOUT,
) -> Path | None:
    """Read ``<id>.txt``, write ``<id>.summary.md`` next to it.

    Returns the summary path on success, or None if anything went wrong.
    Never raises: callers can treat summarization as best-effort.
    """
    txt_path = Path(txt_path)
    if not txt_path.is_file():
        log.warn(f"transcript not found at {txt_path}; skipping summary")
        return None

    try:
        transcript = txt_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        log.warn(f"could not read transcript at {txt_path}: {exc}. Summary skipped.")
        return None
    if not transcript.strip():
        log.warn("transcript is empty; skipping summary")
        return None

    try:
        prompt = load_prompt(prompt_path)
    except OSError as exc:
        log.warn(f"could not read summary prompt: {exc}. Summary skipped.")
        return None

    log.info(f"ollama: model={model}")
    text = generate(
        transcript,
        model=model,
        prompt=prompt,
        endpoint=endpoint,
        timeout=timeout,
    )
    if text is None:
        return None

    out = txt_path.with_name(f"{txt_path.stem}.summary.md")
    try:
        out.write_text(text + "\n", encoding="utf-8")
    except OSError as exc:
        log.warn(f"could not write summary to {out}: {exc}. Summary skipped.")
        return None
    log.info(f"summary: {out}")
    return out
