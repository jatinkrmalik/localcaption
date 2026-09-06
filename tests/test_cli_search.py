"""Tests for `localcaption search`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from localcaption.cli import main


def test_top_level_help_mentions_search(capsys) -> None:
    rc = main([])
    assert rc == 2
    assert "search" in capsys.readouterr().out


def test_search_help(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["search", "--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "search term" in out.lower() or "past transcripts" in out.lower()


def test_search_subcommand_prints_ranked_hits(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    index = tmp_path / "index.jsonl"
    monkeypatch.setenv("LOCALCAPTION_INDEX_PATH", str(index))

    txt = tmp_path / "vid123.txt"
    txt.write_text("Welcome\nFirst, let's install\n")
    (tmp_path / "vid123.json").write_text(
        json.dumps(
            {
                "transcription": [
                    {"offsets": {"from": 0, "to": 1000}, "text": "Welcome"},
                    {
                        "offsets": {"from": 150_000, "to": 155_000},
                        "text": "First, let's install",
                    },
                ]
            }
        )
    )
    index.write_text(
        json.dumps(
            {
                "id": "vid123",
                "url": "https://youtu.be/vid123",
                "title": "Test Lecture",
                "duration": 200,
                "chapters": [],
                "transcript": str(txt),
            }
        )
        + "\n"
    )

    rc = main(["search", "install"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "vid123" in out
    assert "02:30" in out
    assert "install" in out.lower()
    assert "Test Lecture" in out


def test_search_no_matches(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setenv("LOCALCAPTION_INDEX_PATH", str(tmp_path / "empty.jsonl"))
    rc = main(["search", "nothing"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "No matches" in out


def test_search_not_treated_as_url(monkeypatch) -> None:
    def fake_transcribe_url(url, **kw):
        raise AssertionError("search must not fall through to transcribe")

    monkeypatch.setattr("localcaption.cli.transcribe_url", fake_transcribe_url)
    with pytest.raises(SystemExit) as excinfo:
        main(["search", "--help"])
    assert excinfo.value.code == 0
