"""Tests for the JSONL search index and ranking."""

from __future__ import annotations

import json
from pathlib import Path

from localcaption.index import (
    default_index_path,
    search_index,
    upsert_index,
)


def test_index_path_env_override(monkeypatch, tmp_path: Path) -> None:
    custom = tmp_path / "custom" / "idx.jsonl"
    monkeypatch.setenv("LOCALCAPTION_INDEX_PATH", str(custom))
    assert default_index_path() == custom


def test_index_path_xdg_fallback(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("LOCALCAPTION_INDEX_PATH", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert default_index_path() == tmp_path / "xdg" / "localcaption" / "index.jsonl"


def test_upsert_replaces_same_id(tmp_path: Path) -> None:
    path = tmp_path / "index.jsonl"
    upsert_index({"id": "a", "title": "first", "transcript": ""}, path)
    upsert_index({"id": "b", "title": "other", "transcript": ""}, path)
    upsert_index({"id": "a", "title": "updated", "transcript": ""}, path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["id"] for r in rows] == ["b", "a"]
    assert rows[1]["title"] == "updated"


def _seed_video(tmp_path: Path, video_id: str, title: str, segments: list[dict]) -> Path:
    txt = tmp_path / f"{video_id}.txt"
    txt.write_text("\n".join(s["text"] for s in segments) + "\n")
    (tmp_path / f"{video_id}.json").write_text(
        json.dumps(
            {
                "transcription": [
                    {
                        "offsets": {
                            "from": int(s["start"] * 1000),
                            "to": int(s["end"] * 1000),
                        },
                        "text": s["text"],
                    }
                    for s in segments
                ]
            }
        )
    )
    return txt


def test_search_ranked_matches_with_timestamps(tmp_path: Path) -> None:
    path = tmp_path / "index.jsonl"
    t1 = _seed_video(
        tmp_path,
        "aaa",
        "Install guide",
        [
            {"start": 0.0, "end": 2.0, "text": "Welcome"},
            {"start": 150.0, "end": 155.0, "text": "First, let's install pip"},
            {"start": 160.0, "end": 165.0, "text": "Then install ffmpeg too"},
        ],
    )
    t2 = _seed_video(
        tmp_path,
        "bbb",
        "Unrelated talk",
        [
            {"start": 10.0, "end": 12.0, "text": "We install once"},
        ],
    )
    upsert_index(
        {
            "id": "aaa",
            "url": "https://youtu.be/aaa",
            "title": "Install guide",
            "duration": 200,
            "chapters": [{"title": "Setup", "start_time": 150, "end_time": 180}],
            "transcript": str(t1),
        },
        path,
    )
    upsert_index(
        {
            "id": "bbb",
            "url": "https://youtu.be/bbb",
            "title": "Unrelated talk",
            "duration": 30,
            "chapters": [],
            "transcript": str(t2),
        },
        path,
    )

    hits = search_index("install", path)
    assert hits
    # Two transcript hits in aaa (plus chapter title "Setup" does not match).
    # Title "Install guide" adds a bonus, so aaa outranks bbb.
    assert hits[0].id == "aaa"
    ids = [h.id for h in hits]
    assert "bbb" in ids
    timed = [h for h in hits if h.start is not None and "install" in h.text.lower()]
    assert any(abs(h.start - 150.0) < 0.01 for h in timed)
    assert any("ffmpeg" in h.text for h in timed)


def test_search_no_index(tmp_path: Path) -> None:
    assert search_index("anything", tmp_path / "missing.jsonl") == []


def test_upsert_skips_non_dict_rows(tmp_path: Path) -> None:
    path = tmp_path / "index.jsonl"
    path.write_text('[1, 2, 3]\n"just a string"\n{"id": "keep", "title": "ok"}\n')
    upsert_index({"id": "new", "title": "added", "transcript": ""}, path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["id"] for r in rows] == ["keep", "new"]


def test_search_skips_non_dict_rows(tmp_path: Path) -> None:
    path = tmp_path / "index.jsonl"
    txt = tmp_path / "vid.txt"
    txt.write_text("hello world\n")
    path.write_text(
        "[1, 2, 3]\n"
        + json.dumps({"id": "vid", "title": "A talk", "transcript": str(txt)})
        + "\n"
    )
    hits = search_index("hello", path)
    assert len(hits) == 1
    assert hits[0].id == "vid"


def test_search_dotted_transcript_uses_sibling_json(tmp_path: Path) -> None:
    path = tmp_path / "index.jsonl"
    txt = tmp_path / "talk.final.txt"
    txt.write_text("First, let's install\n")
    (tmp_path / "talk.final.json").write_text(
        json.dumps(
            {
                "transcription": [
                    {
                        "offsets": {"from": 150_000, "to": 155_000},
                        "text": "First, let's install",
                    }
                ]
            }
        )
    )
    upsert_index(
        {"id": "talk.final", "title": "A talk", "transcript": str(txt)},
        path,
    )
    hits = search_index("install", path)
    assert len(hits) == 1
    assert hits[0].start == 150.0


def test_search_txt_only_fallback(tmp_path: Path) -> None:
    path = tmp_path / "index.jsonl"
    txt = tmp_path / "vid.txt"
    txt.write_text("no timestamps here but we mention ffmpeg\n")
    upsert_index({"id": "vid", "title": "A talk", "transcript": str(txt)}, path)
    hits = search_index("ffmpeg", path)
    assert len(hits) == 1
    assert hits[0].start is None
    assert "ffmpeg" in hits[0].text


def test_search_chapter_title_hit(tmp_path: Path) -> None:
    path = tmp_path / "index.jsonl"
    txt = tmp_path / "vid.txt"
    txt.write_text("hello\n")
    upsert_index(
        {
            "id": "vid",
            "url": "https://youtu.be/vid",
            "title": "A talk",
            "chapters": [{"title": "Q&A", "start_time": 500}],
            "transcript": str(txt),
        },
        path,
    )
    hits = search_index("Q&A", path)
    assert len(hits) == 1
    assert hits[0].start == 500
    assert hits[0].text.startswith("[chapter]")
