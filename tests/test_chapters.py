"""Tests for chapter extraction and chaptered markdown."""

from __future__ import annotations

import json
from pathlib import Path

from localcaption.chapters import (
    Chapter,
    Segment,
    build_chaptered_markdown,
    chapters_from_info,
    format_timestamp,
    load_segments,
    write_chapters_json,
)

INFO_DICT = {
    "id": "abc",
    "title": "A talk",
    "duration": 600,
    "chapters": [
        {"title": "Intro", "start_time": 0, "end_time": 150},
        {"title": "Setup", "start_time": 150, "end_time": 494},
        {"title": "Demo", "start_time": 494, "end_time": 600},
    ],
}


def test_chapters_from_info_fixture() -> None:
    chapters = chapters_from_info(INFO_DICT)
    assert [c.title for c in chapters] == ["Intro", "Setup", "Demo"]
    assert chapters[0].start_time == 0
    assert chapters[1].start_time == 150
    assert chapters[1].end_time == 494


def test_chapters_from_info_fills_missing_end() -> None:
    chapters = chapters_from_info(
        {
            "chapters": [
                {"title": "A", "start_time": 0},
                {"title": "B", "start_time": 30},
            ]
        }
    )
    assert chapters[0].end_time == 30
    assert chapters[1].end_time is None


def test_chapters_from_info_empty() -> None:
    assert chapters_from_info(None) == []
    assert chapters_from_info({}) == []
    assert chapters_from_info({"chapters": None}) == []
    assert chapters_from_info({"chapters": []}) == []


def test_write_chapters_json(tmp_path: Path) -> None:
    chapters = chapters_from_info(INFO_DICT)
    path = write_chapters_json(tmp_path / "abc.chapters.json", chapters)
    payload = json.loads(path.read_text())
    assert payload[0] == {"title": "Intro", "start_time": 0, "end_time": 150}


def test_format_timestamp() -> None:
    assert format_timestamp(0) == "00:00"
    assert format_timestamp(150) == "02:30"
    assert format_timestamp(494) == "08:14"
    assert format_timestamp(3661) == "1:01:01"


def test_chaptered_markdown_slices_by_start() -> None:
    chapters = [
        Chapter("Intro", 0, 150),
        Chapter("Setup", 150, 494),
        Chapter("Demo", 494, 600),
    ]
    segments = [
        Segment(0.0, 2.0, "Welcome to the show"),
        Segment(160.0, 165.0, "First, let's install"),
        Segment(500.0, 504.0, "Now the demo begins"),
    ]
    md = build_chaptered_markdown(chapters, segments)
    assert md.startswith("## 00:00 Intro\n")
    assert "Welcome to the show" in md.split("## 02:30 Setup")[0]
    assert "First, let's install" in md.split("## 02:30 Setup")[1]
    assert "Now the demo begins" in md.split("## 08:14 Demo")[1]


def test_load_segments_from_whisper_json(tmp_path: Path) -> None:
    json_path = tmp_path / "vid.json"
    json_path.write_text(
        json.dumps(
            {
                "transcription": [
                    {"offsets": {"from": 0, "to": 1500}, "text": " Hello"},
                    {"offsets": {"from": 2000, "to": 3000}, "text": " world"},
                ]
            }
        )
    )
    segs = load_segments(tmp_path / "vid.txt")
    assert len(segs) == 2
    assert segs[0].text == "Hello"
    assert segs[0].start == 0.0
    assert segs[1].start == 2.0


def test_chapters_from_info_skips_bad_start() -> None:
    chapters = chapters_from_info(
        {
            "chapters": [
                {"title": "Bad", "start_time": "nope"},
                {"title": "Good", "start_time": 12},
            ]
        }
    )
    assert [c.title for c in chapters] == ["Good"]
    assert chapters[0].start_time == 12


def test_load_segments_keeps_dotted_stem(tmp_path: Path) -> None:
    json_path = tmp_path / "ep.12.json"
    json_path.write_text(
        json.dumps(
            {
                "transcription": [
                    {"offsets": {"from": 1500, "to": 2000}, "text": "hello"},
                ]
            }
        )
    )
    from_txt = load_segments(tmp_path / "ep.12.txt")
    from_base = load_segments(tmp_path / "ep.12")
    assert [s.text for s in from_txt] == ["hello"]
    assert from_txt[0].start == 1.5
    assert [s.text for s in from_base] == ["hello"]


def test_load_segments_skips_bad_offsets(tmp_path: Path) -> None:
    json_path = tmp_path / "vid.json"
    json_path.write_text(
        json.dumps(
            {
                "transcription": [
                    {"offsets": {"from": "nope"}, "text": "skip me"},
                    {"offsets": {"from": 1000, "to": 2000}, "text": "keep me"},
                ]
            }
        )
    )
    segs = load_segments(tmp_path / "vid.txt")
    assert [s.text for s in segs] == ["keep me"]


def test_load_segments_from_srt_fallback(tmp_path: Path) -> None:
    (tmp_path / "vid.srt").write_text(
        "1\n00:00:00,000 --> 00:00:01,500\nHello\n\n"
        "2\n00:00:02,000 --> 00:00:03,000\nworld\n"
    )
    segs = load_segments(tmp_path / "vid.txt")
    assert [s.text for s in segs] == ["Hello", "world"]
    assert segs[1].start == 2.0
