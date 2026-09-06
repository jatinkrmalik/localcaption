"""Tests for batch transcription: parse, resume, mixed outcomes."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from localcaption.batch import (
    BatchItem,
    BatchResult,
    read_url_list,
    transcribe_urls,
    video_id_for,
)
from localcaption.errors import DownloadError
from localcaption.pipeline import PipelineResult
from localcaption.whisper import BACKEND_FASTER_WHISPER, TranscriptionResult


def _result(url: str, duration_s: float | None = 1120.0) -> PipelineResult:
    fake = MagicMock(spec=TranscriptionResult)
    fake.existing.return_value = {}
    return PipelineResult(
        source_url=url,
        audio_path=None,
        wav_path=None,
        transcripts=fake,
        duration_s=duration_s,
    )


class TestReadUrlList:
    def test_skips_empty_and_comment_lines(self, tmp_path: Path) -> None:
        listing = tmp_path / "urls.txt"
        listing.write_text(
            "# podcast queue\n"
            "\n"
            "https://www.youtube.com/watch?v=aircAruvnKk\n"
            "  \n"
            "# ignore this\n"
            "https://youtu.be/PSRJfaAYkW4\n"
            "/tmp/talk.mp4\n",
            encoding="utf-8",
        )
        assert read_url_list(listing) == [
            "https://www.youtube.com/watch?v=aircAruvnKk",
            "https://youtu.be/PSRJfaAYkW4",
            "/tmp/talk.mp4",
        ]

    def test_empty_file(self, tmp_path: Path) -> None:
        listing = tmp_path / "empty.txt"
        listing.write_text("", encoding="utf-8")
        assert read_url_list(listing) == []

    def test_relative_local_paths_are_vs_list_file(self, tmp_path: Path) -> None:
        media = tmp_path / "podcasts"
        media.mkdir()
        listing = media / "urls.txt"
        listing.write_text("ep.mp3\n./also.wav\n", encoding="utf-8")
        assert read_url_list(listing) == [str(media / "ep.mp3"), str(media / "also.wav")]

    def test_expands_user_in_local_paths(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        listing = tmp_path / "urls.txt"
        listing.write_text("~/talk.mp4\n", encoding="utf-8")
        assert read_url_list(listing) == [str(tmp_path / "talk.mp4")]


class TestVideoIdFor:
    def test_youtube_watch_url(self) -> None:
        assert video_id_for("https://www.youtube.com/watch?v=aircAruvnKk") == "aircAruvnKk"

    def test_youtu_be(self) -> None:
        assert video_id_for("https://youtu.be/PSRJfaAYkW4") == "PSRJfaAYkW4"

    def test_youtube_shorts_and_extra_query(self) -> None:
        assert video_id_for("https://www.youtube.com/shorts/aircAruvnKk") == "aircAruvnKk"
        assert (
            video_id_for("https://www.youtube.com/watch?v=aircAruvnKk&t=30")
            == "aircAruvnKk"
        )

    def test_local_files_same_stem_different_dirs(self, tmp_path: Path) -> None:
        a = tmp_path / "show1" / "audio.mp3"
        b = tmp_path / "show2" / "audio.mp3"
        assert video_id_for(str(a)) != video_id_for(str(b))
        assert "audio" in video_id_for(str(a))

    def test_other_site_namespaces_host(self) -> None:
        assert video_id_for("https://vimeo.com/148751763") == "vimeo.com_148751763"

    def test_url_path_not_just_last_segment(self) -> None:
        a = "https://cdn.example.com/show1/latest.mp3"
        b = "https://cdn.example.com/show2/latest.mp3"
        assert video_id_for(a) != video_id_for(b)
        assert video_id_for(a) == "cdn.example.com_show1_latest.mp3"


class TestTranscribeUrls:
    def test_skip_existing(self, monkeypatch, tmp_path: Path) -> None:
        out = tmp_path / "transcripts"
        vid = "aircAruvnKk"
        existing = out / vid / f"{vid}.txt"
        existing.parent.mkdir(parents=True)
        existing.write_text("already done", encoding="utf-8")

        called: list[str] = []

        def fake_transcribe_url(url, **kw):
            called.append(url)
            return _result(url)

        monkeypatch.setattr("localcaption.batch.transcribe_url", fake_transcribe_url)

        result = transcribe_urls(
            [f"https://www.youtube.com/watch?v={vid}"],
            out_dir=out,
            whisper_dir=tmp_path / "whisper.cpp",
        )
        assert called == []
        assert len(result.items) == 1
        assert result.items[0].status == "skipped"
        assert result.items[0].video_id == vid
        assert result.exit_code() == 0
        assert "skipped" in result.summary()

    def test_mixed_success_and_failure(self, monkeypatch, tmp_path: Path) -> None:
        ok_url = "https://www.youtube.com/watch?v=aircAruvnKk"
        bad_url = "https://www.youtube.com/watch?v=BYizgB2FcAQ"
        third = "https://youtu.be/PSRJfaAYkW4"

        def fake_transcribe_url(url, **kw):
            if "BYizgB2FcAQ" in url:
                raise DownloadError("HTTP 403")
            return _result(url, duration_s=323.0)

        monkeypatch.setattr("localcaption.batch.transcribe_url", fake_transcribe_url)

        result = transcribe_urls(
            [ok_url, bad_url, third],
            out_dir=tmp_path / "transcripts",
            whisper_dir=tmp_path / "whisper.cpp",
        )
        assert [i.status for i in result.items] == ["ok", "failed", "ok"]
        assert result.items[1].error is not None
        assert "DownloadError" in result.items[1].error
        assert "HTTP 403" in result.items[1].error
        assert result.exit_code() == 1

        table = result.summary()
        assert "✅" in table
        assert "❌" in table
        assert "aircAruvnKk" in table
        assert "BYizgB2FcAQ" in table
        assert "total: 3" in table
        assert "ok: 2" in table
        assert "failed: 1" in table

    def test_passes_isolated_out_dir_and_stem(self, monkeypatch, tmp_path: Path) -> None:
        captured: dict[str, object] = {}

        def fake_transcribe_url(url, **kw):
            captured.update(kw)
            captured["url"] = url
            return _result(url)

        monkeypatch.setattr("localcaption.batch.transcribe_url", fake_transcribe_url)

        out = tmp_path / "transcripts"
        transcribe_urls(
            ["https://youtu.be/PSRJfaAYkW4"],
            out_dir=out,
            whisper_dir=tmp_path / "w",
            model="small.en",
        )
        assert captured["out_dir"] == out / "PSRJfaAYkW4"
        assert captured["stem"] == "PSRJfaAYkW4"
        assert captured["model"] == "small.en"

    def test_forwards_backend(self, monkeypatch, tmp_path: Path) -> None:
        captured: dict[str, object] = {}

        def fake_transcribe_url(url, **kw):
            captured["backend"] = kw.get("backend")
            return _result(url)

        monkeypatch.setattr("localcaption.batch.transcribe_url", fake_transcribe_url)
        transcribe_urls(
            ["https://youtu.be/PSRJfaAYkW4"],
            out_dir=tmp_path / "transcripts",
            backend=BACKEND_FASTER_WHISPER,
        )
        assert captured["backend"] == BACKEND_FASTER_WHISPER

    def test_empty_list(self, tmp_path: Path) -> None:
        result = transcribe_urls(
            [],
            out_dir=tmp_path / "transcripts",
            whisper_dir=tmp_path / "w",
        )
        assert result.items == []
        assert result.exit_code() == 0
        assert "total: 0" in result.summary()

    def test_continues_after_failure(self, monkeypatch, tmp_path: Path) -> None:
        order: list[str] = []

        def fake_transcribe_url(url, **kw):
            order.append(url)
            if "fail" in url:
                raise DownloadError("nope")
            return _result(url)

        monkeypatch.setattr("localcaption.batch.transcribe_url", fake_transcribe_url)
        # Non-YouTube so video_id_for doesn't need a 11-char id.
        urls = [
            "https://example.com/ok1",
            "https://example.com/fail",
            "https://example.com/ok2",
        ]
        result = transcribe_urls(urls, out_dir=tmp_path / "t", whisper_dir=tmp_path / "w")
        assert order == urls
        assert [i.status for i in result.items] == ["ok", "failed", "ok"]

    def test_same_stem_local_files_both_run(self, monkeypatch, tmp_path: Path) -> None:
        called: list[str] = []

        def fake_transcribe_url(url, **kw):
            called.append(url)
            return _result(url)

        monkeypatch.setattr("localcaption.batch.transcribe_url", fake_transcribe_url)
        urls = [
            str(tmp_path / "show1" / "audio.mp3"),
            str(tmp_path / "show2" / "audio.mp3"),
        ]
        result = transcribe_urls(urls, out_dir=tmp_path / "t", whisper_dir=tmp_path / "w")
        assert called == urls
        assert [i.status for i in result.items] == ["ok", "ok"]
        assert result.items[0].video_id != result.items[1].video_id

    def test_expands_tilde_before_transcribe(self, monkeypatch, tmp_path: Path) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        captured: list[str] = []

        def fake_transcribe_url(url, **kw):
            captured.append(url)
            return _result(url)

        monkeypatch.setattr("localcaption.batch.transcribe_url", fake_transcribe_url)
        transcribe_urls(
            ["~/talk.mp4"],
            out_dir=tmp_path / "t",
            whisper_dir=tmp_path / "w",
        )
        assert captured == [str(tmp_path / "talk.mp4")]


class TestBatchResultSummary:
    def test_exit_zero_when_only_skips(self) -> None:
        result = BatchResult(
            items=[
                BatchItem(
                    source="https://youtu.be/PSRJfaAYkW4",
                    video_id="PSRJfaAYkW4",
                    status="skipped",
                    duration_s=None,
                    elapsed_s=0.0,
                )
            ],
            wall_clock_s=0,
        )
        assert result.exit_code() == 0
