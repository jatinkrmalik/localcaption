"""Tests for the pipeline orchestration layer."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from localcaption.pipeline import PipelineResult, _is_local_file, transcribe_url


class TestIsLocalFile:
    def test_existing_file_returns_true(self, tmp_path: Path) -> None:
        video = tmp_path / "video.mp4"
        video.write_text("fake")
        assert _is_local_file(str(video)) is True

    def test_relative_existing_file_returns_true(self, tmp_path: Path) -> None:
        video = tmp_path / "video.mp4"
        video.write_text("fake")
        import os
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            assert _is_local_file("./video.mp4") is True
            assert _is_local_file("video.mp4") is True
        finally:
            os.chdir(old_cwd)

    def test_url_returns_false(self) -> None:
        assert _is_local_file("https://www.youtube.com/watch?v=dQw4w9WgXcQ") is False
        assert _is_local_file("http://example.com/video.mp4") is False

    def test_nonexistent_path_returns_false(self, tmp_path: Path) -> None:
        assert _is_local_file(str(tmp_path / "does_not_exist.mp4")) is False

    def test_file_url_returns_false(self, tmp_path: Path) -> None:
        assert _is_local_file(f"file://{tmp_path}/video.mp4") is False


class TestTranscribeUrlLocalFile:
    def test_local_file_skips_download(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        video = tmp_path / "my_video.mp4"
        video.write_text("fake video")
        out_dir = tmp_path / "out"
        whisper_dir = tmp_path / "whisper.cpp"

        download_called = False

        def fake_download(url, work_dir):
            nonlocal download_called
            download_called = True
            return work_dir / "downloaded.mp4"

        def fake_to_whisper_wav(src, dst):
            dst.write_text("fake wav")
            return dst

        fake_transcripts = MagicMock()
        fake_transcripts.existing.return_value = {}

        def fake_transcribe(wav, model, out_base, *, whisper_dir, language):
            return fake_transcripts

        monkeypatch.setattr("localcaption.pipeline.download_audio", fake_download)
        monkeypatch.setattr("localcaption.pipeline.to_whisper_wav", fake_to_whisper_wav)
        monkeypatch.setattr("localcaption.pipeline.transcribe", fake_transcribe)

        result = transcribe_url(
            str(video),
            out_dir=out_dir,
            whisper_dir=whisper_dir,
            model="base.en",
            keep_intermediate=True,
        )

        assert download_called is False
        assert isinstance(result, PipelineResult)
        assert result.source_url == str(video)
        assert result.audio_path == video.resolve()

    def test_local_file_uses_correct_stem_for_output(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        video = tmp_path / "interview.mkv"
        video.write_text("fake video")
        out_dir = tmp_path / "out"
        whisper_dir = tmp_path / "whisper.cpp"

        captured = {}

        def fake_to_whisper_wav(src, dst):
            dst.write_text("fake wav")
            return dst

        def fake_transcribe(wav, model, out_base, *, whisper_dir, language):
            captured["out_base"] = out_base
            fake_transcripts = MagicMock()
            fake_transcripts.existing.return_value = {}
            return fake_transcripts

        monkeypatch.setattr("localcaption.pipeline.to_whisper_wav", fake_to_whisper_wav)
        monkeypatch.setattr("localcaption.pipeline.transcribe", fake_transcribe)

        transcribe_url(
            str(video),
            out_dir=out_dir,
            whisper_dir=whisper_dir,
            model="base.en",
        )

        assert captured["out_base"] == out_dir / "interview"

    def test_url_still_calls_download(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        out_dir = tmp_path / "out"
        whisper_dir = tmp_path / "whisper.cpp"
        work_dir = out_dir / ".work"
        work_dir.mkdir(parents=True)

        def fake_download(url, work_dir):
            downloaded = work_dir / "yt_video.m4a"
            downloaded.write_text("fake audio")
            return downloaded

        def fake_to_whisper_wav(src, dst):
            dst.write_text("fake wav")
            return dst

        fake_transcripts = MagicMock()
        fake_transcripts.existing.return_value = {}

        def fake_transcribe(wav, model, out_base, *, whisper_dir, language):
            return fake_transcripts

        monkeypatch.setattr("localcaption.pipeline.download_audio", fake_download)
        monkeypatch.setattr("localcaption.pipeline.to_whisper_wav", fake_to_whisper_wav)
        monkeypatch.setattr("localcaption.pipeline.transcribe", fake_transcribe)

        result = transcribe_url(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            out_dir=out_dir,
            whisper_dir=whisper_dir,
            model="base.en",
        )

        assert isinstance(result, PipelineResult)
        assert result.source_url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    def test_keep_intermediate_preserves_local_audio_path(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        video = tmp_path / "podcast.mp3"
        video.write_text("fake audio")
        out_dir = tmp_path / "out"
        whisper_dir = tmp_path / "whisper.cpp"

        def fake_to_whisper_wav(src, dst):
            dst.write_text("fake wav")
            return dst

        fake_transcripts = MagicMock()
        fake_transcripts.existing.return_value = {}

        def fake_transcribe(wav, model, out_base, *, whisper_dir, language):
            return fake_transcripts

        monkeypatch.setattr("localcaption.pipeline.to_whisper_wav", fake_to_whisper_wav)
        monkeypatch.setattr("localcaption.pipeline.transcribe", fake_transcribe)

        result = transcribe_url(
            str(video),
            out_dir=out_dir,
            whisper_dir=whisper_dir,
            model="base.en",
            keep_intermediate=True,
        )

        assert result.audio_path == video.resolve()
        assert result.wav_path is not None
        assert result.wav_path.name == "podcast.16k.wav"

    def test_no_keep_intermediate_clears_paths(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        video = tmp_path / "podcast.mp3"
        video.write_text("fake audio")
        out_dir = tmp_path / "out"
        whisper_dir = tmp_path / "whisper.cpp"

        def fake_to_whisper_wav(src, dst):
            dst.write_text("fake wav")
            return dst

        fake_transcripts = MagicMock()
        fake_transcripts.existing.return_value = {}

        def fake_transcribe(wav, model, out_base, *, whisper_dir, language):
            return fake_transcripts

        monkeypatch.setattr("localcaption.pipeline.to_whisper_wav", fake_to_whisper_wav)
        monkeypatch.setattr("localcaption.pipeline.transcribe", fake_transcribe)

        result = transcribe_url(
            str(video),
            out_dir=out_dir,
            whisper_dir=whisper_dir,
            model="base.en",
            keep_intermediate=False,
        )

        assert result.audio_path is None
        assert result.wav_path is None
