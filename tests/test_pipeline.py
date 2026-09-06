"""Tests for the pipeline orchestration layer."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from localcaption.errors import DependencyError
from localcaption.pipeline import PipelineResult, _is_local_file, transcribe_url
from localcaption.whisper import BACKEND_FASTER_WHISPER, DEFAULT_BACKEND


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

        def fake_transcribe(wav, model, out_base, *, whisper_dir, language, **kwargs):
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

        def fake_transcribe(wav, model, out_base, *, whisper_dir, language, **kwargs):
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

    def test_explicit_stem_overrides_filename(self, monkeypatch, tmp_path: Path) -> None:
        video = tmp_path / "interview.mkv"
        video.write_text("fake video")
        out_dir = tmp_path / "out"
        captured: dict[str, Path] = {}

        def fake_to_whisper_wav(src, dst):
            dst.write_text("fake wav")
            return dst

        def fake_transcribe(wav, model, out_base, *, whisper_dir, language, **kwargs):
            captured["out_base"] = out_base
            fake_transcripts = MagicMock()
            fake_transcripts.existing.return_value = {}
            return fake_transcripts

        monkeypatch.setattr("localcaption.pipeline.to_whisper_wav", fake_to_whisper_wav)
        monkeypatch.setattr("localcaption.pipeline.transcribe", fake_transcribe)

        transcribe_url(
            str(video),
            out_dir=out_dir,
            whisper_dir=tmp_path / "whisper.cpp",
            model="base.en",
            stem="custom_id",
        )

        assert captured["out_base"] == out_dir / "custom_id"

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

        def fake_transcribe(wav, model, out_base, *, whisper_dir, language, **kwargs):
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

        def fake_transcribe(wav, model, out_base, *, whisper_dir, language, **kwargs):
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

        def fake_transcribe(wav, model, out_base, *, whisper_dir, language, **kwargs):
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
        assert result.summary is None

    def test_forwards_backend(self, monkeypatch, tmp_path: Path, dummy_faster_whisper) -> None:
        video = tmp_path / "clip.mp4"
        video.write_text("fake")
        captured: dict = {}

        def fake_to_whisper_wav(src, dst):
            dst.write_text("fake wav")
            return dst

        def fake_transcribe(wav, model, out_base, *, whisper_dir, language, **kwargs):
            captured["backend"] = kwargs.get("backend", DEFAULT_BACKEND)
            fake_transcripts = MagicMock()
            fake_transcripts.existing.return_value = {}
            return fake_transcripts

        monkeypatch.setattr("localcaption.pipeline.to_whisper_wav", fake_to_whisper_wav)
        monkeypatch.setattr("localcaption.pipeline.transcribe", fake_transcribe)

        transcribe_url(
            str(video),
            out_dir=tmp_path / "out",
            backend=BACKEND_FASTER_WHISPER,
        )
        assert captured["backend"] == BACKEND_FASTER_WHISPER

    def test_missing_faster_whisper_extra_skips_download(self, monkeypatch, tmp_path: Path) -> None:
        monkeypatch.setitem(sys.modules, "faster_whisper", None)
        download_called = False

        def fake_download(url, work_dir):
            nonlocal download_called
            download_called = True
            return work_dir / "downloaded.m4a"

        monkeypatch.setattr("localcaption.pipeline.download_audio", fake_download)

        with pytest.raises(DependencyError, match="localcaption\\[faster\\]"):
            transcribe_url(
                "https://example.com/v",
                out_dir=tmp_path / "out",
                backend=BACKEND_FASTER_WHISPER,
            )
        assert download_called is False

    def test_unknown_backend_skips_download(self, monkeypatch, tmp_path: Path) -> None:
        download_called = False

        def fake_download(url, work_dir):
            nonlocal download_called
            download_called = True
            return work_dir / "downloaded.m4a"

        monkeypatch.setattr("localcaption.pipeline.download_audio", fake_download)

        with pytest.raises(DependencyError, match="Unknown transcription backend"):
            transcribe_url(
                "https://example.com/v",
                out_dir=tmp_path / "out",
                backend="mlx",
            )
        assert download_called is False

    def test_whisper_cpp_without_dir_skips_download(self, monkeypatch, tmp_path: Path) -> None:
        download_called = False

        def fake_download(url, work_dir):
            nonlocal download_called
            download_called = True
            return work_dir / "downloaded.m4a"

        monkeypatch.setattr("localcaption.pipeline.download_audio", fake_download)

        with pytest.raises(DependencyError, match="requires a whisper.cpp directory"):
            transcribe_url("https://example.com/v", out_dir=tmp_path / "out")
        assert download_called is False


class TestTranscribeUrlSummary:
    def _stub_stages(self, monkeypatch) -> None:
        def fake_to_whisper_wav(src, dst):
            dst.write_text("fake wav")
            return dst

        def fake_transcribe(wav, model, out_base, *, whisper_dir, language, **kwargs):
            txt = out_base.with_suffix(".txt")
            txt.parent.mkdir(parents=True, exist_ok=True)
            txt.write_text("hello from the talk")
            fake_transcripts = MagicMock()
            fake_transcripts.txt = txt
            fake_transcripts.existing.return_value = {"txt": txt}
            return fake_transcripts

        monkeypatch.setattr("localcaption.pipeline.to_whisper_wav", fake_to_whisper_wav)
        monkeypatch.setattr("localcaption.pipeline.transcribe", fake_transcribe)

    def test_summary_true_writes_and_returns_path(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        video = tmp_path / "interview.mkv"
        video.write_text("fake video")
        out_dir = tmp_path / "out"
        whisper_dir = tmp_path / "whisper.cpp"
        self._stub_stages(monkeypatch)

        written: dict[str, object] = {}

        def fake_write_summary(txt, *, model, prompt_path):
            written["txt"] = txt
            written["model"] = model
            written["prompt_path"] = prompt_path
            out = Path(txt).with_name(f"{Path(txt).stem}.summary.md")
            out.write_text("sum")
            return out

        monkeypatch.setattr("localcaption.pipeline.write_summary", fake_write_summary)

        result = transcribe_url(
            str(video),
            out_dir=out_dir,
            whisper_dir=whisper_dir,
            model="base.en",
            summary=True,
            summary_model="mistral",
            summary_prompt=tmp_path / "prompt.txt",
        )

        assert result.summary == out_dir / "interview.summary.md"
        assert written["model"] == "mistral"
        assert written["prompt_path"] == tmp_path / "prompt.txt"
        assert Path(written["txt"]).name == "interview.txt"

    def test_summary_off_by_default_does_not_call_writer(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        video = tmp_path / "interview.mkv"
        video.write_text("fake video")
        called = {"n": 0}

        def fake_write_summary(*args, **kwargs):
            called["n"] += 1
            return None

        self._stub_stages(monkeypatch)
        monkeypatch.setattr("localcaption.pipeline.write_summary", fake_write_summary)

        result = transcribe_url(
            str(video),
            out_dir=tmp_path / "out",
            whisper_dir=tmp_path / "whisper.cpp",
            model="base.en",
        )
        assert called["n"] == 0
        assert result.summary is None

    def test_summary_failure_does_not_fail_pipeline(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        video = tmp_path / "interview.mkv"
        video.write_text("fake video")
        self._stub_stages(monkeypatch)

        def fake_write_summary(*args, **kwargs):
            return None

        monkeypatch.setattr("localcaption.pipeline.write_summary", fake_write_summary)

        result = transcribe_url(
            str(video),
            out_dir=tmp_path / "out",
            whisper_dir=tmp_path / "whisper.cpp",
            model="base.en",
            summary=True,
        )
        assert isinstance(result, PipelineResult)
        assert result.summary is None
