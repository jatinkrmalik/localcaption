"""Tests for the default transcribe subcommand and dispatcher."""

from __future__ import annotations

from pathlib import Path

import pytest

from localcaption.cli import main


class TestCliHelpText:
    def test_transcribe_help_mentions_local_file(self, capsys) -> None:
        with pytest.raises(SystemExit) as excinfo:
            main(["--help"])
        assert excinfo.value.code == 0
        out = capsys.readouterr().out
        assert "local video/audio file" in out

    def test_top_level_help_mentions_url_or_file(self, capsys) -> None:
        rc = main([])
        assert rc == 2
        out = capsys.readouterr().out
        assert "url-or-file" in out


class TestCliBackendFlag:
    def test_help_mentions_backend(self, capsys) -> None:
        with pytest.raises(SystemExit) as excinfo:
            main(["--help"])
        assert excinfo.value.code == 0
        out = capsys.readouterr().out
        assert "--backend" in out
        assert "faster-whisper" in out
        assert "whisper-cpp" in out

    def test_backend_flag_forwarded(self, monkeypatch) -> None:
        sentinel: dict = {}

        def fake_transcribe_url(url, **kw):
            sentinel.update(kw)
            sentinel["url"] = url
            raise SystemExit(0)

        monkeypatch.setattr("localcaption.cli.transcribe_url", fake_transcribe_url)
        with pytest.raises(SystemExit):
            main(["--backend", "faster-whisper", "https://example.com/v"])
        assert sentinel["backend"] == "faster-whisper"

    def test_env_var_selects_backend(self, monkeypatch) -> None:
        sentinel: dict = {}

        def fake_transcribe_url(url, **kw):
            sentinel.update(kw)
            raise SystemExit(0)

        monkeypatch.setenv("LOCALCAPTION_BACKEND", "faster-whisper")
        monkeypatch.setattr("localcaption.cli.transcribe_url", fake_transcribe_url)
        with pytest.raises(SystemExit):
            main(["https://example.com/v"])
        assert sentinel["backend"] == "faster-whisper"

    def test_flag_overrides_env(self, monkeypatch) -> None:
        sentinel: dict = {}

        def fake_transcribe_url(url, **kw):
            sentinel.update(kw)
            raise SystemExit(0)

        monkeypatch.setenv("LOCALCAPTION_BACKEND", "faster-whisper")
        monkeypatch.setattr("localcaption.cli.transcribe_url", fake_transcribe_url)
        with pytest.raises(SystemExit):
            main(["--backend", "whisper-cpp", "https://example.com/v"])
        assert sentinel["backend"] == "whisper-cpp"

    def test_unknown_env_backend_errors(self, monkeypatch, capsys) -> None:
        monkeypatch.setenv("LOCALCAPTION_BACKEND", "mlx")
        rc = main(["https://example.com/v"])
        assert rc == 1
        err = capsys.readouterr().err
        assert "Unknown transcription backend" in err

    def test_faster_whisper_skips_ggml_preflight(self, monkeypatch, tmp_path: Path) -> None:
        whisper_dir = tmp_path / "whisper.cpp"
        whisper_dir.mkdir()
        monkeypatch.setenv("LOCALCAPTION_WHISPER_DIR", str(whisper_dir))
        sentinel: dict = {}

        def fake_transcribe_url(url, **kw):
            sentinel.update(kw)
            raise SystemExit(0)

        monkeypatch.setattr("localcaption.cli.transcribe_url", fake_transcribe_url)
        with pytest.raises(SystemExit):
            main(["--backend", "faster-whisper", "https://example.com/v"])
        assert sentinel["backend"] == "faster-whisper"


class TestCliLocalFileDispatch:
    def test_local_file_path_passed_to_pipeline(self, monkeypatch, tmp_path: Path) -> None:
        video = tmp_path / "my_video.mp4"
        video.write_text("fake")
        sentinel: dict[str, str] = {}

        def fake_transcribe_url(url, **kw):
            sentinel["url"] = url
            raise SystemExit(0)

        monkeypatch.setattr("localcaption.cli.transcribe_url", fake_transcribe_url)
        with pytest.raises(SystemExit):
            main([str(video)])
        assert sentinel["url"] == str(video)

    def test_url_still_passed_to_pipeline(self, monkeypatch) -> None:
        sentinel: dict[str, str] = {}

        def fake_transcribe_url(url, **kw):
            sentinel["url"] = url
            raise SystemExit(0)

        monkeypatch.setattr("localcaption.cli.transcribe_url", fake_transcribe_url)
        with pytest.raises(SystemExit):
            main(["https://www.youtube.com/watch?v=dQw4w9WgXcQ"])
        assert sentinel["url"] == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    def test_relative_local_file_passed_to_pipeline(self, monkeypatch, tmp_path: Path) -> None:
        video = tmp_path / "relative.mp4"
        video.write_text("fake")
        sentinel: dict[str, str] = {}

        def fake_transcribe_url(url, **kw):
            sentinel["url"] = url
            raise SystemExit(0)

        monkeypatch.setattr("localcaption.cli.transcribe_url", fake_transcribe_url)
        with pytest.raises(SystemExit):
            main(["./relative.mp4"])
        assert sentinel["url"] == "./relative.mp4"
