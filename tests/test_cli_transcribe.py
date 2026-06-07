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
