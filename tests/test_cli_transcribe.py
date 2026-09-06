"""Tests for the default transcribe subcommand and dispatcher."""

from __future__ import annotations

import sys
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
        assert "default: small.en" in out
        assert "--batch" in out

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

    def test_backend_flag_forwarded(self, monkeypatch, dummy_faster_whisper) -> None:
        sentinel: dict = {}

        def fake_transcribe_url(url, **kw):
            sentinel.update(kw)
            sentinel["url"] = url
            raise SystemExit(0)

        monkeypatch.setattr("localcaption.cli.transcribe_url", fake_transcribe_url)
        with pytest.raises(SystemExit):
            main(["--backend", "faster-whisper", "https://example.com/v"])
        assert sentinel["backend"] == "faster-whisper"

    def test_env_var_selects_backend(self, monkeypatch, dummy_faster_whisper) -> None:
        sentinel: dict = {}

        def fake_transcribe_url(url, **kw):
            sentinel.update(kw)
            raise SystemExit(0)

        monkeypatch.setenv("LOCALCAPTION_BACKEND", "faster-whisper")
        monkeypatch.setattr("localcaption.cli.transcribe_url", fake_transcribe_url)
        with pytest.raises(SystemExit):
            main(["https://example.com/v"])
        assert sentinel["backend"] == "faster-whisper"

    def test_flag_overrides_env(self, monkeypatch, tmp_path: Path) -> None:
        sentinel: dict = {}

        def fake_transcribe_url(url, **kw):
            sentinel.update(kw)
            raise SystemExit(0)

        monkeypatch.setenv("LOCALCAPTION_BACKEND", "faster-whisper")
        monkeypatch.setattr("localcaption.cli.transcribe_url", fake_transcribe_url)
        missing = tmp_path / "no-whisper"
        with pytest.raises(SystemExit):
            main([
                "--backend", "whisper-cpp",
                "--whisper-dir", str(missing),
                "https://example.com/v",
            ])
        assert sentinel["backend"] == "whisper-cpp"

    def test_unknown_env_backend_errors(self, monkeypatch, capsys) -> None:
        monkeypatch.setenv("LOCALCAPTION_BACKEND", "mlx")
        rc = main(["https://example.com/v"])
        assert rc == 1
        err = capsys.readouterr().err
        assert "Unknown transcription backend" in err

    def test_faster_whisper_skips_ggml_preflight(
        self, monkeypatch, tmp_path: Path, dummy_faster_whisper
    ) -> None:
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

    def test_missing_extra_fails_before_pipeline(self, monkeypatch, capsys) -> None:
        monkeypatch.setitem(sys.modules, "faster_whisper", None)
        called: list[int] = []

        def fake_transcribe_url(url, **kw):
            called.append(1)
            raise SystemExit(0)

        monkeypatch.setattr("localcaption.cli.transcribe_url", fake_transcribe_url)
        rc = main(["--backend", "faster-whisper", "https://example.com/v"])
        assert rc == 1
        assert called == []
        assert "localcaption[faster]" in capsys.readouterr().err


class TestCliLocalFileDispatch:
    def test_local_file_path_passed_to_pipeline(self, monkeypatch, tmp_path: Path) -> None:
        video = tmp_path / "my_video.mp4"
        video.write_text("fake")
        sentinel: dict[str, str] = {}

        def fake_transcribe_url(url, **kw):
            sentinel["url"] = url
            sentinel["model"] = kw.get("model")
            raise SystemExit(0)

        monkeypatch.setattr("localcaption.cli.transcribe_url", fake_transcribe_url)
        monkeypatch.setattr("localcaption.cli._ensure_model_available", lambda *_a, **_k: True)
        with pytest.raises(SystemExit):
            main([str(video)])
        assert sentinel["url"] == str(video)
        assert sentinel["model"] == "small.en"

    def test_url_still_passed_to_pipeline(self, monkeypatch) -> None:
        sentinel: dict[str, str] = {}

        def fake_transcribe_url(url, **kw):
            sentinel["url"] = url
            sentinel["model"] = kw.get("model")
            raise SystemExit(0)

        monkeypatch.setattr("localcaption.cli.transcribe_url", fake_transcribe_url)
        monkeypatch.setattr("localcaption.cli._ensure_model_available", lambda *_a, **_k: True)
        with pytest.raises(SystemExit):
            main(["https://www.youtube.com/watch?v=dQw4w9WgXcQ"])
        assert sentinel["url"] == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert sentinel["model"] == "small.en"

    def test_relative_local_file_passed_to_pipeline(self, monkeypatch, tmp_path: Path) -> None:
        video = tmp_path / "relative.mp4"
        video.write_text("fake")
        sentinel: dict[str, str] = {}

        def fake_transcribe_url(url, **kw):
            sentinel["url"] = url
            sentinel["model"] = kw.get("model")
            raise SystemExit(0)

        monkeypatch.setattr("localcaption.cli.transcribe_url", fake_transcribe_url)
        monkeypatch.setattr("localcaption.cli._ensure_model_available", lambda *_a, **_k: True)
        with pytest.raises(SystemExit):
            main(["./relative.mp4"])
        assert sentinel["url"] == "./relative.mp4"
        assert sentinel["model"] == "small.en"


class TestCliBatch:
    def test_batch_flag_dispatches_parsed_urls(self, monkeypatch, tmp_path: Path, capsys) -> None:
        listing = tmp_path / "urls.txt"
        listing.write_text(
            "# queue\n"
            "https://www.youtube.com/watch?v=aircAruvnKk\n"
            "\n"
            "https://youtu.be/PSRJfaAYkW4\n",
            encoding="utf-8",
        )
        sentinel: dict[str, object] = {}

        def fake_transcribe_urls(urls, **kw):
            sentinel["urls"] = urls
            sentinel["out_dir"] = kw["out_dir"]
            sentinel["model"] = kw["model"]
            sentinel["backend"] = kw.get("backend")
            from localcaption.batch import BatchResult

            return BatchResult(items=[], wall_clock_s=0.0)

        monkeypatch.setattr("localcaption.cli.transcribe_urls", fake_transcribe_urls)
        out = tmp_path / "transcripts"
        whisper = tmp_path / "missing-whisper"  # not a dir → skip model prompt
        rc = main(
            [
                "--batch",
                str(listing),
                "-o",
                str(out),
                "-m",
                "small.en",
                "--whisper-dir",
                str(whisper),
            ]
        )
        assert rc == 0
        assert sentinel["urls"] == [
            "https://www.youtube.com/watch?v=aircAruvnKk",
            "https://youtu.be/PSRJfaAYkW4",
        ]
        assert sentinel["out_dir"] == out
        assert sentinel["model"] == "small.en"
        assert sentinel["backend"] == "whisper-cpp"
        assert "total: 0" in capsys.readouterr().out

    def test_batch_missing_file_exits_1(self, tmp_path: Path) -> None:
        rc = main(
            [
                "--batch",
                str(tmp_path / "nope.txt"),
                "--whisper-dir",
                str(tmp_path / "missing-whisper"),
            ]
        )
        assert rc == 1

    def test_batch_and_url_rejected(self, tmp_path: Path) -> None:
        listing = tmp_path / "urls.txt"
        listing.write_text("https://youtu.be/PSRJfaAYkW4\n", encoding="utf-8")
        with pytest.raises(SystemExit) as excinfo:
            main(["https://youtu.be/PSRJfaAYkW4", "--batch", str(listing)])
        assert excinfo.value.code == 2

    def test_neither_url_nor_batch_exits_2(self, capsys) -> None:
        with pytest.raises(SystemExit) as excinfo:
            main(["-m", "small.en"])
        assert excinfo.value.code == 2
        err = capsys.readouterr().err
        assert "provide a URL/file or --batch FILE" in err
        assert "not both" not in err

    def test_batch_nonzero_when_items_fail(self, monkeypatch, tmp_path: Path) -> None:
        listing = tmp_path / "urls.txt"
        listing.write_text("https://youtu.be/PSRJfaAYkW4\n", encoding="utf-8")

        def fake_transcribe_urls(urls, **kw):
            from localcaption.batch import BatchItem, BatchResult

            return BatchResult(
                items=[
                    BatchItem(
                        source=urls[0],
                        video_id="PSRJfaAYkW4",
                        status="failed",
                        duration_s=None,
                        elapsed_s=0.0,
                        error="DownloadError: HTTP 403",
                    )
                ],
                wall_clock_s=1.0,
            )

        monkeypatch.setattr("localcaption.cli.transcribe_urls", fake_transcribe_urls)
        rc = main(
            [
                "--batch",
                str(listing),
                "--whisper-dir",
                str(tmp_path / "missing-whisper"),
            ]
        )
        assert rc == 1


class TestCliSummaryFlags:
    def test_help_mentions_summary_flags(self, capsys) -> None:
        with pytest.raises(SystemExit) as excinfo:
            main(["--help"])
        assert excinfo.value.code == 0
        out = capsys.readouterr().out
        assert "--summary" in out
        assert "--summary-model" in out
        assert "--summary-prompt" in out
        assert "llama3.1:8b" in out

    def test_summary_flags_forwarded_to_pipeline(self, monkeypatch, tmp_path: Path) -> None:
        captured: dict = {}
        prompt = tmp_path / "prompt.txt"
        prompt.write_text("hi")

        def fake_transcribe_url(url, **kw):
            captured.update(kw)
            captured["url"] = url
            raise SystemExit(0)

        monkeypatch.setattr("localcaption.cli.transcribe_url", fake_transcribe_url)
        monkeypatch.setattr("localcaption.cli._ensure_model_available", lambda *_a, **_k: True)
        with pytest.raises(SystemExit):
            main(
                [
                    "https://example.com/v",
                    "--summary",
                    "--summary-model",
                    "mistral",
                    "--summary-prompt",
                    str(prompt),
                ]
            )
        assert captured["summary"] is True
        assert captured["summary_model"] == "mistral"
        assert captured["summary_prompt"] == prompt

    def test_summary_off_by_default(self, monkeypatch) -> None:
        captured: dict = {}

        def fake_transcribe_url(url, **kw):
            captured.update(kw)
            raise SystemExit(0)

        monkeypatch.setattr("localcaption.cli.transcribe_url", fake_transcribe_url)
        monkeypatch.setattr("localcaption.cli._ensure_model_available", lambda *_a, **_k: True)
        with pytest.raises(SystemExit):
            main(["https://example.com/v"])
        assert captured["summary"] is False
