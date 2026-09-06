"""Smoke tests: the package imports cleanly and exposes the expected public API."""

from __future__ import annotations

import importlib


def test_package_imports() -> None:
    mod = importlib.import_module("localcaption")
    assert hasattr(mod, "__version__")
    assert isinstance(mod.__version__, str)


def test_cli_help_does_not_crash(capsys) -> None:
    import pytest

    from localcaption.cli import main

    # argparse exits with SystemExit(0) on --help; we just want it not to blow up.
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0

    captured = capsys.readouterr()
    assert "localcaption" in captured.out


def test_pipeline_public_surface() -> None:
    from localcaption.pipeline import PipelineResult, transcribe_url

    assert callable(transcribe_url)
    assert PipelineResult.__dataclass_fields__.keys() >= {
        "source_url", "audio_path", "wav_path", "transcripts", "summary",
    }
