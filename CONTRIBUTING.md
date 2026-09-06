# Contributing to localcaption

Thanks for considering a contribution! This project is intentionally small —
the goal is to stay a thin, dependable orchestrator over `yt-dlp`, `ffmpeg`,
and `whisper.cpp`. PRs that keep that surface tidy are very welcome.

## Quick start

```bash
git clone https://github.com/jatinkrmalik/localcaption
cd localcaption
./scripts/setup.sh                  # builds whisper.cpp, sets up venv
source .venv/bin/activate
pip install -e .[dev]               # if not already done by setup.sh
pytest                              # run the test suite
ruff check src tests                # lint
```

## Project layout

```
src/localcaption/
├── __init__.py
├── __main__.py        # python -m localcaption
├── _logging.py        # tiny stdout logger (no logging-module config)
├── audio.py           # ffmpeg → 16 kHz mono WAV (stage 2)
├── batch.py           # public Python API: transcribe_urls(...)
├── cli.py             # argparse entry point (the `localcaption` script)
├── download.py        # yt-dlp Python API wrapper (stage 1)
├── errors.py          # exception hierarchy
├── pipeline.py        # public Python API: transcribe_url(...)
├── summary.py         # optional local Ollama summary
├── whisper.py         # Backend protocol + transcribe() dispatcher
└── backends/          # whisper.cpp (default) and faster-whisper
scripts/
└── setup.sh           # bootstraps whisper.cpp + venv + model
tests/                 # pytest suite
```

Each pipeline stage is its own module so you can swap one out (e.g. replace
the `whisper.cpp` backend with `faster-whisper`) without touching the others.

## Pull request checklist

- [ ] `pytest` passes.
- [ ] `ruff check src tests` passes (the CI runs both).
- [ ] New behaviour is covered by a test.
- [ ] User-visible changes are noted in `CHANGELOG.md` under `## [Unreleased]`.
- [ ] Public APIs have docstrings.

## Reporting bugs

Please include:

- The exact command you ran.
- The output of `localcaption --version` and `yt-dlp --version`.
- Your OS + Python version (`python --version`).
- Whether `whisper.cpp` was built with CMake or `make`.

For YouTube-side issues (HTTP 4xx, "Sign in to confirm…"), try
`pip install -U yt-dlp` first — those are usually upstream extractor breakage,
not bugs in this project.

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md).
By participating you agree to abide by its terms.
