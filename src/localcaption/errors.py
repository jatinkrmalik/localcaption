"""Custom exceptions for localcaption.

Defining a small hierarchy makes it easy for callers (and tests) to catch
a specific failure mode rather than scraping error strings.
"""

from __future__ import annotations


class LocalCaptionError(Exception):
    """Base class for all localcaption errors."""


class DependencyError(LocalCaptionError):
    """A required external tool or model is missing."""


class DownloadError(LocalCaptionError):
    """yt-dlp failed to fetch the requested media."""


class AudioConversionError(LocalCaptionError):
    """ffmpeg failed to produce the expected WAV file."""


class TranscriptionError(LocalCaptionError):
    """whisper.cpp failed to produce a transcript."""


class InstallError(LocalCaptionError):
    """An automated install step (system dep, whisper.cpp clone/build) failed."""
