"""Version helpers for MERN."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import re


def _version_from_pyproject() -> str:
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    if not pyproject.exists():
        raise FileNotFoundError(pyproject)
    match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject.read_text(), re.MULTILINE)
    if match is None:
        raise RuntimeError(f"Could not find project version in {pyproject}")
    return match.group(1)


def get_version() -> str:
    """Return the local source version, falling back to installed package metadata."""
    try:
        return _version_from_pyproject()
    except FileNotFoundError:
        try:
            return version("mern")
        except PackageNotFoundError:
            return "0+unknown"


__version__ = get_version()
