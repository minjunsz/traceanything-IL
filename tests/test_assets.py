"""Resumable, size-checked asset downloads (stub HTTP responses, no network)."""

import urllib.request
from pathlib import Path

import pytest

from markovian_policy.data.assets import ASSETS, download

PAYLOAD = bytes(range(256)) * 40


class FakeResponse:
    def __init__(self, data: bytes, status: int) -> None:
        self._data, self.status, self._pos = data, status, 0

    def read(self, n: int) -> bytes:
        chunk, self._pos = self._data[self._pos : self._pos + n], self._pos + n
        return chunk

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def server(data: bytes, honor_range: bool = True, truncate: int | None = None):
    def opener(request: urllib.request.Request) -> FakeResponse:
        header = request.headers.get("Range")
        start = int(header.split("=")[1].rstrip("-")) if header else 0
        if header and honor_range:
            body, status = data[start:], 206
        else:
            body, status = data, 200
        return FakeResponse(body[:truncate] if truncate else body, status)

    return opener


def test_download_is_atomic_and_verified(tmp_path: Path) -> None:
    dest = download("http://x/f", tmp_path / "a" / "f.bin", len(PAYLOAD), server(PAYLOAD))
    assert dest.read_bytes() == PAYLOAD and not list(tmp_path.rglob("*.part"))


def test_interrupted_download_resumes_from_the_partial_file(tmp_path: Path) -> None:
    with pytest.raises(OSError, match="rerun to resume"):
        download("http://x/f", tmp_path / "f.bin", len(PAYLOAD), server(PAYLOAD, truncate=3000))
    assert (tmp_path / "f.bin.part").stat().st_size == 3000
    assert download("http://x/f", tmp_path / "f.bin", len(PAYLOAD), server(PAYLOAD)).read_bytes() == PAYLOAD


def test_server_ignoring_ranges_restarts_the_file(tmp_path: Path) -> None:
    (tmp_path / "f.bin.part").write_bytes(b"junk" * 100)
    download("http://x/f", tmp_path / "f.bin", len(PAYLOAD), server(PAYLOAD, honor_range=False))
    assert (tmp_path / "f.bin").read_bytes() == PAYLOAD


def test_complete_file_is_not_downloaded_again(tmp_path: Path) -> None:
    (tmp_path / "f.bin").write_bytes(PAYLOAD)

    def forbidden(request: urllib.request.Request) -> FakeResponse:
        raise AssertionError("must not download")

    assert download("http://x/f", tmp_path / "f.bin", len(PAYLOAD), forbidden).exists()


def test_registered_assets_are_well_formed() -> None:
    assert set(ASSETS) == {"trace-anything", "human-demos"}
    assert all(a.url.startswith("https://") and a.size > 0 for a in ASSETS.values())
