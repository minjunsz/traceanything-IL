"""Downloadable assets (pretrained weights, human teleoperation logs) with resumable, size-checked downloads."""

import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from markovian_policy.paths import HUMAN_LOGS_ZIP, TRACE_WEIGHTS

CHUNK_BYTES = 1 << 20


@dataclass(frozen=True)
class Asset:
    name: str
    url: str
    size: int  # expected size in bytes, verified after the download
    path: Path  # default destination
    description: str


ASSETS: dict[str, Asset] = {
    "trace-anything": Asset(
        "trace-anything",
        "https://huggingface.co/depth-anything/trace-anything/resolve/main/trace_anything.pt",
        2_590_307_933,
        TRACE_WEIGHTS,
        "Pretrained TraceAnything weights (Hugging Face: depth-anything/trace-anything)",
    ),
    "human-demos": Asset(
        "human-demos",
        "https://github.com/google-research/relay-policy-learning/raw/master/kitchen_demos_multitask.zip",
        695_505_475,
        HUMAN_LOGS_ZIP,
        "Human teleoperation logs of the Relay Policy Learning kitchen (Git LFS file of that repository)",
    ),
}


def download(
    url: str, dest: Path, expected_size: int | None = None, opener: Callable[[urllib.request.Request], object] | None = None
) -> Path:
    """Download `url` to `dest`. Resumes a partial `<dest>.part`, writes atomically and verifies the size.

    An existing `dest` of the expected size is kept as is.
    """
    if dest.exists() and (expected_size is None or dest.stat().st_size == expected_size):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    done = part.stat().st_size if part.exists() else 0
    request = urllib.request.Request(url, headers={"Range": f"bytes={done}-"} if done else {})
    with (opener or urllib.request.urlopen)(request) as response, open(part, "ab" if done else "wb") as out:  # type: ignore[attr-defined]
        if done and getattr(response, "status", 206) != 206:  # the server ignored the range: start over
            out.seek(0)
            out.truncate()
        while chunk := response.read(CHUNK_BYTES):
            out.write(chunk)
    size = part.stat().st_size
    if expected_size is not None and size != expected_size:
        raise OSError(f"{dest.name}: downloaded {size} bytes, expected {expected_size} (partial file kept at {part}; rerun to resume)")
    part.replace(dest)
    return dest
