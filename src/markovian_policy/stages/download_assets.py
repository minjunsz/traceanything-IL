"""Stage: download the pretrained TraceAnything weights and the human teleoperation logs."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from markovian_policy.data.assets import ASSETS, download


@dataclass
class Config:
    assets: tuple[Literal["trace-anything", "human-demos"], ...] = ("trace-anything", "human-demos")
    weights_path: Path = ASSETS["trace-anything"].path
    human_logs_path: Path = ASSETS["human-demos"].path


def run(config: Config) -> list[Path]:
    destinations = {"trace-anything": config.weights_path, "human-demos": config.human_logs_path}
    paths: list[Path] = []
    for name in config.assets:
        asset = ASSETS[name]
        print(f"{asset.name}: {asset.description}\n  {asset.url} -> {destinations[name]} ({asset.size / 2**30:.2f} GiB)")
        paths.append(download(asset.url, destinations[name], asset.size))
    return paths
