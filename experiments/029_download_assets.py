"""Download the pretrained TraceAnything weights (2.6 GB) and the human teleoperation logs (0.7 GB).

    python experiments/029_download_assets.py [--assets trace-anything human-demos]

Files go to `data/` (git-ignored). Interrupted downloads resume; sizes are verified.
"""

import tyro

from markovian_policy.stages import download_assets

if __name__ == "__main__":
    download_assets.run(tyro.cli(download_assets.Config))
