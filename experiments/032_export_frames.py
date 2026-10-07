"""Export the camera frames of a dataset as uncompressed .npy (memory-mapped by the image-policy training).

    python experiments/032_export_frames.py --zarr-path data/human.zarr      # -> data/human.raw/{scene,wrist}.npy
Then train with `--data.mmap-dir data/human.raw`. CPU only; resumable (finished files are kept).
"""

import tyro

from markovian_policy.stages import export_frames

if __name__ == "__main__":
    export_frames.run(tyro.cli(export_frames.Config))
