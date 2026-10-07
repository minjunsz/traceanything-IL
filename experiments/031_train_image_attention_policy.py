"""Train the image-based attention baseline (double R3M ResNet18 encoder, cross-attention UNet; no TraceAnything).

Launched through the sbatch script next to this file (multi-GPU). `n_obs_steps` is the history length and the horizon
follows it (the paper uses horizon = n_obs_steps + 14); data and policy must agree. A 16-frame history on human data:
    --data.n-obs-steps 16 --policy.n-obs-steps 16 --data.horizon 30 --policy.horizon 30 \
    --data.lowdim-keys agent_pos --policy.lowdim-keys agent_pos --policy.lowdim-dim 9 --train.output-dir RUN

The two most recent frames also go through a second encoder (`--policy.short-range-obs-horizon`, dropout
`--policy.short-range-dropout`); `--policy.short-range-obs-horizon None` uses a single encoder.
Needs R3M's weights at data/weights/r3m_resnet18.pt (`--policy.r3m-weights` to change).
"""

import tyro

from markovian_policy.stages import train_image_attention

if __name__ == "__main__":
    train_image_attention.run(tyro.cli(train_image_attention.Config))
