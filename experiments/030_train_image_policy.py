"""Train the image-based baseline policy (R3M ResNet18 per camera on raw frames, FiLM UNet; no TraceAnything).

Launched through the sbatch script next to this file (multi-GPU). `n_obs_steps` is the history length and the horizon
follows it (the paper uses horizon = n_obs_steps + 14), e.g. an 8-frame history on human data:
    --data.n-obs-steps 8 --policy.n-obs-steps 8 --data.horizon 22 --policy.horizon 22 \
    --data.lowdim-keys agent_pos --policy.lowdim-keys agent_pos --policy.lowdim-dim 9 --train.output-dir RUN

Needs R3M's weights at data/weights/r3m_resnet18.pt (`--policy.r3m-weights` to change; `None` trains from scratch).
"""

import tyro

from markovian_policy.stages import train_image

if __name__ == "__main__":
    train_image.run(tyro.cli(train_image.Config))
