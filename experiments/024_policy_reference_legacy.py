"""Reference outputs of a trained fork policy (run in the `diffusion-policy-l40` pixi env, CPU is enough).

Builds the fork's DiffusionUnetTraceAnythingAttentionPolicy from its yaml, loads a fork checkpoint, and records for
fixed inputs (cached-token obs, so the frozen encoder is never used): the UNet noise prediction and the DDIM action
prediction from a fixed seed. Also stores the policy weights minus the frozen encoder (tiny compared with the 8 GB
checkpoint) so tests/test_policy_equivalence.py can load them into the src policy.

    python experiments/024_policy_reference_legacy.py [--ckpt PATH]   ->  output/policy_reference/fork_policy.pt
"""

import argparse
import sys
from pathlib import Path

sys.path[:0] = ["third_party/diffusion-policy-experiments", "third_party/TraceAnything"]

import dill
import hydra
import torch
from omegaconf import OmegaConf

FORK = Path("third_party/diffusion-policy-experiments")
DEFAULT_CKPT = FORK / "data/outputs/franka_kitchen/8_obs_trace_anything_markovian_expert_cache_echo5/checkpoints"
DEFAULT_CKPT = DEFAULT_CKPT / "epoch=001-val_loss=0.0099-val_ddim_mse=0.009088.ckpt"
B, TOKENS, TOKEN_DIM = 2, 768, 1024

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
parser.add_argument("--weights", choices=["ema_model", "model"], default="ema_model", help="the fork evaluates its EMA model")
args = parser.parse_args()

OmegaConf.register_new_resolver("eval", eval, replace=True)
cfg = OmegaConf.load(FORK / "config/franka_kitchen/8_obs_trace_anything_markovian_expert.yaml")
policy = hydra.utils.instantiate(cfg.policy)
payload = torch.load(args.ckpt, map_location="cpu", pickle_module=dill, weights_only=False)
policy.load_state_dict(payload["state_dicts"][args.weights])
# The fork's load_state_dict only keeps keys the freshly built model already has, and its normalizer starts empty,
# so the checkpoint's normalizer.* entries are dropped (fork eval loads normalizer.pt separately). Restore them.
from diffusion_policy.model.common.normalizer import LinearNormalizer

normalizer = LinearNormalizer()
prefix = "normalizer."
normalizer.load_state_dict({k[len(prefix) :]: v for k, v in payload["state_dicts"][args.weights].items() if k.startswith(prefix)})
policy.set_normalizer(normalizer)
policy.eval()

g = torch.Generator().manual_seed(0)
horizon, To = cfg.horizon, cfg.n_obs_steps
obs = {
    "agent_pos": torch.randn(B, horizon, 9, generator=g),
    "subtask_sequence": torch.eye(28)[torch.randint(0, 28, (B, horizon), generator=g)],
    "trace_tokens": torch.randn(B, TOKENS, TOKEN_DIM, generator=g),
}
actions = torch.randn(B, horizon, 9, generator=g)
noisy = torch.randn(B, horizon, 9, generator=g)
timesteps = torch.tensor([3, 57])

with torch.no_grad():
    nobs = policy.normalizer.normalize({k: v for k, v in obs.items() if k in policy.normalizer.params_dict})
    nobs = {**obs, **nobs}  # tokens are passed through unnormalized unless the fork normalizer has an entry for them
    tokens, positions, modalities, ranges = policy._encode_obs(nobs, B)
    noise_pred = policy.model(
        sample=noisy, timestep=timesteps, global_cond=tokens, temporal_positions=positions,
        modality_indices=modalities, range_indices=ranges,
    )  # fmt: skip
    torch.manual_seed(0)
    action = policy.predict_action({"obs": obs}, use_DDIM=True)

state = {k: v for k, v in policy.state_dict().items() if not k.startswith("trace_encoder.")}
out = Path("output/policy_reference")
out.mkdir(parents=True, exist_ok=True)
torch.save(
    {
        "state_dict": state, "obs": obs, "noisy": noisy, "timesteps": timesteps, "noise_pred": noise_pred,
        "cond_tokens": tokens, "action": action["action"], "action_pred": action["action_pred"],
        "ckpt": str(args.ckpt), "weights": args.weights,
    },
    out / "fork_policy.pt",
)  # fmt: skip
print("saved", out / "fork_policy.pt", "| noise_pred", tuple(noise_pred.shape), "| action", tuple(action["action"].shape))
