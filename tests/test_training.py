"""Samplers, top-k checkpoints, dataset serving and the Trainer end to end (CPU, synthetic data)."""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from conftest import LENGTH, N_EPISODES
from markovian_policy.data import trace_cache
from markovian_policy.data.dataset import DataConfig, FrankaKitchenDataset
from markovian_policy.policies.lowdim import DiffusionUnetLowdimPolicy
from markovian_policy.training.checkpoint import TopKCheckpoints, load_payload
from markovian_policy.training.config import TrainConfig
from markovian_policy.training.loop import Trainer
from markovian_policy.training.samplers import EchoDistributedSampler, ResumableDistributedSampler

LOWDIM = DataConfig(horizon=8, n_obs_steps=2, lowdim_keys=("agent_pos",), cameras=(), val_ratio=0.2)


def test_resumable_sampler_skips_the_consumed_prefix() -> None:
    sampler = ResumableDistributedSampler(list(range(50)), num_replicas=2, rank=0, seed=1)
    sampler.set_epoch(3)
    full = list(sampler)
    sampler.skip_samples = 7
    assert list(sampler) == full[7:] and len(sampler) == len(full) - 7


def test_echo_sampler_repeats_every_index() -> None:
    sampler = EchoDistributedSampler(list(range(40)), num_replicas=2, rank=1, echo=3, block=6, seed=0)
    sampler.set_epoch(2)
    indices = list(sampler)
    assert len(indices) == len(sampler) == 20 * 3
    counts = np.bincount(indices, minlength=40)
    assert set(counts[counts > 0]) == {3}  # this rank's 20 indices, three times each
    assert indices == list(sampler)  # reproducible for a given epoch


def test_topk_keeps_the_best_and_deletes_the_rest(tmp_path: Path) -> None:
    topk = TopKCheckpoints(tmp_path, k=2)
    for epoch, metric in enumerate([0.5, 0.3, 0.4, 0.2]):
        if path := topk.path_for(epoch, metric):
            path.write_text("x")
    assert sorted(topk.best.values()) == [0.2, 0.3]
    assert len(list(tmp_path.glob("*.ckpt"))) == 2


def test_dataset_windows_and_normalizer(demo_zarr: Path) -> None:
    dataset = FrankaKitchenDataset(dataclasses.replace(LOWDIM, zarr_path=demo_zarr))
    item = dataset[0]
    assert item["obs"]["agent_pos"].shape == (8, 9) and item["action"].shape == (8, 9)
    assert len(dataset) > 0 and len(dataset.get_validation_dataset()) > 0
    normalized = dataset.get_normalizer()["action"].normalize(item["action"])
    assert normalized.abs().max() <= 1.0 + 1e-5


def test_dataset_serves_cached_trace_tokens_per_camera(demo_zarr: Path, tmp_path: Path) -> None:
    n_obs, P, D = 2, 3, 5
    ends = np.arange(1, N_EPISODES + 1) * LENGTH
    _, offsets = trace_cache.window_layout(ends, n_obs)
    caches = {}
    for camera in ("scene", "wrist"):
        cache = tmp_path / f"cache_{camera}"
        cache.mkdir()
        tokens = np.random.default_rng(len(camera)).normal(size=(offsets[-1], P, D)).astype(np.float16)
        np.save(cache / trace_cache.TOKENS_FILE, tokens)
        (cache / trace_cache.META_FILE).write_text(json.dumps({"n_obs_steps": n_obs, "episode_ends": ends.tolist()}))
        (cache / trace_cache.COMPLETE_FILE).write_text("ok")
        caches[camera] = cache
    config = DataConfig(zarr_path=demo_zarr, horizon=8, n_obs_steps=n_obs, cameras=("scene", "wrist"), trace_cache_dirs=caches)
    dataset = FrankaKitchenDataset(config)
    assert "scene" not in dataset.replay_buffer  # rgb frames are never loaded with a cache
    item = dataset[5]
    for camera in ("scene", "wrist"):
        key = trace_cache.tokens_obs_key(camera, 2)
        expected = np.load(caches[camera] / trace_cache.TOKENS_FILE)[dataset.cache_rows[5]].astype(np.float32)
        np.testing.assert_array_equal(item["obs"][key].numpy(), expected)


def _lowdim_policy(dataset: FrankaKitchenDataset) -> DiffusionUnetLowdimPolicy:
    policy = DiffusionUnetLowdimPolicy(
        obs_dim=9, action_dim=9,
        ddpm_scheduler=DDPMScheduler(num_train_timesteps=10, beta_schedule="squaredcos_cap_v2"),
        horizon=8, n_action_steps=4, n_obs_steps=2, num_ddpm_inference_steps=4, num_ddim_inference_steps=2,
        diffusion_step_embed_dim=16, down_dims=(16, 32),
    )  # fmt: skip
    policy.set_normalizer(dataset.get_normalizer())
    return policy


def _trainer(demo_zarr: Path, out: Path, **overrides) -> Trainer:
    dataset = FrankaKitchenDataset(dataclasses.replace(LOWDIM, zarr_path=demo_zarr))
    config = TrainConfig(
        output_dir=out, batch_size=16, num_workers=0, num_epochs=2, total_train_steps=None, mixed_precision="no",
        cpu=True, max_val_steps=2, topk=2, **overrides,
    )  # fmt: skip
    return Trainer(config, _lowdim_policy(dataset), dataset, dataset.get_validation_dataset())


def test_trainer_runs_logs_and_checkpoints(demo_zarr: Path, tmp_path: Path) -> None:
    trainer = _trainer(demo_zarr, tmp_path / "run")
    trainer.fit()
    rows = [json.loads(line) for line in (tmp_path / "run" / "logs.json.txt").read_text().splitlines()]
    epoch_rows = [r for r in rows if "val_ddim_mse" in r]
    assert len(epoch_rows) == 2 and all(np.isfinite(r["val_loss"]) and "val_ddpm_mse" in r for r in epoch_rows)
    assert trainer.state.global_step == 2 * trainer.steps_per_epoch
    assert (tmp_path / "run" / "normalizer.pt").exists()
    payload = load_payload(tmp_path / "run" / "checkpoints" / "latest.ckpt")
    assert payload.epoch == 1 and payload.epoch_step == 0 and payload.ema_step == trainer.state.global_step
    assert payload.train_config["batch_size"] == 16
    assert len(list((tmp_path / "run" / "checkpoints").glob("epoch=*.ckpt"))) == 2


def test_trainer_resumes_mid_epoch(demo_zarr: Path, tmp_path: Path) -> None:
    out = tmp_path / "run"
    first = _trainer(demo_zarr, out, checkpoint_every_frac=0.25)
    stop_at = first.steps_per_epoch + 5  # interrupt in the second epoch, after its first fraction checkpoint (3 steps in)
    original_step = first.ema.step

    def interrupted(model: torch.nn.Module) -> None:
        if first.state.global_step >= stop_at:
            raise KeyboardInterrupt
        original_step(model)

    first.ema.step = interrupted  # type: ignore[method-assign]
    with pytest.raises(KeyboardInterrupt):
        first.fit()

    saved = load_payload(out / "checkpoints" / "latest.ckpt")
    assert saved.epoch == 1 and saved.epoch_step > 0

    second = _trainer(demo_zarr, out, checkpoint_every_frac=0.25)
    assert (second.state.epoch, second.state.epoch_step, second.state.global_step) == (1, saved.epoch_step, saved.global_step)
    assert second.ema.optimization_step == saved.ema_step
    second.fit()
    assert second.state.global_step == 2 * second.steps_per_epoch  # the interrupted epoch was finished, not repeated
