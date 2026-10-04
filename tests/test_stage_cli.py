"""The stages' tyro CLIs parse into the typed configs (nothing is run)."""

from pathlib import Path

import tyro

from markovian_policy.stages import cache_tokens, download_assets, evaluate, generate_expert, generate_human, train


def test_train_cli_overrides_nested_fields() -> None:
    config = tyro.cli(
        train.Config,
        args=[
            "--data.trace-cache-dirs", "scene", "/cache/scene", "wrist", "/cache/wrist",
            "--train.batch-size", "128", "--train.echo-factor", "3", "--train.checkpoint-every-frac", "0.25",
            "--policy.down-dims", "128", "256", "--train.ema.power", "0.7", "--train.keep-every-epochs", "5",
        ],
    )  # fmt: skip
    assert config.data.trace_cache_dirs == {"scene": Path("/cache/scene"), "wrist": Path("/cache/wrist")}
    assert (config.train.batch_size, config.train.echo_factor, config.train.checkpoint_every_frac) == (128, 3, 0.25)
    assert config.policy.down_dims == (128, 256) and config.train.ema.power == 0.7 and config.train.keep_every_epochs == 5
    assert config.data.cameras == config.policy.cameras == ("scene", "wrist")  # defaults agree


def test_data_stage_clis() -> None:
    expert = tyro.cli(generate_expert.Config, args=["--n-episodes", "10", "--env.noise-ratio", "0"])
    assert expert.n_episodes == 10 and expert.env.noise_ratio == 0
    human = tyro.cli(generate_human.Config, args=["--max-logs", "8", "--keep-incomplete"])
    assert human.max_logs == 8 and human.keep_incomplete


def test_cache_and_eval_stage_clis() -> None:
    cache = tyro.cli(cache_tokens.Config, args=["encode", "--zarr-path", "d.zarr", "--out-dir", "c", "--camera", "wrist", "--rank", "1"])
    assert (cache.mode, cache.cache.camera, cache.cache.rank) == ("encode", "wrist", 1)
    evaluation = tyro.cli(
        evaluate.Config, args=["--checkpoint", "m.ckpt", "--out-dir", "o", "--rollout.n-action-steps", "4", "--n-required-subtasks", "3"]
    )
    assert evaluation.rollout.n_action_steps == 4 and evaluation.n_required_subtasks == 3


def test_download_stage_cli() -> None:
    config = tyro.cli(download_assets.Config, args=["--assets", "human-demos", "--human-logs-path", "x/logs.zip"])
    assert config.assets == ("human-demos",) and config.human_logs_path == Path("x/logs.zip")
