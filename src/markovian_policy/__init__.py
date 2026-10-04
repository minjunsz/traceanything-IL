"""Markovian policies for long-horizon manipulation in the Franka kitchen.

The package is organised in layers, from low-level to high-level:

- `sim`: MuJoCo kitchen as a gymnasium environment (physics, actuation, task definitions).
- `experts`: scripted Markovian expert (finite state machines + operational-space control).
- `perception`: frozen visual encoders (TraceAnything).
- `nn`, `diffusion`, `policies`: networks, diffusion math and the diffusion policies built from them.
- `data`: datasets, token caches and the demonstration-generation pipeline.
- `training`, `evaluation`: the trainer and the closed-loop evaluator.
- `stages`: one facade per pipeline stage (generate data, cache tokens, train, evaluate).
"""
