"""Stage: gather training logs and evaluation results of all runs into tidy CSV tables for plotting."""

from markovian_policy.reporting.collect import CollectConfig, collect

Config = CollectConfig
run = collect
