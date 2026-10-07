"""Collect all runs' training logs and evaluation results into tidy CSV tables (output/results/*.csv).

    python experiments/034_collect_results.py [--runs-dir runs] [--eval-dir output/eval] [--out-dir output/results]

Rerun any time (cheap, CPU only): the tables are rebuilt from runs/<run>/ and <eval-dir>/<run>/T_a_<H>/<checkpoint>/.
Put facts that are not in the logs (data, encoder, job ids) in runs/<run>/meta.json; they become `meta_*` columns of runs.csv.
"""

import tyro

from markovian_policy.stages import collect_results

if __name__ == "__main__":
    collect_results.run(tyro.cli(collect_results.Config))
