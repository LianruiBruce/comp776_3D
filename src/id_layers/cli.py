from __future__ import annotations

import argparse

from .config import resolve_experiment_config
from .experiment import run_experiment
from .probe_experiment import run_probe_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a layer-wise face identity experiment")
    parser.add_argument("--config", required=True, help="Path to an experiment YAML file")
    parser.add_argument("--run-id", default=None, help="Optional explicit immutable run ID")
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    config = resolve_experiment_config(arguments.config)
    runner = (
        run_probe_experiment
        if config["experiment"]["kind"] == "equal_capacity_probes"
        else run_experiment
    )
    run_dir = runner(config, run_id=arguments.run_id)
    print(run_dir)


if __name__ == "__main__":
    main()
