"""Write clearly marked synthetic self-endpoint planning scenarios to a NEW directory."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from id_layers.likeness_power import DEFAULT_POWER_CONFIG, run_power_grid  # noqa: E402
from id_layers.likeness_study import analyze_likeness_study, synthetic_study_example  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON or YAML; optional top-level power key")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = dict(DEFAULT_POWER_CONFIG)
    if args.config:
        config_path = args.config if args.config.is_absolute() else ROOT / args.config
        content = config_path.read_text(encoding="utf-8")
        if config_path.suffix.lower() == ".json":
            supplied = json.loads(content)
        else:
            import yaml

            supplied = yaml.safe_load(content)
        config.update(supplied.get("power", supplied))
    output = (args.output if args.output.is_absolute() else ROOT / args.output).resolve()
    protected = (ROOT / "artifacts/runs/20260903T050000Z_photomaker_v2_generation_pilot").resolve()
    if output == protected or output.is_relative_to(protected):
        raise ValueError("Cannot write inside frozen source run")
    if output.exists():
        raise FileExistsError("Output directory must be new; completed simulations are immutable")
    output.mkdir(parents=True)
    (output / "resolved_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output / "status.json").write_text(
        json.dumps({"status": "running", "evidence_type": "SIMULATION_ONLY"}), encoding="utf-8"
    )
    try:
        start = time.perf_counter()
        results = run_power_grid(config)
        (output / "power_scenarios.json").write_text(
            json.dumps(results, indent=2, allow_nan=False), encoding="utf-8"
        )
        with (output / "power_scenarios.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
        demonstration = synthetic_study_example(seed=config["seed"])
        keys = ("identities", "photos", "candidates", "raters", "assignments", "responses")
        analysis = analyze_likeness_study(
            *(demonstration[k] for k in keys), bootstrap_resamples=300, seed=config["seed"]
        )
        (output / "synthetic_schema_example.json").write_text(
            json.dumps(demonstration, indent=2, allow_nan=False), encoding="utf-8"
        )
        (output / "synthetic_analysis.json").write_text(
            json.dumps(analysis, indent=2, allow_nan=False), encoding="utf-8"
        )
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for axis, endpoint in zip(axes, ("D", "E"), strict=True):
            for missing in config["missing_rates"]:
                selected = [
                    r
                    for r in results
                    if r["endpoint"] == endpoint
                    and r["latent_mean_D"] == 0.5
                    and r["latent_mean_E"] == 0.5
                    and r["identity_gap_sd"] == 1.0
                    and r["missing_mechanism"] == "mcar"
                    and r["requested_missing_rate"] == missing
                ]
                if selected:
                    selected.sort(key=lambda r: r["identity_count"])
                    axis.plot(
                        [r["identity_count"] for r in selected],
                        [r["two_sided_zero_test_power"] for r in selected],
                        "o-",
                        label=f"Missing {missing:.0%}",
                    )
            axis.axhline(0.8, color="gray", linestyle="--", linewidth=1)
            axis.set(
                xlabel="Recruited identities",
                ylabel="Two-sided power against zero",
                ylim=(0, 1.02),
                title=f"SIMULATION ONLY: {endpoint}",
            )
            axis.legend()
        fig.suptitle(
            "Assumed latent D=0.5, E=0.5; identity gap SD=1.0; MCAR\n"
            "Not empirical human variance or a final sample-size decision",
            fontsize=10,
        )
        fig.savefig(output / "power_scenarios.png", dpi=160)
        fig.savefig(output / "power_scenarios.svg")
        plt.close(fig)
        elapsed = time.perf_counter() - start
        (output / "performance.json").write_text(
            json.dumps(
                {
                    "cpu_wall_seconds": elapsed,
                    "gpu_seconds": 0,
                    "new_download_bytes": 0,
                    "scenario_endpoint_rows": len(results),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (output / "status.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "evidence_type": "SIMULATION_ONLY",
                    "row_count": len(results),
                    "cpu_wall_seconds": elapsed,
                }
            ),
            encoding="utf-8",
        )
    except Exception as error:
        (output / "status.json").write_text(
            json.dumps(
                {"status": "failed", "error": str(error), "evidence_type": "SIMULATION_ONLY"}
            ),
            encoding="utf-8",
        )
        raise
    print(
        json.dumps(
            {
                "output": str(output),
                "scenario_endpoint_rows": len(results),
                "evidence_type": "SIMULATION_ONLY",
            }
        )
    )


if __name__ == "__main__":
    main()
