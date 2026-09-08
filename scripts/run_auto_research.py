"""check/run/resume/summarize the bounded automatic research workflow.

Examples (.venv-auto):
  python scripts/run_auto_research.py check
  python scripts/run_auto_research.py run --task diagnostics
  python scripts/run_auto_research.py resume
  python scripts/run_auto_research.py summarize
Each task has immutable numbered attempts; completed tasks are only verified/read.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TORCH_HOME", str(ROOT / "artifacts/cache/auto_research/torch"))
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from id_layers.auto_research import (  # noqa: E402
    Budget,
    BudgetStop,
    digest,
    dump,
    outside_source,
    tree_hashes,
)
from id_layers.process_supervision import (  # noqa: E402
    SupervisedTimeout,
    TerminationUnverified,
    run_supervised,
)
from id_layers.utils import (  # noqa: E402
    environment_metadata,
    python_package_lock,
    snapshot_research_sources,
)

GPU_TASK_LIMITS = {
    "diagnostics": 900,
    "vae": 1200,
    "photomaker": 1800,
    "pulid_smoke": 1800,
    "pulid_full": 2400,
}
TERMINATION_ALLOWANCE_SECONDS = 15.0


def read_records(source):
    return [
        json.loads(line)
        for line in (source / "generation_manifest.jsonl").read_text().splitlines()
        if line.strip()
    ]


def initialize(config):
    run = outside_source(ROOT / config["run_dir"], ROOT / config["source_run"])
    run.mkdir(parents=True, exist_ok=True)
    source = ROOT / config["source_run"]
    hashes = run / "source_hashes_before.json"
    if not hashes.exists():
        dump(hashes, tree_hashes(source))
    if not (run / "config.resolved.yaml").exists():
        (run / "config.resolved.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
    else:
        frozen = yaml.safe_load((run / "config.resolved.yaml").read_text())
        if frozen != config:
            raise ValueError("Configuration changed after initialization; create a new run")
    return run, source


def task_output(run, name):
    base = run / name
    attempts = sorted(base.glob("attempt_*")) if base.exists() else []
    return base / f"attempt_{len(attempts) + 1:02d}"


def perform(name, root, source, output, config, lock, run, reuse_generation=None):
    from id_layers import auto_diagnostics as diag

    if name == "audit":
        records = read_records(source)
        frozen = yaml.safe_load((source / "config.resolved.yaml").read_text())
        audit = diag.validate_generation_records(
            records,
            source,
            expected_count=192,
            expected_identities=frozen["data"]["identities"],
            expected_conditions=[r["name"] for r in frozen["generation"]["conditions"]],
            expected_prompts=list(frozen["generation"]["prompts"]),
        )
        pairs = diag.build_seed_pairs(records, expected_pair_count=96)
        dataset = pd.read_csv(source / "dataset_manifest.csv")
        assert len(dataset) == 56 and dataset.sample_id.nunique() == 56
        for row in dataset.itertuples():
            if digest(root / "data" / row.source_relative_path) != row.source_sha256:
                raise ValueError(f"Real input hash mismatch {row.sample_id}")
        from id_layers.auto_generation import prepare_generation_suite

        tasks, selection, gallery = prepare_generation_suite(
            pd.read_csv(root / "data/manifests/fei_full_seed20260902.csv"), root / "data"
        )
        dump(output / "generation_selection.json", selection)
        dump(output / "seed_pairs.json", pairs)
        # Old anonymous NPZ is deliberately not reused: every embedding recomputed.
        return {
            "generation_audit": audit,
            "real_inputs_verified": len(dataset),
            "seed_pairs": len(pairs),
            "new_tasks": len(tasks),
            "selection": selection,
            "legacy_npz_reused": False,
        }
    if name == "diagnostics":
        from id_layers.auto_evaluation import evaluate_collection

        frozen = pd.read_csv(source / "evaluation_v3/generated_identity_metrics.csv")
        records = read_records(source)
        if (
            set(frozen.sample_id) != set(r["sample_id"] for r in records)
            or frozen.sample_id.duplicated().any()
        ):
            raise ValueError("Frozen detection IDs do not uniquely match manifest")
        return evaluate_collection(
            root=root,
            source=source,
            output=output,
            records=records,
            dataset=pd.read_csv(source / "dataset_manifest.csv"),
            lock=lock,
            gallery_conditions=[
                "frontal_smile",
                "left_three_quarter",
                "right_three_quarter_expression",
            ],
            expected_seed_pairs=96,
            frozen_detection=frozen.set_index("sample_id", drop=False).to_dict("index"),
        )
    if name == "vae":
        from id_layers.auto_vae import run_vae

        return run_vae(root=root, source=source, output=output, lock=lock)
    if name in ("photomaker", "pulid_smoke", "pulid_full"):
        from id_layers.auto_evaluation import evaluate_collection
        from id_layers.auto_generation import (
            prepare_generation_suite,
            run_generation_cells,
            smoke_tasks,
        )
        from id_layers.generation import PhotoMakerV2Backend

        manifest = pd.read_csv(root / "data/manifests/fei_full_seed20260902.csv")
        tasks, selection, gallery = prepare_generation_suite(manifest, root / "data")
        extra_assets = {}
        if name.startswith("pulid"):
            compatibility = json.loads((run / "pulid_compatibility.json").read_text())
            if name == "pulid_smoke" and datetime.now(UTC) > datetime.fromisoformat(
                compatibility["deadline_utc"]
            ):
                raise RuntimeError("PuLID 60-minute compatibility deadline elapsed")
            tasks = [
                replace(
                    t,
                    prompt=t.prompt.replace("person img", "person"),
                    num_inference_steps=25,
                    guidance_scale=7.0,
                    start_merge_step=0,
                )
                for t in tasks
            ]
            extra_assets = {
                "pulid": json.loads(
                    (root / "artifacts/cache/auto_research/pulid/assets.resolved.json").read_text()
                ),
                "source_patch": json.loads(
                    (root / "artifacts/cache/auto_research/pulid/source_patch.json").read_text()
                ),
            }
        if name == "pulid_smoke":
            tasks = smoke_tasks(tasks, selection)
        ids = sorted({t.target_identity_id for t in tasks})
        subset = manifest.loc[
            manifest.identity_id.isin(ids)
            & manifest.condition.isin(
                [
                    "frontal_neutral",
                    "frontal_smile",
                    "left_three_quarter",
                    "right_three_quarter_expression",
                ]
            )
        ].copy()
        if name == "photomaker":

            def factory():
                return PhotoMakerV2Backend(
                    tasks,
                    photomaker_source=root / "third_party/PhotoMaker",
                    insightface_source=root / "third_party/InsightFace/python-package",
                    insightface_root=root / "artifacts/cache/models/insightface",
                    base_model=root / "artifacts/cache/models/PhotoMaker/RealVisXL_V4.0",
                    adapter_path=root
                    / "artifacts/cache/models/PhotoMaker/PhotoMaker-V2/photomaker-v2.bin",
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                    dtype="float16",
                    cpu_offload=True,
                )
        else:
            if name == "pulid_full":
                previous = json.loads((run / "tasks.json").read_text()).get("pulid_smoke", {})
                if previous.get("state") != "complete":
                    raise RuntimeError("PuLID smoke has not passed")
                smoke_output = root / previous["output"]
                if not json.loads((smoke_output / "result.json").read_text()).get(
                    "smoke_single_face_gate_passed"
                ):
                    raise RuntimeError("PuLID single-face engineering smoke gate has not passed")
            from pulid_auto_worker import make_backend

            def factory():
                return make_backend(tasks)

        generation_dir = reuse_generation or output / "generation"
        result = run_generation_cells(
            tasks,
            generation_dir,
            factory,
            run_metadata={
                "resolved_config": {
                    "auto_research": config,
                    "actual_generation": {
                        "steps": tasks[0].num_inference_steps,
                        "cfg": tasks[0].guidance_scale,
                        "sampler": "dpmpp_2m"
                        if name.startswith("pulid")
                        else "EulerDiscreteScheduler",
                        "latent_device": "cpu",
                        "same_base": "RealVisXL_V4.0",
                        "pulid_settings": {
                            "id_scale": 0.8,
                            "num_zero": 20,
                            "ortho_v2": True,
                            "vae_dtype": "float32",
                        }
                        if name.startswith("pulid")
                        else None,
                    },
                },
                "asset_lock": {**lock, **extra_assets},
                "selection": selection,
                "input_subset": json.loads(subset.to_json(orient="records")),
                "environment": environment_metadata(root),
            },
            forbidden_source=source,
            latent_device="cpu",
            resume=reuse_generation is not None,
        )
        if result.get("complete") != len(tasks):
            raise RuntimeError(f"Generation matrix not complete; failures preserved: {result}")
        result["evaluation"] = evaluate_collection(
            root=root,
            source=generation_dir,
            output=output / "evaluation",
            records=read_records(generation_dir),
            dataset=subset,
            lock=lock,
            gallery_conditions=[
                "frontal_smile",
                "left_three_quarter",
                "right_three_quarter_expression",
            ],
            expected_seed_pairs=len(tasks) // 2,
        )
        result["generation_path"] = str(generation_dir.relative_to(root))
        if name == "pulid_smoke":
            coverage = pd.read_csv(output / "evaluation/coverage.csv")
            generated_ids = {r["sample_id"] for r in read_records(generation_dir)}
            selected = coverage[coverage.sample_id.isin(generated_ids)]
            result["smoke_single_face_gate_passed"] = bool(
                len(selected) == 16 and selected.end_to_end_valid.all()
            )
            if not result["smoke_single_face_gate_passed"]:
                result["failures"] = {
                    "smoke_face_gate": "Not all16 frozen cells contain exactly one detected face"
                }
        return result
    if name == "simulation":
        existing = run / "simulation/status.json"
        if existing.exists() and json.loads(existing.read_text()).get("status") == "complete":
            actual = json.loads((run / "simulation/resolved_config.json").read_text())
            if any(actual.get(k) != v for k, v in config["power"].items()):
                raise ValueError("Existing simulation configuration differs")
            return {
                "simulated_only": True,
                "reused_completed": True,
                "output": str(run / "simulation"),
                **json.loads(existing.read_text()),
            }
        from id_layers.likeness_power import run_power_grid

        rows = run_power_grid(config["power"])
        pd.DataFrame(rows).to_csv(output / "power_scenarios.csv", index=False)
        return {"simulated_only": True, "scenario_endpoint_rows": len(rows)}
    raise ValueError(f"Unknown task: {name}")


def execute(name, config, retry_failed=False):
    run, source = initialize(config)
    statuspath = run / "tasks.json"
    states = json.loads(statuspath.read_text()) if statuspath.exists() else {}
    previous = states.get(name, {})
    if previous.get("state") in ("complete", "complete_with_limitations"):
        print(f"{name}: already complete; not overwritten", flush=True)
        return
    if previous.get("state") == "failed" and not retry_failed:
        print(
            f"{name}: failed attempt retained; requires --retry-failed for engineering correction",
            flush=True,
        )
        return
    output = task_output(run, name)
    output.mkdir(parents=True)
    if name == "vae" and previous.get("output"):
        previous_output = ROOT / previous["output"]
        # Reuse terminal reconstruction units in a new attempt, without touching
        # the previous evidence or retrying a failed photo. Downstream analysis
        # is reconstructed from the frozen controls/reconstructions.
        for folder in ("units", "control", "reconstruction"):
            if (previous_output / folder).exists():
                shutil.copytree(previous_output / folder, output / folder)
        for filename in ("config.json", "vae_runtime.json"):
            if (previous_output / filename).exists():
                shutil.copyfile(previous_output / filename, output / filename)
    reuse_generation = None
    if name in ("photomaker", "pulid_smoke", "pulid_full"):
        for attempt in sorted((run / name).glob("attempt_*"), reverse=True):
            candidate = attempt / "generation"
            if (candidate / "generation_manifest.jsonl").exists():
                reuse_generation = candidate
                break
    (output / "config.resolved.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    shutil.copyfile(ROOT / "configs/assets.lock.yaml", output / "assets.lock.resolved.yaml")
    dump(
        output / "source_snapshot_manifest.json",
        snapshot_research_sources(ROOT, output / "source_snapshot"),
    )
    dump(output / "environment.json", environment_metadata(ROOT))
    (output / "environment.lock.txt").write_text(python_package_lock(), encoding="utf-8")
    states[name] = {"state": "running", "output": str(output.relative_to(ROOT))}
    dump(statuspath, states)
    lock = yaml.safe_load((ROOT / "configs/assets.lock.yaml").read_text())
    budget = Budget(run / "resource_ledger.json")
    start = time.perf_counter()
    try:
        if name in ("diagnostics", "vae", "photomaker", "pulid_smoke", "pulid_full"):
            # Worst-case serial-task allowance. CLI parent applies a hard remaining-budget timeout.
            with budget.gpu(name, GPU_TASK_LIMITS[name]):
                result = perform(name, ROOT, source, output, config, lock, run, reuse_generation)
        else:
            result = perform(name, ROOT, source, output, config, lock, run, reuse_generation)
        dump(output / "result.json", result)
        state = (
            "complete_with_limitations"
            if (
                result.get("failures")
                or result.get("model_failures")
                or result.get("failed_unit_count", 0)
                or result.get("status") == "complete_with_failures"
            )
            else "complete"
        )
        states[name].update(state=state, elapsed_seconds=time.perf_counter() - start)
    except BudgetStop as e:
        states[name].update(
            state="budget_stopped", reason=str(e), elapsed_seconds=time.perf_counter() - start
        )
    except Exception as e:
        states[name].update(
            state="failed",
            reason=f"{type(e).__name__}: {e}",
            elapsed_seconds=time.perf_counter() - start,
        )
        (output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        traceback.print_exc()
    dump(output / "status.json", states[name])
    latest = json.loads(statuspath.read_text()) if statuspath.exists() else {}
    latest[name] = states[name]
    dump(statuspath, latest)
    print(json.dumps(states[name], ensure_ascii=False), flush=True)


def summarize(config):
    run, source = initialize(config)
    before = json.loads((run / "source_hashes_before.json").read_text())
    after = tree_hashes(source)
    changed = sorted(
        set(before) ^ set(after) | {k for k in set(before) & set(after) if before[k] != after[k]}
    )
    dump(
        run / "source_integrity.json",
        {"unchanged": not changed, "file_count": len(before), "changed_files": changed},
    )
    if changed:
        raise RuntimeError("Immutable source run changed")
    states = json.loads((run / "tasks.json").read_text()) if (run / "tasks.json").exists() else {}
    from id_layers.auto_analysis import analyze_auto_collection, analyze_vae_collection

    for name in ("diagnostics", "photomaker", "pulid_smoke", "pulid_full"):
        state = states.get(name, {})
        if state.get("state") not in ("complete", "complete_with_limitations"):
            continue
        evaluation = ROOT / state["output"]
        if name != "diagnostics":
            evaluation = evaluation / "evaluation"
        destination = run / (name + "_analysis")
        if not (destination / "summary.json").exists():
            analyze_auto_collection(
                evaluation,
                destination,
                new_expression_protocol=name != "diagnostics",
                resamples=config["diagnostics"]["bootstrap_resamples"],
                seed=config["diagnostics"]["bootstrap_seed"],
            )
    if (
        states.get("vae", {}).get("state") in ("complete", "complete_with_limitations")
        and not (run / "vae_analysis/summary.json").exists()
    ):
        analyze_vae_collection(ROOT / states["vae"]["output"], run / "vae_analysis")
    ledger = json.loads((run / "resource_ledger.json").read_text())
    supplement = run / "pulid_download_ledger.json"
    supplemental = (
        json.loads(supplement.read_text()) if supplement.exists() else {"download_bytes": 0}
    )
    combined = {
        "main_download_bytes": ledger["download_bytes"],
        "pulid_download_bytes": supplemental["download_bytes"],
        "total_download_bytes": ledger["download_bytes"] + supplemental["download_bytes"],
        "gpu_seconds": ledger["gpu_seconds"],
        "download_limit_bytes": config["budget"]["download_bytes"],
        "gpu_limit_seconds": config["budget"]["gpu_seconds"],
    }
    combined["within_budget"] = bool(
        combined["total_download_bytes"] <= combined["download_limit_bytes"]
        and combined["gpu_seconds"] <= combined["gpu_limit_seconds"]
    )
    if not combined["within_budget"]:
        raise BudgetStop("Combined ledger exceeds budget")
    dump(run / "resource_summary.json", combined)
    result = {
        "scope": config["scope"],
        "source_unchanged": True,
        "tasks": states,
        "resource_summary": combined,
        "resource_ledger": ledger,
    }
    dump(run / "summary.json", result)
    print(json.dumps({"source_unchanged": True, "tasks": states}, ensure_ascii=False, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["check", "run", "resume", "summarize"])
    p.add_argument("--config", type=Path, default=ROOT / "configs/auto_research.yaml")
    p.add_argument("--task")
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--worker", action="store_true")
    args = p.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if args.command == "summarize":
        return summarize(config)
    tasks = (
        ["audit"]
        if args.command == "check"
        else ([args.task] if args.task else [t for t in config["tasks"] if t != "summary"])
    )
    if args.worker:
        if len(tasks) != 1:
            raise ValueError("Worker expects exactly one task")
        return execute(tasks[0], config, args.retry_failed)
    run, _ = initialize(config)
    for name in tasks:
        known = (
            json.loads((run / "tasks.json").read_text()) if (run / "tasks.json").exists() else {}
        )
        if known.get(name, {}).get("state") in ("complete", "complete_with_limitations"):
            print(f"{name}: already complete; no worker launched", flush=True)
            continue
        ledger = Budget(run / "resource_ledger.json")
        remaining = ledger.state["gpu_limit_seconds"] - ledger.state["gpu_seconds"]
        accounted_before_launch = ledger.state["gpu_seconds"]
        timeout = None
        if name in GPU_TASK_LIMITS:
            if ledger.state.get("active_gpu"):
                raise BudgetStop(
                    "An active or unverified GPU worker remains; no new worker launched"
                )
            reason = None
            if remaining < GPU_TASK_LIMITS[name] + TERMINATION_ALLOWANCE_SECONDS:
                reason = (
                    "Insufficient remaining GPU budget for the next whole task "
                    "and verified termination"
                )
            timeout = min(remaining - TERMINATION_ALLOWANCE_SECONDS, GPU_TASK_LIMITS[name])
            if name == "pulid_smoke":
                compatibility = json.loads((run / "pulid_compatibility.json").read_text())
                compatibility_remaining = (
                    datetime.fromisoformat(compatibility["deadline_utc"]) - datetime.now(UTC)
                ).total_seconds()
                timeout = min(timeout, compatibility_remaining - TERMINATION_ALLOWANCE_SECONDS)
                if timeout <= 0:
                    reason = (
                        "PuLID compatibility time limit leaves no room to start "
                        "and stop another worker"
                    )
            if reason:
                known[name] = {
                    **known.get(name, {}),
                    "state": "budget_stopped",
                    "reason": reason,
                    "worker_launched": False,
                }
                dump(run / "tasks.json", known)
                print(f"{name}: {reason}", flush=True)
                continue
        executable = (
            str(ROOT / ".venv-pulid/Scripts/python.exe")
            if name.startswith("pulid")
            else sys.executable
        )
        command = [
            executable,
            str(Path(__file__).resolve()),
            "run",
            "--config",
            str(args.config),
            "--task",
            name,
            "--worker",
        ]
        if args.retry_failed:
            command.append("--retry-failed")
        log = run / f"{name}_console_{time.time_ns()}.log"
        print(f"{name}: log {log}", flush=True)
        with log.open("w", encoding="utf-8") as f:
            try:
                run_supervised(
                    command,
                    cwd=ROOT,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=timeout,
                )
            except TerminationUnverified as error:
                ledger = Budget(run / "resource_ledger.json")
                ledger.kind = "gpu"
                active = ledger.state.setdefault(
                    "active_gpu", {"name": name, "reservation_seconds": 0}
                )
                active["reservation_seconds"] = max(
                    active.get("reservation_seconds", 0), error.elapsed_seconds
                )
                active["termination_unverified"] = True
                active["termination_report"] = error.report
                ledger.save()
                states = (
                    json.loads((run / "tasks.json").read_text())
                    if (run / "tasks.json").exists()
                    else {}
                )
                states.setdefault(name, {}).update(
                    state="blocked_external", reason=str(error), termination_verified=False
                )
                dump(run / "tasks.json", states)
                raise BudgetStop(
                    "Worker termination unverified; active GPU reservation retained"
                ) from error
            except SupervisedTimeout as error:
                ledger = Budget(run / "resource_ledger.json")
                ledger.kind = "gpu"
                active = ledger.state.pop("active_gpu", {"name": name})
                already_charged = max(0, ledger.state["gpu_seconds"] - accounted_before_launch)
                charge = max(0, error.elapsed_seconds - already_charged)
                ledger.state["gpu_seconds"] += charge
                ledger.state["gpu_tasks"].append(
                    {
                        **active,
                        "elapsed_seconds": charge,
                        "status": "hard_timeout",
                        "supervised_process_elapsed_seconds": error.elapsed_seconds,
                        "previously_accounted_seconds": already_charged,
                        "termination_report": error.termination_report,
                        "accounting": (
                            "verified process-tree termination; entire process wall "
                            "time charged without duplicating an already completed "
                            "GPU context"
                        ),
                    }
                )
                ledger.save()
                states = (
                    json.loads((run / "tasks.json").read_text())
                    if (run / "tasks.json").exists()
                    else {}
                )
                states.setdefault(name, {}).update(
                    state="budget_stopped",
                    reason="Worker tree terminated at task/global/compatibility time limit",
                    termination_verified=True,
                )
                dump(run / "tasks.json", states)
                raise BudgetStop(states[name]["reason"]) from error
        print(log.read_text(encoding="utf-8")[-4500:], flush=True)


if __name__ == "__main__":
    main()
