"""Record an explicitly stopped own GPU worker without erasing incomplete cells.

This maintenance command must only be called after verifying the worker has exited.
It conservatively charges the entire wall time since the persistent GPU reservation.
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from id_layers.auto_research import Budget, dump  # noqa: E402 - checkout import after path setup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    import psutil

    for process in psutil.process_iter(["cmdline"]):
        try:
            command = " ".join(process.info["cmdline"] or [])
            if (
                "run_auto_research.py" in command
                and "--worker" in command
                and f"--task {args.task}" in command
            ):
                raise RuntimeError("Worker still running; cannot reconcile")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    run = ROOT / "artifacts/runs/20260905_auto_research"
    budget = Budget(run / "resource_ledger.json")
    budget.kind = "gpu"
    active = budget.state.pop("active_gpu")
    if active["name"] != args.task:
        raise ValueError("Wrong active GPU task")
    seconds = (datetime.now(UTC) - datetime.fromisoformat(active["started_utc"])).total_seconds()
    record = {
        **active,
        "elapsed_seconds": seconds,
        "status": "interrupted_engineering",
        "reason": args.reason,
        "accounting": "entire reservation wall interval; includes post-stop bookkeeping",
    }
    budget.state["gpu_seconds"] += seconds
    budget.state["gpu_tasks"].append(record)
    budget.save()
    states = json.loads((run / "tasks.json").read_text())
    output = ROOT / states[args.task]["output"]
    dump(output / "interruption.json", record)
    states[args.task].update(state="interrupted_engineering", reason=args.reason)
    dump(output / "status.json", states[args.task])
    dump(run / "tasks.json", states)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
