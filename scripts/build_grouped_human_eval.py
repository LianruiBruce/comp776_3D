from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from id_layers.grouped_human_eval import (  # noqa: E402
    GroupedHumanEvalError,
    build_grouped_human_eval,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the frozen v2 grouped pairwise identity-likeness study from the "
            "completed 192-image PhotoMaker pilot."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Completed generation run containing generation_manifest.jsonl.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/fei_full_seed20260902.csv"),
        help="FEI manifest containing original source_relative_path values.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Resolved generation config; defaults to RUN_DIR/config.resolved.yaml.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root against which FEI source_relative_path values are resolved.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output directory; defaults to RUN_DIR/human_eval_v2.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260904,
        help="Fixed opaque-ID, side-balancing, and trial-order seed.",
    )
    parser.add_argument(
        "--title",
        default="Blinded face identity-likeness evaluation",
        help="Participant-facing title; do not include method or condition names.",
    )
    return parser


def _from_repository_root(path: Path) -> Path:
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    run_dir = _from_repository_root(args.run_dir).resolve()
    manifest = _from_repository_root(args.manifest).resolve()
    data_root = _from_repository_root(args.data_root).resolve()
    config = None if args.config is None else _from_repository_root(args.config).resolve()
    output = (
        (run_dir / "human_eval_v2").resolve()
        if args.output is None
        else _from_repository_root(args.output).resolve()
    )
    try:
        build = build_grouped_human_eval(
            run_dir=run_dir,
            manifest_path=manifest,
            output_dir=output,
            config_path=config,
            data_root=data_root,
            seed=args.seed,
            title=args.title,
        )
    except (FileExistsError, FileNotFoundError, GroupedHumanEvalError) as error:
        parser.error(str(error))

    summary = {
        "status": "complete",
        "study_id": build.study_id,
        "output": str(build.output_dir),
        "unique_pairs": build.unique_pair_count,
        "display_forms": len(build.form_html_paths),
        "trials_per_form": build.display_trial_count // len(build.form_html_paths),
        "display_trials": build.display_trial_count,
        "participant_forms": [str(path) for path in build.form_html_paths],
        "answer_key": str(build.answer_key_path),
        "build_manifest": str(build.build_manifest_path),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
