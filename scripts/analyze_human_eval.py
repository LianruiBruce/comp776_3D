from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from id_layers.human_eval_analysis import (
    HumanEvalAnalysisError,
    analyze_human_eval,
    write_human_eval_analysis,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and analyze blinded human-evaluation response JSON files using the "
            "researcher-only answer key."
        )
    )
    parser.add_argument(
        "--answer-key",
        type=Path,
        required=True,
        help="Researcher-only private answer-key JSON produced by build_human_eval.py.",
    )
    parser.add_argument(
        "--responses",
        type=Path,
        nargs="+",
        required=True,
        help="One or more complete participant response JSON files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination for the decoded aggregate analysis JSON.",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260903)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        report = analyze_human_eval(
            args.answer_key,
            args.responses,
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.bootstrap_seed,
            confidence_level=args.confidence_level,
        )
        output = write_human_eval_analysis(args.output, report)
    except (FileNotFoundError, HumanEvalAnalysisError) as error:
        parser.error(str(error))

    summary = {
        "status": "complete",
        "study_id": report["study_id"],
        "rater_count": report["validation"]["rater_count"],
        "trial_count": report["validation"]["trial_count"],
        "output": str(output),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
