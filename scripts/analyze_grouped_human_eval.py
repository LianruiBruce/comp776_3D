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

from id_layers.grouped_human_eval_analysis import (  # noqa: E402
    GroupedHumanEvalAnalysisError,
    analyze_grouped_human_eval,
    write_grouped_human_eval_analysis,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and analyze the ten grouped v2 P0/P1/P3 pairwise response JSON "
            "files using the researcher-only answer key."
        )
    )
    parser.add_argument(
        "--answer-key",
        type=Path,
        default=Path("human_eval_v2/private/answer_key.json"),
        help=("Private v2 answer key. Defaults to human_eval_v2/private/answer_key.json."),
    )
    parser.add_argument(
        "--responses",
        type=Path,
        nargs="+",
        required=True,
        help=(
            "Exactly ten response JSON files, or one or more directories whose immediate "
            "*.json children are response files."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("human_eval_v2/analysis/analysis.json"),
        help=("Full analysis JSON path. Item and identity CSV files are written beside it."),
    )
    return parser


def _expand_response_paths(inputs: Sequence[Path]) -> list[Path]:
    paths: list[Path] = []
    for input_path in inputs:
        if input_path.is_dir():
            paths.extend(sorted(input_path.glob("*.json")))
        else:
            paths.append(input_path)
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    response_paths = _expand_response_paths(args.responses)
    try:
        report = analyze_grouped_human_eval(args.answer_key, response_paths)
        outputs = write_grouped_human_eval_analysis(args.output, report)
    except (FileNotFoundError, OSError, GroupedHumanEvalAnalysisError) as error:
        parser.error(str(error))

    summary = {
        "status": "complete",
        "study_id": report["study_id"],
        "form_count": report["validation"]["form_count"],
        "participant_count": report["validation"]["participant_count"],
        "unique_item_count": report["validation"]["unique_item_count"],
        "exploratory_only": True,
        "outputs": {name: str(path) for name, path in outputs.items()},
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
