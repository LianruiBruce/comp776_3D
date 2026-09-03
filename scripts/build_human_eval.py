from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from id_layers.human_eval import (
    HumanEvalError,
    build_human_eval,
    write_private_answer_key,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build an offline, blinded human face-identity evaluation from a reference "
            "manifest and generation results."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Reference-image manifest in CSV, JSON, or JSONL format.",
    )
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Generation results or run manifest in CSV, JSON, or JSONL format.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help=(
            "Output directory (index.html, blinded_trials.json, media/) or an .html "
            "file (with sibling blinded JSON and media directory)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=776,
        help="Fixed randomization seed (default: 776).",
    )
    parser.add_argument(
        "--asset-root",
        type=Path,
        help="Optional first root used to resolve relative image paths.",
    )
    parser.add_argument(
        "--references-per-identity",
        type=int,
        default=3,
        help="Number of held-out images shown per reference identity (default: 3).",
    )
    parser.add_argument(
        "--title",
        default="Blinded face identity evaluation",
        help="Participant-facing study title; do not put condition or method names here.",
    )
    parser.add_argument(
        "--answer-key",
        type=Path,
        help=(
            "Optional researcher-only unblinding JSON. Keep it outside the files shared "
            "with participants."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        build = build_human_eval(
            args.manifest,
            args.results,
            args.output,
            seed=args.seed,
            asset_root=args.asset_root,
            references_per_identity=args.references_per_identity,
            title=args.title,
        )
        answer_key_path = None
        if args.answer_key is not None:
            answer_key_path = write_private_answer_key(
                args.answer_key,
                build.private_answer_key,
            )
    except (FileNotFoundError, HumanEvalError) as error:
        parser.error(str(error))

    summary = {
        "status": "complete",
        "study_id": build.public_study["study_id"],
        "trial_count": build.trial_count,
        "html": str(build.html_path),
        "blinded_json": str(build.blinded_json_path),
        "media_directory": str(build.media_dir),
        "private_answer_key": None if answer_key_path is None else str(answer_key_path),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
