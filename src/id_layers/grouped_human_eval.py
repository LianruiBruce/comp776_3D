# ruff: noqa: E501
from __future__ import annotations

import hashlib
import io
import json
import os
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import yaml
from PIL import Image

from .human_eval import (
    HumanEvalError,
    _opaque_id,
    _read_records,
    _sha256_file,
    _stable_digest,
    _write_text_atomic,
)

PUBLIC_SCHEMA_VERSION = "identity-layers-grouped-human-eval/v2"
RESPONSE_SCHEMA_VERSION = "identity-layers-grouped-human-response/v2"
ANSWER_KEY_SCHEMA_VERSION = "identity-layers-grouped-human-eval-key/v2"
BUILD_SCHEMA_VERSION = "identity-layers-grouped-human-eval-build/v2"

REFERENCE_CONDITIONS = (
    "frontal_smile",
    "left_three_quarter",
    "right_three_quarter_expression",
)
CONTRASTS: dict[str, tuple[str, str]] = {
    # In v2, the first condition is always the preregistered focal condition.
    "P0": ("k4_diverse_full", "k1_full"),
    "P1": ("k4_diverse_full", "k4_repeat_full"),
    "P3": ("global_target_patch_donor", "global_donor_patch_target"),
}
CONTENT_BLOCKS = ("A", "B")
DISPLAY_COPIES_PER_BLOCK = 5
EXPECTED_IDENTITIES = 8
EXPECTED_PROMPTS = 2
EXPECTED_SEEDS = 2
EXPECTED_GENERATIONS = 192
EXPECTED_UNIQUE_PAIRS = 96
EXPECTED_PAIRS_PER_CONTENT_BLOCK = 48
EXPECTED_TRIALS_PER_FORM = 48
EXPECTED_DISPLAY_FORMS = 10
EXPECTED_DISPLAY_TRIALS = 480
EXPECTED_REFERENCE_IMAGES = EXPECTED_IDENTITIES * len(REFERENCE_CONDITIONS)
EXPECTED_RELEVANT_GENERATED_IMAGES = 160


class GroupedHumanEvalError(HumanEvalError):
    """Raised when the frozen grouped pairwise design cannot be built exactly."""


@dataclass(frozen=True)
class GroupedHumanEvalBuild:
    output_dir: Path
    participant_dir: Path
    media_dir: Path
    answer_key_path: Path
    build_manifest_path: Path
    form_html_paths: tuple[Path, ...]
    form_json_paths: tuple[Path, ...]
    study_id: str
    unique_pair_count: int
    display_trial_count: int


def _required_text(value: Any, context: str) -> str:
    if value is None or not str(value).strip():
        raise GroupedHumanEvalError(f"{context} must be a non-empty string")
    return str(value).strip()


def _required_int(value: Any, context: str) -> int:
    if isinstance(value, bool):
        raise GroupedHumanEvalError(f"{context} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise GroupedHumanEvalError(f"{context} must be an integer") from error
    if isinstance(value, float) and not value.is_integer():
        raise GroupedHumanEvalError(f"{context} must be an integer")
    return result


def _resolve_inside(base: Path, value: str, context: str) -> Path:
    raw = Path(value)
    target = raw.resolve() if raw.is_absolute() else (base / raw).resolve()
    root = base.resolve()
    try:
        target.relative_to(root)
    except ValueError as error:
        raise GroupedHumanEvalError(f"{context} escapes {root}: {value}") from error
    if not target.is_file():
        raise FileNotFoundError(f"{context} does not exist: {target}")
    return target


def _load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GroupedHumanEvalError(f"Expected a YAML mapping in {path}")
    return value


def _ordered_design_axes(config: Mapping[str, Any]) -> tuple[list[str], list[str], list[int]]:
    try:
        identities = [str(value) for value in config["data"]["identities"]]
        prompts = [str(value) for value in config["generation"]["prompts"]]
        seeds = [int(value) for value in config["generation"]["seeds"]]
    except (KeyError, TypeError, ValueError) as error:
        raise GroupedHumanEvalError(
            "Resolved generation config lacks ordered identities, prompts, or seeds"
        ) from error
    if len(identities) != EXPECTED_IDENTITIES or len(set(identities)) != len(identities):
        raise GroupedHumanEvalError(
            f"Frozen v2 design requires {EXPECTED_IDENTITIES} unique identities"
        )
    if len(prompts) != EXPECTED_PROMPTS or len(set(prompts)) != len(prompts):
        raise GroupedHumanEvalError(f"Frozen v2 design requires {EXPECTED_PROMPTS} prompts")
    if len(seeds) != EXPECTED_SEEDS or len(set(seeds)) != len(seeds):
        raise GroupedHumanEvalError(f"Frozen v2 design requires {EXPECTED_SEEDS} base seeds")
    return identities, prompts, seeds


def _load_original_references(
    manifest_path: Path,
    data_root: Path,
    identities: Sequence[str],
) -> dict[str, tuple[dict[str, Any], ...]]:
    rows = _read_records(manifest_path, ("samples", "manifest", "records"))
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    identity_set = set(identities)
    condition_set = set(REFERENCE_CONDITIONS)
    for index, row in enumerate(rows, start=1):
        identity = str(row.get("identity_id", "")).strip()
        condition = str(row.get("condition", "")).strip()
        if identity not in identity_set or condition not in condition_set:
            continue
        key = (identity, condition)
        if key in by_key:
            raise GroupedHumanEvalError(f"Duplicate reference manifest cell {key!r}")
        source_relative_path = _required_text(
            row.get("source_relative_path"),
            f"reference manifest row {index} source_relative_path",
        )
        source_path = _resolve_inside(
            data_root,
            source_relative_path,
            f"reference manifest row {index} source image",
        )
        expected_sha256 = _required_text(
            row.get("source_sha256"), f"reference manifest row {index} source_sha256"
        ).lower()
        observed_sha256 = _sha256_file(source_path)
        if observed_sha256 != expected_sha256:
            raise GroupedHumanEvalError(
                f"Reference SHA256 mismatch for {row.get('sample_id')}: "
                f"{observed_sha256} != {expected_sha256}"
            )
        by_key[key] = {
            "sample_id": _required_text(
                row.get("sample_id"), f"reference manifest row {index} sample_id"
            ),
            "identity_id": identity,
            "condition": condition,
            "source_relative_path": source_relative_path,
            "source_path": source_path,
            "source_sha256": observed_sha256,
        }

    expected_keys = {
        (identity, condition) for identity in identities for condition in REFERENCE_CONDITIONS
    }
    if set(by_key) != expected_keys:
        missing = sorted(expected_keys - set(by_key))
        unexpected = sorted(set(by_key) - expected_keys)
        raise GroupedHumanEvalError(
            f"Original-reference matrix mismatch: missing={missing}, unexpected={unexpected}"
        )
    references = {
        identity: tuple(by_key[(identity, condition)] for condition in REFERENCE_CONDITIONS)
        for identity in identities
    }
    if sum(len(value) for value in references.values()) != EXPECTED_REFERENCE_IMAGES:
        raise AssertionError("Frozen original-reference count changed")
    return references


def _validate_generation_route(row: Mapping[str, Any], context: str) -> None:
    target = _required_text(row.get("identity_id"), f"{context} identity_id")
    donor = _required_text(row.get("donor_identity_id"), f"{context} donor_identity_id")
    condition = _required_text(row.get("condition"), f"{context} condition")
    global_identity = row.get("global_embedding_identity_id")
    patch_identity = row.get("patch_image_identity_id")
    expected = {
        "k1_full": (target, target),
        "k4_diverse_full": (target, target),
        "k4_repeat_full": (target, target),
        "global_target_patch_donor": (target, donor),
        "global_donor_patch_target": (donor, target),
    }
    if condition not in expected:
        return
    observed = (
        None if global_identity is None else str(global_identity),
        None if patch_identity is None else str(patch_identity),
    )
    if observed != expected[condition]:
        raise GroupedHumanEvalError(
            f"Identity-channel route mismatch in {context}: {observed} != {expected[condition]}"
        )


def _load_generations(
    generation_manifest: Path,
    run_dir: Path,
    identities: Sequence[str],
    prompts: Sequence[str],
    seeds: Sequence[int],
) -> dict[tuple[str, str, int, str], dict[str, Any]]:
    rows = _read_records(generation_manifest, ("results", "generations", "records"))
    if len(rows) != EXPECTED_GENERATIONS:
        raise GroupedHumanEvalError(
            f"Frozen v2 design requires the complete {EXPECTED_GENERATIONS}-row generation manifest; "
            f"observed {len(rows)}"
        )
    relevant_conditions = {value for pair in CONTRASTS.values() for value in pair}
    expected_keys = {
        (identity, prompt, seed, condition)
        for identity in identities
        for prompt in prompts
        for seed in seeds
        for condition in relevant_conditions
    }
    by_key: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    for index, row in enumerate(rows, start=1):
        context = f"generation manifest row {index}"
        if str(row.get("status")) != "complete":
            raise GroupedHumanEvalError(f"{context} is not complete")
        identity = _required_text(row.get("identity_id"), f"{context} identity_id")
        prompt_id = _required_text(row.get("prompt_id"), f"{context} prompt_id")
        base_seed = _required_int(row.get("base_seed"), f"{context} base_seed")
        condition = _required_text(row.get("condition"), f"{context} condition")
        if condition not in relevant_conditions:
            continue
        key = (identity, prompt_id, base_seed, condition)
        if key in by_key:
            raise GroupedHumanEvalError(f"Duplicate generation cell {key!r}")
        _validate_generation_route(row, context)
        if (
            _required_int(row.get("width"), f"{context} width") != 1024
            or _required_int(row.get("height"), f"{context} height") != 1024
        ):
            raise GroupedHumanEvalError(f"{context} is not a 1024x1024 generation")
        output_relative_path = _required_text(
            row.get("output_relative_path"), f"{context} output_relative_path"
        )
        output_path = _resolve_inside(run_dir, output_relative_path, f"{context} generated image")
        expected_sha256 = _required_text(
            row.get("output_sha256"), f"{context} output_sha256"
        ).lower()
        observed_sha256 = _sha256_file(output_path)
        if observed_sha256 != expected_sha256:
            raise GroupedHumanEvalError(
                f"Output SHA256 mismatch for {row.get('sample_id')}: "
                f"{observed_sha256} != {expected_sha256}"
            )
        with Image.open(output_path) as image:
            if image.size != (1024, 1024):
                raise GroupedHumanEvalError(
                    f"Generated image is not 1024x1024: {output_path} has {image.size}"
                )
            image.verify()
        by_key[key] = {
            "sample_id": _required_text(row.get("sample_id"), f"{context} sample_id"),
            "identity_id": identity,
            "donor_identity_id": _required_text(
                row.get("donor_identity_id"), f"{context} donor_identity_id"
            ),
            "condition": condition,
            "prompt_id": prompt_id,
            "base_seed": base_seed,
            "effective_seed": _required_int(row.get("effective_seed"), f"{context} effective_seed"),
            "latent_sha256": _required_text(row.get("latent_sha256"), f"{context} latent_sha256"),
            "output_relative_path": output_relative_path,
            "output_path": output_path,
            "output_sha256": observed_sha256,
        }
    if set(by_key) != expected_keys:
        missing = sorted(expected_keys - set(by_key))
        unexpected = sorted(set(by_key) - expected_keys)
        raise GroupedHumanEvalError(
            f"Relevant generation matrix mismatch: missing={missing}, unexpected={unexpected}"
        )
    unique_paths = {record["output_path"] for record in by_key.values()}
    if len(unique_paths) != EXPECTED_RELEVANT_GENERATED_IMAGES:
        raise GroupedHumanEvalError(
            "The three frozen contrasts must reference exactly "
            f"{EXPECTED_RELEVANT_GENERATED_IMAGES} unique generated images; "
            f"observed {len(unique_paths)}"
        )
    return by_key


def _build_pairs(
    generations: Mapping[tuple[str, str, int, str], Mapping[str, Any]],
    references: Mapping[str, tuple[dict[str, Any], ...]],
    identities: Sequence[str],
    prompts: Sequence[str],
    seeds: Sequence[int],
    seed: int,
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for contrast_code, (focal_condition, comparator_condition) in CONTRASTS.items():
        for identity_index, identity in enumerate(identities):
            for prompt_index, prompt_id in enumerate(prompts):
                for seed_index, base_seed in enumerate(seeds):
                    focal = generations[(identity, prompt_id, base_seed, focal_condition)]
                    comparator = generations[(identity, prompt_id, base_seed, comparator_condition)]
                    for field in (
                        "identity_id",
                        "donor_identity_id",
                        "prompt_id",
                        "base_seed",
                        "effective_seed",
                        "latent_sha256",
                    ):
                        if focal[field] != comparator[field]:
                            raise GroupedHumanEvalError(
                                f"Unpaired {field} in {contrast_code}/{identity}/{prompt_id}/"
                                f"{base_seed}: {focal[field]!r} != {comparator[field]!r}"
                            )
                    block_index = prompt_index ^ seed_index
                    if contrast_code == "P1":
                        # P0 and P1 both contain k4_diverse. Sending P1 to the
                        # complementary content block prevents a participant from
                        # seeing the same generated image twice in one form.
                        block_index ^= 1
                    content_block = CONTENT_BLOCKS[block_index]
                    pair_id = _opaque_id(
                        "pair",
                        seed,
                        contrast_code,
                        identity,
                        prompt_id,
                        base_seed,
                    )
                    pairs.append(
                        {
                            "pair_id": pair_id,
                            "content_block": content_block,
                            "contrast_code": contrast_code,
                            "focal_condition": focal_condition,
                            "comparator_condition": comparator_condition,
                            "identity_id": identity,
                            "identity_index": identity_index,
                            "donor_identity_id": focal["donor_identity_id"],
                            "prompt_id": prompt_id,
                            "prompt_index": prompt_index,
                            "base_seed": base_seed,
                            "seed_index": seed_index,
                            "effective_seed": focal["effective_seed"],
                            "latent_sha256": focal["latent_sha256"],
                            "references": references[identity],
                            "focal": focal,
                            "comparator": comparator,
                        }
                    )
    pair_ids = [str(pair["pair_id"]) for pair in pairs]
    if len(pairs) != EXPECTED_UNIQUE_PAIRS or len(pair_ids) != len(set(pair_ids)):
        raise GroupedHumanEvalError(
            f"Expected {EXPECTED_UNIQUE_PAIRS} unique pairs, observed {len(set(pair_ids))}"
        )
    block_counts = Counter(str(pair["content_block"]) for pair in pairs)
    if block_counts != Counter({"A": 48, "B": 48}):
        raise GroupedHumanEvalError(f"XOR content-block counts are not 48/48: {block_counts}")
    for pair in pairs:
        block_index = int(pair["prompt_index"]) ^ int(pair["seed_index"])
        if pair["contrast_code"] == "P1":
            block_index ^= 1
        expected_block = CONTENT_BLOCKS[block_index]
        if pair["content_block"] != expected_block:
            raise AssertionError("Prompt-index XOR seed-index assignment changed")
    return pairs


def _anonymous_png(source: Path, media_dir: Path, html_parent: Path) -> tuple[str, str]:
    """Decode and re-encode a source as metadata-free RGB PNG with an opaque filename."""
    with Image.open(source) as image:
        rgb = image.convert("RGB")
        buffer = io.BytesIO()
        rgb.save(buffer, format="PNG", optimize=False, compress_level=9)
    payload = buffer.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    destination = media_dir / f"img_{digest[:24]}.png"
    media_dir.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _sha256_file(destination) != digest:
            raise RuntimeError(f"Opaque media collision at {destination}")
    else:
        temporary = destination.with_suffix(".png.tmp")
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    relative = Path(os.path.relpath(destination, html_parent)).as_posix()
    return quote(relative, safe="/"), digest


def _json_for_script(value: Any) -> str:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return (
        serialized.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def render_grouped_pairwise_html(study: Mapping[str, Any]) -> str:
    payload = _json_for_script(study)
    template = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Blinded face identity-likeness evaluation</title>
  <style>
    :root { color-scheme: light dark; font-family: Inter, system-ui, sans-serif; }
    body { margin: 0; background: #111827; color: #f9fafb; }
    main { max-width: 1180px; margin: auto; padding: 24px; }
    .topbar, .nav, .status, .special { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
    .topbar, .nav { justify-content: space-between; }
    .panel { margin-top: 16px; background: #1f2937; border: 1px solid #374151; border-radius: 14px; padding: 18px; }
    .references { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin: 12px 0 20px; }
    .options { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }
    img { display: block; width: 100%; max-height: 500px; object-fit: contain; background: #0b1020; border-radius: 10px; }
    .references img { max-height: 230px; }
    .choice { border: 2px solid transparent; border-radius: 12px; padding: 10px; background: #111827; }
    .choice.selected { border-color: #60a5fa; box-shadow: 0 0 0 2px #1d4ed8; }
    .label { font-size: 1.1rem; font-weight: 700; margin: 8px 0; }
    button, input { font: inherit; }
    button { border: 1px solid #6b7280; border-radius: 8px; padding: 9px 14px; cursor: pointer; background: #374151; color: #fff; }
    button:hover, button:focus-visible { background: #4b5563; outline: 2px solid #93c5fd; }
    button.primary { background: #2563eb; border-color: #3b82f6; }
    button.selected { background: #1d4ed8; border-color: #93c5fd; }
    button:disabled { opacity: .45; cursor: default; }
    input { border: 1px solid #6b7280; border-radius: 7px; padding: 8px; background: #111827; color: inherit; }
    .special { justify-content: center; margin: 20px 0 4px; }
    .nav { margin-top: 18px; }
    .muted { color: #cbd5e1; }
    .warning { color: #fbbf24; min-height: 1.2em; }
    progress { width: min(420px, 100%); height: 14px; }
    h1, h2, p { margin-top: 0; }
    @media (max-width: 700px) { main { padding: 12px; } .panel { padding: 12px; } .options { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
<main>
  <div class="topbar">
    <div>
      <h1 id="study-title"></h1>
      <div class="status">
        <span id="form-label"></span>
        <span id="progress-text"></span>
        <progress id="progress" max="1" value="0"></progress>
      </div>
    </div>
    <label>Participant code <input id="participant" required autocomplete="off" maxlength="80" placeholder="required"></label>
  </div>
  <p class="muted">Judge only how much the face resembles the reference person. Ignore pose, expression, lighting, background, style, and overall image quality.</p>
  <p id="warning" class="warning" role="status"></p>
  <section id="trial" class="panel" aria-live="polite"></section>
  <div class="nav">
    <button id="previous" type="button">Previous</button>
    <span id="position" class="muted"></span>
    <button id="next" type="button">Next</button>
    <button id="download" class="primary" type="button">Download responses JSON</button>
  </div>
</main>
<script id="study-data" type="application/json">__STUDY_JSON__</script>
<script>
(() => {
  "use strict";
  const study = JSON.parse(document.getElementById("study-data").textContent);
  const storageKey = `identity-layers-grouped-human-eval:${study.study_id}:${study.form_id}`;
  const trialIds = new Set(study.trials.map(t => t.trial_id));
  let state = {participant_id: "", responses: {}, current: 0};
  const participant = document.getElementById("participant");
  const warning = document.getElementById("warning");

  function loadState() {
    try {
      const saved = JSON.parse(localStorage.getItem(storageKey) || "null");
      if (!saved || typeof saved !== "object") return;
      state.participant_id = typeof saved.participant_id === "string" ? saved.participant_id : "";
      state.current = Number.isInteger(saved.current) ? Math.max(0, Math.min(saved.current, study.trials.length - 1)) : 0;
      if (saved.responses && typeof saved.responses === "object") {
        for (const [trialId, response] of Object.entries(saved.responses)) {
          if (trialIds.has(trialId) && response && typeof response.selection === "string") state.responses[trialId] = response;
        }
      }
    } catch (error) {
      warning.textContent = "Browser storage is unavailable; download responses regularly.";
    }
  }

  function saveState() {
    try { localStorage.setItem(storageKey, JSON.stringify(state)); }
    catch (error) { warning.textContent = "Responses could not be stored locally."; }
  }

  function image(src, alt) {
    const node = document.createElement("img");
    node.src = src;
    node.alt = alt;
    node.decoding = "async";
    return node;
  }

  function select(value) {
    const trial = study.trials[state.current];
    const allowed = new Set([...trial.options.map(option => option.option_id), ...trial.special_responses]);
    if (!allowed.has(value)) return;
    state.responses[trial.trial_id] = {selection: value, updated_at: new Date().toISOString()};
    saveState();
    render();
  }

  function choice(option, label, selected) {
    const card = document.createElement("div");
    card.className = `choice${selected ? " selected" : ""}`;
    card.appendChild(image(option.src, `Generated option ${label}`));
    const heading = document.createElement("div");
    heading.className = "label";
    heading.textContent = label;
    card.appendChild(heading);
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = `Choose ${label}`;
    button.className = selected ? "selected" : "";
    button.addEventListener("click", () => select(option.option_id));
    card.appendChild(button);
    return card;
  }

  function render() {
    const answered = Object.keys(state.responses).length;
    const trial = study.trials[state.current];
    const selected = state.responses[trial.trial_id]?.selection || null;
    document.getElementById("study-title").textContent = study.title;
    document.getElementById("form-label").textContent = `Form ${study.form_id}`;
    document.getElementById("progress-text").textContent = `${answered} / ${study.trials.length} answered`;
    const progress = document.getElementById("progress");
    progress.max = study.trials.length;
    progress.value = answered;
    document.getElementById("position").textContent = `Question ${state.current + 1} of ${study.trials.length}`;
    document.getElementById("previous").disabled = state.current === 0;
    document.getElementById("next").disabled = state.current === study.trials.length - 1;
    document.getElementById("download").disabled = (
      !participant.value.trim() || answered !== study.trials.length
    );
    const container = document.getElementById("trial");
    container.replaceChildren();
    const title = document.createElement("h2");
    title.textContent = "Which generated face looks more like the reference person?";
    container.appendChild(title);
    const references = document.createElement("div");
    references.className = "references";
    trial.reference_images.forEach((item, index) => references.appendChild(image(item.src, `Identity reference ${index + 1}`)));
    container.appendChild(references);
    const options = document.createElement("div");
    options.className = "options";
    trial.options.forEach((option, index) => options.appendChild(choice(option, index === 0 ? "A" : "B", selected === option.option_id)));
    container.appendChild(options);
    const special = document.createElement("div");
    special.className = "special";
    for (const value of trial.special_responses) {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = value === "tie" ? "Tie / equally similar" : "Unjudgeable";
      button.className = selected === value ? "selected" : "";
      button.addEventListener("click", () => select(value));
      special.appendChild(button);
    }
    container.appendChild(special);
  }

  function downloadResponses() {
    state.participant_id = participant.value.trim();
    if (!state.participant_id) {
      warning.textContent = "Enter the required participant code before downloading.";
      render();
      return;
    }
    const responses = study.trials.filter(t => state.responses[t.trial_id]).map(t => ({trial_id: t.trial_id, ...state.responses[t.trial_id]}));
    if (responses.length !== study.trials.length) {
      warning.textContent = "Answer all 48 questions before downloading the final response.";
      render();
      return;
    }
    const output = {
      schema_version: "identity-layers-grouped-human-response/v2",
      study_id: study.study_id,
      form_id: study.form_id,
      participant_id: state.participant_id,
      completed: true,
      trial_count: study.trials.length,
      responses
    };
    const blob = new Blob([JSON.stringify(output, null, 2) + "\n"], {type: "application/json"});
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    const safeParticipant = state.participant_id.replace(/[^A-Za-z0-9_.-]+/g, "_");
    link.download = `${study.study_id}-${study.form_id}-${safeParticipant}-responses.json`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(link.href);
  }

  document.getElementById("previous").addEventListener("click", () => { state.current -= 1; saveState(); render(); });
  document.getElementById("next").addEventListener("click", () => { state.current += 1; saveState(); render(); });
  document.getElementById("download").addEventListener("click", downloadResponses);
  participant.addEventListener("input", () => { state.participant_id = participant.value; saveState(); render(); });
  document.addEventListener("keydown", event => {
    if (event.target === participant) return;
    const trial = study.trials[state.current];
    if (event.key === "1" || event.key === "2") select(trial.options[Number(event.key) - 1].option_id);
    else if (event.key.toLowerCase() === "t") select("tie");
    else if (event.key.toLowerCase() === "u") select("unjudgeable");
    else if (event.key === "ArrowLeft" && state.current > 0) { state.current -= 1; saveState(); render(); }
    else if (event.key === "ArrowRight" && state.current < study.trials.length - 1) { state.current += 1; saveState(); render(); }
  });

  loadState();
  participant.value = state.participant_id;
  render();
})();
</script>
</body>
</html>
"""
    html = template.replace("__STUDY_JSON__", payload)
    if "data:image" in html.lower():
        raise AssertionError("Grouped human-evaluation HTML must not embed image data URIs")
    return html


def _base_focal_left(
    pairs: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], bool]:
    assignments: dict[tuple[str, str], bool] = {}
    contrast_indices = {code: index for index, code in enumerate(CONTRASTS)}
    for content_block_index, content_block in enumerate(CONTENT_BLOCKS):
        for contrast_code, contrast_index in contrast_indices.items():
            group = [
                pair
                for pair in pairs
                if pair["content_block"] == content_block and pair["contrast_code"] == contrast_code
            ]
            if len(group) != 16:
                raise GroupedHumanEvalError(
                    f"Expected 16 pairs for {content_block}/{contrast_code}; observed {len(group)}"
                )
            for pair in group:
                parity = (int(pair["identity_index"]) + contrast_index + content_block_index) % 2
                assignments[(content_block, str(pair["pair_id"]))] = bool(
                    int(pair["prompt_index"]) ^ parity
                )
    return assignments


def _stage_assets(
    pairs: Sequence[dict[str, Any]],
    media_dir: Path,
    participant_dir: Path,
) -> tuple[dict[Path, tuple[str, str]], dict[str, str]]:
    source_paths: set[Path] = set()
    for pair in pairs:
        source_paths.update(reference["source_path"] for reference in pair["references"])
        source_paths.add(pair["focal"]["output_path"])
        source_paths.add(pair["comparator"]["output_path"])
    if len(source_paths) != EXPECTED_REFERENCE_IMAGES + EXPECTED_RELEVANT_GENERATED_IMAGES:
        raise GroupedHumanEvalError(
            "Unexpected unique source-media count: "
            f"{len(source_paths)} != "
            f"{EXPECTED_REFERENCE_IMAGES + EXPECTED_RELEVANT_GENERATED_IMAGES}"
        )
    staged: dict[Path, tuple[str, str]] = {}
    media_hashes: dict[str, str] = {}
    for source in sorted(source_paths, key=lambda path: path.as_posix()):
        href, digest = _anonymous_png(source, media_dir, participant_dir)
        staged[source] = (href, digest)
        previous = media_hashes.setdefault(href, digest)
        if previous != digest:
            raise AssertionError("One public media path mapped to multiple hashes")
    return staged, dict(sorted(media_hashes.items()))


def _public_reference_images(
    pair: Mapping[str, Any], staged: Mapping[Path, tuple[str, str]]
) -> list[dict[str, str]]:
    return [{"src": staged[reference["source_path"]][0]} for reference in pair["references"]]


def _build_forms(
    pairs: Sequence[dict[str, Any]],
    staged: Mapping[Path, tuple[str, str]],
    all_media_hashes: Mapping[str, str],
    *,
    study_id: str,
    title: str,
    seed: int,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    base_sides = _base_focal_left(pairs)
    public_forms: dict[str, dict[str, Any]] = {}
    private_trials: list[dict[str, Any]] = []
    for content_block in CONTENT_BLOCKS:
        content_pairs = [pair for pair in pairs if pair["content_block"] == content_block]
        if len(content_pairs) != EXPECTED_PAIRS_PER_CONTENT_BLOCK:
            raise AssertionError("Content block no longer contains 48 pairs")
        for copy_index in range(DISPLAY_COPIES_PER_BLOCK):
            form_id = f"{content_block}{copy_index + 1:02d}"
            public_trials: list[dict[str, Any]] = []
            for pair in content_pairs:
                focal_left = base_sides[(content_block, str(pair["pair_id"]))]
                if copy_index % 2 == 1:
                    focal_left = not focal_left
                trial_id = _opaque_id("trial", seed, form_id, pair["pair_id"])
                focal_option_id = _opaque_id("option", seed, form_id, pair["pair_id"], "focal")
                comparator_option_id = _opaque_id(
                    "option", seed, form_id, pair["pair_id"], "comparator"
                )
                options = [
                    {
                        "option_id": focal_option_id,
                        "src": staged[pair["focal"]["output_path"]][0],
                    },
                    {
                        "option_id": comparator_option_id,
                        "src": staged[pair["comparator"]["output_path"]][0],
                    },
                ]
                if not focal_left:
                    options.reverse()
                public_trials.append(
                    {
                        "trial_id": trial_id,
                        "kind": "pairwise_identity_likeness",
                        "reference_images": _public_reference_images(pair, staged),
                        "options": options,
                        "special_responses": ["tie", "unjudgeable"],
                    }
                )
                private_trials.append(
                    {
                        "trial_id": trial_id,
                        "pair_id": pair["pair_id"],
                        "form_id": form_id,
                        "content_block": content_block,
                        "display_copy": copy_index + 1,
                        "contrast_code": pair["contrast_code"],
                        "focal_condition": pair["focal_condition"],
                        "comparator_condition": pair["comparator_condition"],
                        "focal_side": "left" if focal_left else "right",
                        "identity_id": pair["identity_id"],
                        "identity_index": pair["identity_index"],
                        "donor_identity_id": pair["donor_identity_id"],
                        "prompt_id": pair["prompt_id"],
                        "prompt_index": pair["prompt_index"],
                        "base_seed": pair["base_seed"],
                        "seed_index": pair["seed_index"],
                        "effective_seed": pair["effective_seed"],
                        "latent_sha256": pair["latent_sha256"],
                        "option_roles": {
                            focal_option_id: "focal",
                            comparator_option_id: "comparator",
                        },
                        "option_conditions": {
                            focal_option_id: pair["focal_condition"],
                            comparator_option_id: pair["comparator_condition"],
                        },
                        "source_result_ids": {
                            focal_option_id: pair["focal"]["sample_id"],
                            comparator_option_id: pair["comparator"]["sample_id"],
                        },
                        "source_output_relative_paths": {
                            focal_option_id: pair["focal"]["output_relative_path"],
                            comparator_option_id: pair["comparator"]["output_relative_path"],
                        },
                        "source_output_sha256": {
                            focal_option_id: pair["focal"]["output_sha256"],
                            comparator_option_id: pair["comparator"]["output_sha256"],
                        },
                        "reference_sample_ids": [
                            reference["sample_id"] for reference in pair["references"]
                        ],
                    }
                )
            order_rng = random.Random(int(_stable_digest(seed, "trial-order", form_id)[:16], 16))
            order_rng.shuffle(public_trials)
            used_hrefs = {
                image["src"]
                for trial in public_trials
                for image in [*trial["reference_images"], *trial["options"]]
            }
            public_forms[form_id] = {
                "schema_version": PUBLIC_SCHEMA_VERSION,
                "response_schema_version": RESPONSE_SCHEMA_VERSION,
                "study_id": study_id,
                "form_id": form_id,
                "title": title,
                "task": "pairwise_identity_likeness",
                "trial_count": len(public_trials),
                "randomization": {
                    "seed": seed,
                    "algorithm": (
                        "balanced complementary prompt/seed content blocks; five display "
                        "copies; strict within-form/contrast side balance; sha256-derived "
                        "trial order"
                    ),
                },
                "media": {
                    "policy": "metadata-stripped-rgb-png-with-opaque-content-hash-filenames",
                    "sha256": {href: all_media_hashes[href] for href in sorted(used_hrefs)},
                },
                "trials": public_trials,
            }
    return public_forms, private_trials


def _validate_forms(
    pairs: Sequence[Mapping[str, Any]],
    public_forms: Mapping[str, Mapping[str, Any]],
    private_trials: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected_form_ids = {
        f"{content}{copy_index:02d}"
        for content in CONTENT_BLOCKS
        for copy_index in range(1, DISPLAY_COPIES_PER_BLOCK + 1)
    }
    if set(public_forms) != expected_form_ids:
        raise GroupedHumanEvalError("The frozen ten-form set is incomplete")
    if len(private_trials) != EXPECTED_DISPLAY_TRIALS:
        raise GroupedHumanEvalError(
            f"Expected {EXPECTED_DISPLAY_TRIALS} display trials; observed {len(private_trials)}"
        )
    pair_counts = Counter(str(trial["pair_id"]) for trial in private_trials)
    if set(pair_counts.values()) != {DISPLAY_COPIES_PER_BLOCK}:
        raise GroupedHumanEvalError("Every unique pair must occur in exactly five forms")
    if set(pair_counts) != {str(pair["pair_id"]) for pair in pairs}:
        raise GroupedHumanEvalError("Display trials do not cover the frozen pair set exactly")

    form_summaries: dict[str, Any] = {}
    for form_id in sorted(public_forms):
        private = [trial for trial in private_trials if trial["form_id"] == form_id]
        public = public_forms[form_id]
        if len(private) != EXPECTED_TRIALS_PER_FORM or len(public["trials"]) != 48:
            raise GroupedHumanEvalError(f"{form_id} must contain exactly 48 trials")
        if len({str(trial["pair_id"]) for trial in private}) != 48:
            raise GroupedHumanEvalError(f"{form_id} repeats a unique pair")
        candidate_paths = [
            str(path)
            for trial in private
            for path in trial["source_output_relative_paths"].values()
        ]
        candidate_hashes = [
            str(digest) for trial in private for digest in trial["source_output_sha256"].values()
        ]
        if len(candidate_paths) != 96 or len(set(candidate_paths)) != 96:
            raise GroupedHumanEvalError(f"{form_id} repeats a candidate generated-image path")
        if len(candidate_hashes) != 96 or len(set(candidate_hashes)) != 96:
            raise GroupedHumanEvalError(f"{form_id} repeats a candidate generated-image SHA256")
        contrast_counts = Counter(str(trial["contrast_code"]) for trial in private)
        if contrast_counts != Counter({"P0": 16, "P1": 16, "P3": 16}):
            raise GroupedHumanEvalError(
                f"{form_id} contrast counts are not 16/16/16: {contrast_counts}"
            )
        side_counts: dict[str, dict[str, int]] = {}
        for contrast_code in CONTRASTS:
            sides = Counter(
                str(trial["focal_side"])
                for trial in private
                if trial["contrast_code"] == contrast_code
            )
            if sides != Counter({"left": 8, "right": 8}):
                raise GroupedHumanEvalError(
                    f"{form_id}/{contrast_code} focal-side counts are not 8/8: {sides}"
                )
            side_counts[contrast_code] = dict(sorted(sides.items()))
        total_sides = Counter(str(trial["focal_side"]) for trial in private)
        if total_sides != Counter({"left": 24, "right": 24}):
            raise GroupedHumanEvalError(
                f"{form_id} overall focal-side counts are not 24/24: {total_sides}"
            )
        identity_side_counts: dict[str, dict[str, int]] = {}
        for identity in sorted({str(trial["identity_id"]) for trial in private}):
            sides = Counter(
                str(trial["focal_side"]) for trial in private if trial["identity_id"] == identity
            )
            if sides != Counter({"left": 3, "right": 3}):
                raise GroupedHumanEvalError(
                    f"{form_id}/{identity} focal-side counts are not 3/3: {sides}"
                )
            identity_side_counts[identity] = dict(sorted(sides.items()))
        prompt_side_counts: dict[str, dict[str, int]] = {}
        for prompt_id in sorted({str(trial["prompt_id"]) for trial in private}):
            sides = Counter(
                str(trial["focal_side"]) for trial in private if trial["prompt_id"] == prompt_id
            )
            if sides != Counter({"left": 12, "right": 12}):
                raise GroupedHumanEvalError(
                    f"{form_id}/{prompt_id} focal-side counts are not 12/12: {sides}"
                )
            prompt_side_counts[prompt_id] = dict(sorted(sides.items()))
        seed_side_counts: dict[str, dict[str, int]] = {}
        for base_seed in sorted({int(trial["base_seed"]) for trial in private}):
            sides = Counter(
                str(trial["focal_side"]) for trial in private if trial["base_seed"] == base_seed
            )
            if sides != Counter({"left": 12, "right": 12}):
                raise GroupedHumanEvalError(
                    f"{form_id}/base-seed-{base_seed} focal-side counts are not 12/12: {sides}"
                )
            seed_side_counts[str(base_seed)] = dict(sorted(sides.items()))
        identity_contrast_side_counts: dict[str, dict[str, int]] = {}
        for identity in sorted({str(trial["identity_id"]) for trial in private}):
            for contrast_code in CONTRASTS:
                sides = Counter(
                    str(trial["focal_side"])
                    for trial in private
                    if trial["identity_id"] == identity and trial["contrast_code"] == contrast_code
                )
                if sides != Counter({"left": 1, "right": 1}):
                    raise GroupedHumanEvalError(
                        f"{form_id}/{identity}/{contrast_code} sides are not 1/1: {sides}"
                    )
                identity_contrast_side_counts[f"{identity}/{contrast_code}"] = dict(
                    sorted(sides.items())
                )
        public_ids = [str(trial["trial_id"]) for trial in public["trials"]]
        private_ids = [str(trial["trial_id"]) for trial in private]
        if len(public_ids) != len(set(public_ids)) or set(public_ids) != set(private_ids):
            raise GroupedHumanEvalError(f"{form_id} public/private trial IDs disagree")
        for trial in public["trials"]:
            if len(trial["reference_images"]) != 3 or len(trial["options"]) != 2:
                raise GroupedHumanEvalError(
                    f"{form_id}/{trial['trial_id']} does not have 3 references and 2 options"
                )
        form_summaries[form_id] = {
            "trial_count": 48,
            "unique_candidate_generated_paths": len(set(candidate_paths)),
            "unique_candidate_generated_sha256": len(set(candidate_hashes)),
            "contrast_counts": dict(sorted(contrast_counts.items())),
            "overall_focal_side_counts": dict(sorted(total_sides.items())),
            "focal_side_counts": side_counts,
            "identity_focal_side_counts": identity_side_counts,
            "prompt_focal_side_counts": prompt_side_counts,
            "base_seed_focal_side_counts": seed_side_counts,
            "identity_contrast_focal_side_counts": identity_contrast_side_counts,
        }
    return {
        "unique_pair_count": len(pair_counts),
        "content_block_counts": dict(
            sorted(Counter(str(pair["content_block"]) for pair in pairs).items())
        ),
        "display_form_count": len(public_forms),
        "display_trial_count": len(private_trials),
        "display_count_per_unique_pair": DISPLAY_COPIES_PER_BLOCK,
        "forms": form_summaries,
    }


def _assert_public_blinding(
    public_forms: Mapping[str, Mapping[str, Any]],
    *,
    identities: Sequence[str],
    prompts: Sequence[str],
    pairs: Sequence[Mapping[str, Any]],
) -> None:
    forbidden = {
        *identities,
        *prompts,
        *CONTRASTS,
        *(condition for values in CONTRASTS.values() for condition in values),
        *(str(pair["focal"]["sample_id"]) for pair in pairs),
        *(str(pair["comparator"]["sample_id"]) for pair in pairs),
        *(str(reference["sample_id"]) for pair in pairs for reference in pair["references"]),
    }
    for form_id, study in public_forms.items():
        serialized = json.dumps(study, ensure_ascii=False, sort_keys=True)
        leaked = sorted(token for token in forbidden if token and token in serialized)
        if leaked:
            raise GroupedHumanEvalError(f"Public form {form_id} leaks private labels: {leaked[:5]}")
        if "data:image" in serialized.lower():
            raise GroupedHumanEvalError(f"Public form {form_id} embeds image bytes")


def build_grouped_human_eval(
    *,
    run_dir: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
    config_path: str | Path | None = None,
    data_root: str | Path | None = None,
    seed: int = 20260904,
    title: str = "Blinded face identity-likeness evaluation",
) -> GroupedHumanEvalBuild:
    """Build the frozen v2 A/B grouped pairwise study without changing v1 artifacts."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise GroupedHumanEvalError("seed must be an integer")
    run = Path(run_dir).resolve()
    manifest = Path(manifest_path).resolve()
    output = Path(output_dir).resolve()
    config = (
        (run / "config.resolved.yaml").resolve()
        if config_path is None
        else Path(config_path).resolve()
    )
    data = manifest.parents[1].resolve() if data_root is None else Path(data_root).resolve()
    generation_manifest = run / "generation_manifest.jsonl"
    if not run.is_dir():
        raise FileNotFoundError(run)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite grouped human-evaluation output: {output}")

    resolved_config = _load_config(config)
    identities, prompts, seeds = _ordered_design_axes(resolved_config)
    references = _load_original_references(manifest, data, identities)
    generations = _load_generations(generation_manifest, run, identities, prompts, seeds)
    pairs = _build_pairs(generations, references, identities, prompts, seeds, seed)

    participant_dir = output / "participant"
    media_dir = participant_dir / "media"
    private_dir = output / "private"
    responses_dir = output / "responses"
    participant_dir.mkdir(parents=True, exist_ok=False)
    private_dir.mkdir(parents=True, exist_ok=False)
    responses_dir.mkdir(parents=True, exist_ok=False)
    staged, media_hashes = _stage_assets(pairs, media_dir, participant_dir)
    study_id = _opaque_id(
        "study",
        seed,
        _sha256_file(config),
        _sha256_file(manifest),
        _sha256_file(generation_manifest),
        [pair["pair_id"] for pair in pairs],
        length=20,
    )
    public_forms, private_trials = _build_forms(
        pairs,
        staged,
        media_hashes,
        study_id=study_id,
        title=title,
        seed=seed,
    )
    validation = _validate_forms(pairs, public_forms, private_trials)
    _assert_public_blinding(
        public_forms,
        identities=identities,
        prompts=prompts,
        pairs=pairs,
    )

    form_html_paths: list[Path] = []
    form_json_paths: list[Path] = []
    public_file_hashes: dict[str, str] = {}
    for form_id in sorted(public_forms):
        study = public_forms[form_id]
        json_path = participant_dir / f"{form_id}.blinded.json"
        html_path = participant_dir / f"{form_id}.html"
        _write_text_atomic(
            json_path,
            json.dumps(study, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        )
        _write_text_atomic(html_path, render_grouped_pairwise_html(study))
        form_json_paths.append(json_path)
        form_html_paths.append(html_path)
        public_file_hashes[json_path.relative_to(output).as_posix()] = _sha256_file(json_path)
        public_file_hashes[html_path.relative_to(output).as_posix()] = _sha256_file(html_path)

    pair_records = []
    for pair in pairs:
        pair_records.append(
            {
                key: pair[key]
                for key in (
                    "pair_id",
                    "content_block",
                    "contrast_code",
                    "focal_condition",
                    "comparator_condition",
                    "identity_id",
                    "identity_index",
                    "donor_identity_id",
                    "prompt_id",
                    "prompt_index",
                    "base_seed",
                    "seed_index",
                    "effective_seed",
                    "latent_sha256",
                )
            }
            | {
                "reference_sample_ids": [
                    reference["sample_id"] for reference in pair["references"]
                ],
                "reference_source_relative_paths": [
                    reference["source_relative_path"] for reference in pair["references"]
                ],
                "reference_source_sha256": [
                    reference["source_sha256"] for reference in pair["references"]
                ],
                "focal_result_id": pair["focal"]["sample_id"],
                "focal_output_relative_path": pair["focal"]["output_relative_path"],
                "focal_output_sha256": pair["focal"]["output_sha256"],
                "comparator_result_id": pair["comparator"]["sample_id"],
                "comparator_output_relative_path": pair["comparator"]["output_relative_path"],
                "comparator_output_sha256": pair["comparator"]["output_sha256"],
            }
        )
    answer_key = {
        "schema_version": ANSWER_KEY_SCHEMA_VERSION,
        "study_id": study_id,
        "design": {
            "task": "pairwise_identity_likeness",
            "contrasts": {
                code: {"focal": values[0], "comparator": values[1]}
                for code, values in CONTRASTS.items()
            },
            "content_block_rule": (
                "P0/P3 use prompt_index XOR seed_index: 0=A, 1=B; "
                "P1 uses the complementary block to prevent repeated k4_diverse "
                "exposure within a form"
            ),
            "content_blocks": list(CONTENT_BLOCKS),
            "display_copies_per_content_block": DISPLAY_COPIES_PER_BLOCK,
            "forms": sorted(public_forms),
            "trials_per_form": EXPECTED_TRIALS_PER_FORM,
            "references_per_trial": len(REFERENCE_CONDITIONS),
            "reference_path_column": "source_relative_path",
            "reference_conditions": list(REFERENCE_CONDITIONS),
            "generated_resolution": [1024, 1024],
            "randomization_seed": seed,
        },
        "sources": {
            "run_dir": str(run),
            "generation_manifest": str(generation_manifest),
            "generation_manifest_sha256": _sha256_file(generation_manifest),
            "reference_manifest": str(manifest),
            "reference_manifest_sha256": _sha256_file(manifest),
            "resolved_config": str(config),
            "resolved_config_sha256": _sha256_file(config),
        },
        "validation": validation,
        "pairs": pair_records,
        "trials": private_trials,
    }
    answer_key_path = private_dir / "answer_key.json"
    _write_text_atomic(
        answer_key_path,
        json.dumps(answer_key, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )

    for href, expected_digest in media_hashes.items():
        media_path = (participant_dir / href).resolve()
        if _sha256_file(media_path) != expected_digest:
            raise GroupedHumanEvalError(f"Staged media hash mismatch: {href}")
        with Image.open(media_path) as image:
            if image.format != "PNG" or image.mode != "RGB" or len(image.getexif()) != 0:
                raise GroupedHumanEvalError(f"Public media is not metadata-free RGB PNG: {href}")

    build_manifest = {
        "schema_version": BUILD_SCHEMA_VERSION,
        "state": "complete",
        "study_id": study_id,
        "output_dir": str(output),
        "counts": {
            "source_generation_manifest_rows": EXPECTED_GENERATIONS,
            "original_reference_images": EXPECTED_REFERENCE_IMAGES,
            "relevant_unique_generated_images": EXPECTED_RELEVANT_GENERATED_IMAGES,
            "unique_pairs": EXPECTED_UNIQUE_PAIRS,
            "content_blocks": 2,
            "display_forms": EXPECTED_DISPLAY_FORMS,
            "trials_per_form": EXPECTED_TRIALS_PER_FORM,
            "display_trials": EXPECTED_DISPLAY_TRIALS,
            "displays_per_unique_pair": DISPLAY_COPIES_PER_BLOCK,
            "public_media_files": len(media_hashes),
        },
        "validation": validation,
        "files": {
            "answer_key": {
                "path": answer_key_path.relative_to(output).as_posix(),
                "sha256": _sha256_file(answer_key_path),
            },
            "public_forms": dict(sorted(public_file_hashes.items())),
            "public_media": dict(sorted(media_hashes.items())),
        },
    }
    build_manifest_path = output / "build_manifest.json"
    _write_text_atomic(
        build_manifest_path,
        json.dumps(build_manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )
    _write_text_atomic(
        output / "status.json",
        json.dumps(
            {
                "schema_version": BUILD_SCHEMA_VERSION,
                "state": "complete",
                "study_id": study_id,
                "build_manifest": "build_manifest.json",
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
    )
    return GroupedHumanEvalBuild(
        output_dir=output,
        participant_dir=participant_dir,
        media_dir=media_dir,
        answer_key_path=answer_key_path,
        build_manifest_path=build_manifest_path,
        form_html_paths=tuple(form_html_paths),
        form_json_paths=tuple(form_json_paths),
        study_id=study_id,
        unique_pair_count=EXPECTED_UNIQUE_PAIRS,
        display_trial_count=EXPECTED_DISPLAY_TRIALS,
    )


__all__ = [
    "ANSWER_KEY_SCHEMA_VERSION",
    "BUILD_SCHEMA_VERSION",
    "GroupedHumanEvalBuild",
    "GroupedHumanEvalError",
    "PUBLIC_SCHEMA_VERSION",
    "RESPONSE_SCHEMA_VERSION",
    "build_grouped_human_eval",
    "render_grouped_pairwise_html",
]
