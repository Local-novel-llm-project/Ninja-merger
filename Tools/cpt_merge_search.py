#!/usr/bin/env python3
"""Build and optionally run checkpoint merge candidates for CPT search.

This tool is aimed at trajectories like:

- checkpoint-early
- checkpoint-mid
- checkpoint-late

where the checkpoints are full Hugging Face model directories and the useful
signal is mainly in the text stack. For Gemma4 CPT runs, audio and vision
towers can destabilize direct merge experiments, so the generated recipes
default to excluding those towers from the merge while keeping them in the
final model.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import yaml


AUDIO_VISION_EXCLUDE_LAYERS = [
    "model.audio_tower.",
    "model.vision_tower.",
    "model.embed_audio.",
    "model.embed_vision.",
]

DEFAULT_ANGLE_VELOCITIES = [0.25, 0.35, 0.50]
DEFAULT_LATE_ANGLE_VELOCITIES = [0.20, 0.35]
DEFAULT_LONGARC_VELOCITIES = [0.15, 0.25]


@dataclass(frozen = True)
class Candidate:
    key: str
    description: str
    models: list[dict[str, Any]]
    final_model_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument(
        "checkpoints",
        nargs = "*",
        help = "Checkpoint directories to merge. If omitted, use --checkpoint-root.",
    )
    parser.add_argument(
        "--checkpoint-root",
        type = Path,
        default = None,
        help = "Optional directory used with --checkpoint-glob for auto-discovery.",
    )
    parser.add_argument(
        "--checkpoint-glob",
        type = str,
        default = "check-*",
        help = "Glob for checkpoint auto-discovery under --checkpoint-root.",
    )
    parser.add_argument(
        "--base-model",
        type = str,
        default = None,
        help = "Base model id/path. If omitted, infer model_name from the first checkpoint config.",
    )
    parser.add_argument(
        "--output-root",
        type = Path,
        default = Path("generated/cpt_merge_search"),
        help = "Directory where configs, merged models, and eval outputs are written.",
    )
    parser.add_argument(
        "--torch-dtype",
        type = str,
        default = "bfloat16",
        choices = ("bfloat16", "float16", "float32"),
        help = "Torch dtype for NinjaMerger recipes.",
    )
    parser.add_argument(
        "--merge-models-device",
        type = str,
        default = "cpu",
        help = "Device passed to NinjaMerger as --merge_models_device.",
    )
    parser.add_argument(
        "--merge-audio-vision",
        action = "store_true",
        help = "Merge audio/vision towers too. By default they are excluded from the merge.",
    )
    parser.add_argument(
        "--run-merges",
        action = "store_true",
        help = "Run NinjaMerger for each generated candidate.",
    )
    parser.add_argument(
        "--run-eval",
        action = "store_true",
        help = "After merging, evaluate merged models with run_cpt_batch_eval.py.",
    )
    parser.add_argument(
        "--eval-script",
        type = Path,
        default = Path("/home/synome/exveria1015/unsloth-ft/scripts/run_cpt_batch_eval.py"),
        help = "Path to the CPT batch evaluation script.",
    )
    parser.add_argument(
        "--eval-workdir",
        type = Path,
        default = Path("/home/synome/exveria1015/unsloth-ft"),
        help = "Working directory for the CPT batch evaluation script.",
    )
    parser.add_argument(
        "--prompt-file",
        type = Path,
        default = Path("/home/synome/exveria1015/unsloth-ft/prompts/cpt_eval_prompts_ja.jsonl"),
        help = "Prompt file passed to the CPT evaluation script.",
    )
    return parser.parse_args()


def natural_step_key(path: Path) -> tuple[int, str]:
    matches = re.findall(r"(\d+)", path.name)
    if not matches:
        return (-1, path.name)
    return (int(matches[-1]), path.name)


def resolve_checkpoints(args: argparse.Namespace) -> list[Path]:
    checkpoint_paths = [Path(item).expanduser().resolve() for item in args.checkpoints]

    if args.checkpoint_root is not None:
        root = args.checkpoint_root.expanduser().resolve()
        discovered = [item.resolve() for item in root.glob(args.checkpoint_glob) if item.is_dir()]
        checkpoint_paths.extend(discovered)

    deduped: dict[str, Path] = {}
    for path in checkpoint_paths:
        if not path.is_dir():
            raise FileNotFoundError(f"Checkpoint directory not found: {path}")
        if not (path / "config.json").exists():
            raise FileNotFoundError(f"config.json not found under checkpoint: {path}")
        deduped[str(path)] = path

    resolved = sorted(deduped.values(), key = natural_step_key)
    if len(resolved) < 3:
        raise ValueError("At least three checkpoints are required to build merge candidates.")
    return resolved


def infer_base_model(checkpoint: Path) -> str:
    config_path = checkpoint / "config.json"
    config = json.loads(config_path.read_text(encoding = "utf-8"))
    model_name = config.get("model_name")
    if isinstance(model_name, str) and model_name:
        return model_name
    raise KeyError(f"model_name is missing in {config_path}")


def format_velocity_tag(value: float) -> str:
    return str(value).replace(".", "p")


def unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    ordered: list[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(path)
    return ordered


def layer_velocity_profile(global_velocity: float) -> dict[str, float]:
    return {
        "model.language_model.embed_tokens.": round(min(global_velocity * 0.30, 0.16), 4),
        "model.language_model.norm.": round(min(global_velocity * 0.50, 0.25), 4),
        "lm_head.": round(min(global_velocity * 0.40, 0.20), 4),
        "DEFAULT": round(global_velocity, 4),
    }


def recipe_extras(*, merge_audio_vision: bool) -> dict[str, Any]:
    if merge_audio_vision:
        return {}
    return {"exclude_layers": list(AUDIO_VISION_EXCLUDE_LAYERS)}


def build_direct_angle_candidate(
    *,
    name: str,
    description: str,
    left: Path,
    right: list[Path],
    velocity: float,
    merge_audio_vision: bool,
) -> Candidate:
    model = {
        "name": name,
        "target": None,
        "left": [str(left)],
        "right": [str(item) for item in right],
        "operation": "angle_merge",
        "velocities": layer_velocity_profile(velocity),
        "force_merge_single": True,
        "v2s_single_default": "v1",
        **recipe_extras(merge_audio_vision = merge_audio_vision),
    }
    return Candidate(
        key = name,
        description = description,
        models = [model],
        final_model_name = name,
    )


def build_chain_mix_candidate(
    *,
    key: str,
    description: str,
    start: Path,
    mid: Path,
    end: Path,
    first_velocity: float,
    second_velocity: float,
    merge_audio_vision: bool,
) -> Candidate:
    stage1 = {
        "name": f"{key}_stage1",
        "target": None,
        "left": [str(start)],
        "right": [str(mid)],
        "operation": "mix",
        "velocity": first_velocity,
        **recipe_extras(merge_audio_vision = merge_audio_vision),
    }
    stage2_name = f"{key}_final"
    stage2 = {
        "name": stage2_name,
        "target": "recurrent",
        "left": ["recurrent"],
        "right": [str(end)],
        "operation": "mix",
        "velocity": second_velocity,
        **recipe_extras(merge_audio_vision = merge_audio_vision),
    }
    return Candidate(
        key = key,
        description = description,
        models = [stage1, stage2],
        final_model_name = stage2_name,
    )


def build_candidates(checkpoints: list[Path], merge_audio_vision: bool) -> list[Candidate]:
    mid_index = max(min(len(checkpoints) - 3, len(checkpoints) // 2), 1)
    mid = checkpoints[mid_index]
    pre_mid = checkpoints[max(mid_index - 1, 0)]
    late = checkpoints[-2]
    latest = checkpoints[-1]
    longarc_early = checkpoints[1] if len(checkpoints) >= 5 else checkpoints[0]

    mid_right = unique_paths([pre_mid, late, latest])
    mid_right = [path for path in mid_right if path != mid]

    late_right = unique_paths([mid, latest])
    late_right = [path for path in late_right if path != late]

    longarc_right = unique_paths([longarc_early, pre_mid, late, latest])
    longarc_right = [path for path in longarc_right if path != mid]

    candidates: list[Candidate] = []

    for velocity in DEFAULT_ANGLE_VELOCITIES:
        tag = format_velocity_tag(velocity)
        candidates.append(
            build_direct_angle_candidate(
                name = f"cpt_angle_mid_v{tag}",
                description = (
                    f"Anchor {mid.name} with {pre_mid.name}, {late.name}, {latest.name} "
                    f"using angle_merge at {velocity:.2f}"
                ),
                left = mid,
                right = mid_right,
                velocity = velocity,
                merge_audio_vision = merge_audio_vision,
            )
        )

    for velocity in DEFAULT_LATE_ANGLE_VELOCITIES:
        tag = format_velocity_tag(velocity)
        candidates.append(
            build_direct_angle_candidate(
                name = f"cpt_angle_late_v{tag}",
                description = (
                    f"Anchor {late.name} with {mid.name} and {latest.name} "
                    f"using angle_merge at {velocity:.2f}"
                ),
                left = late,
                right = late_right,
                velocity = velocity,
                merge_audio_vision = merge_audio_vision,
            )
        )

    for velocity in DEFAULT_LONGARC_VELOCITIES:
        tag = format_velocity_tag(velocity)
        candidates.append(
            build_direct_angle_candidate(
                name = f"cpt_angle_longarc_v{tag}",
                description = (
                    f"Anchor {mid.name} with {longarc_early.name}, {pre_mid.name}, "
                    f"{late.name}, {latest.name} using angle_merge at {velocity:.2f}"
                ),
                left = mid,
                right = longarc_right,
                velocity = velocity,
                merge_audio_vision = merge_audio_vision,
            )
        )

    candidates.append(
        build_chain_mix_candidate(
            key = "cpt_chain_mix_mid_late_latest_a",
            description = (
                f"Two-stage late smoothing: {mid.name} -> {late.name} at 0.35, "
                f"then -> {latest.name} at 0.20"
            ),
            start = mid,
            mid = late,
            end = latest,
            first_velocity = 0.35,
            second_velocity = 0.20,
            merge_audio_vision = merge_audio_vision,
        )
    )
    candidates.append(
        build_chain_mix_candidate(
            key = "cpt_chain_mix_premid_late_latest_b",
            description = (
                f"Two-stage late smoothing: {pre_mid.name} -> {late.name} at 0.50, "
                f"then -> {latest.name} at 0.20"
            ),
            start = pre_mid,
            mid = late,
            end = latest,
            first_velocity = 0.50,
            second_velocity = 0.20,
            merge_audio_vision = merge_audio_vision,
        )
    )

    return candidates


def write_candidate_yaml(
    *,
    candidate: Candidate,
    output_path: Path,
    torch_dtype: str,
    base_model: str,
    checkpoints: list[Path],
) -> None:
    output_path.parent.mkdir(parents = True, exist_ok = True)
    payload = {
        "torch_dtype": torch_dtype,
        "models": candidate.models,
    }
    with output_path.open("w", encoding = "utf-8") as handle:
        handle.write(
            "# Generated by Tools/cpt_merge_search.py\n"
            f"# base_model: {base_model}\n"
            f"# checkpoints: {', '.join(path.name for path in checkpoints)}\n"
            f"# candidate: {candidate.key}\n"
            f"# description: {candidate.description}\n\n"
        )
        yaml.safe_dump(payload, handle, allow_unicode = True, sort_keys = False)


def write_plan(
    *,
    plan_path: Path,
    base_model: str,
    checkpoints: list[Path],
    candidates: list[Candidate],
    config_paths: dict[str, Path],
) -> None:
    payload = {
        "base_model": base_model,
        "checkpoints": [str(path) for path in checkpoints],
        "candidates": [
            {
                "key": candidate.key,
                "description": candidate.description,
                "config_path": str(config_paths[candidate.key]),
                "final_model_name": candidate.final_model_name,
            }
            for candidate in candidates
        ],
    }
    plan_path.parent.mkdir(parents = True, exist_ok = True)
    plan_path.write_text(
        json.dumps(payload, ensure_ascii = False, indent = 2),
        encoding = "utf-8",
    )


def run_merge(
    *,
    repo_root: Path,
    config_path: Path,
    merged_root: Path,
    merge_models_device: str,
    torch_dtype: str,
) -> None:
    command = [
        sys.executable,
        str(repo_root / "main.py"),
        "-c",
        str(config_path),
        "-o",
        str(merged_root),
        "-dm",
        merge_models_device,
        "-t",
        torch_dtype,
    ]
    subprocess.run(command, cwd = repo_root, check = True)


def find_final_artifact(merged_root: Path, final_model_name: str) -> Path | None:
    vector_dir = merged_root / "vector"
    if not vector_dir.exists():
        return None
    matches = sorted(
        vector_dir.glob(f"{final_model_name}_*"),
        key = lambda item: item.stat().st_mtime,
        reverse = True,
    )
    if not matches:
        return None
    return matches[0]


def run_eval(
    *,
    eval_script: Path,
    eval_workdir: Path,
    prompt_file: Path,
    base_model: str,
    merged_models: list[Path],
    eval_output_dir: Path,
) -> None:
    if not merged_models:
        return

    command = [
        sys.executable,
        str(eval_script),
        "--prompt-file",
        str(prompt_file),
        "--output-dir",
        str(eval_output_dir),
        "--base-model-name",
        base_model,
        "--discover-glob",
        "__no_auto_discovery__",
    ]
    for merged_model in merged_models:
        command.extend(["--model", str(merged_model)])

    subprocess.run(command, cwd = eval_workdir, check = True)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    checkpoints = resolve_checkpoints(args)
    base_model = args.base_model or infer_base_model(checkpoints[0])
    merge_audio_vision = args.merge_audio_vision

    candidates = build_candidates(checkpoints, merge_audio_vision)
    config_root = args.output_root / "configs"
    merged_root = args.output_root / "merged"
    eval_root = args.output_root / "eval"

    config_paths: dict[str, Path] = {}
    for candidate in candidates:
        config_path = config_root / f"{candidate.key}.yaml"
        write_candidate_yaml(
            candidate = candidate,
            output_path = config_path,
            torch_dtype = args.torch_dtype,
            base_model = base_model,
            checkpoints = checkpoints,
        )
        config_paths[candidate.key] = config_path

    write_plan(
        plan_path = args.output_root / "plan.json",
        base_model = base_model,
        checkpoints = checkpoints,
        candidates = candidates,
        config_paths = config_paths,
    )

    print(f"Base model: {base_model}")
    print("Checkpoints:")
    for checkpoint in checkpoints:
        print(f"  - {checkpoint}")
    print("Candidates:")
    for candidate in candidates:
        print(f"  - {candidate.key}: {candidate.description}")
        print(f"    config: {config_paths[candidate.key]}")

    if not args.run_merges:
        return

    merged_artifacts: list[Path] = []
    for candidate in candidates:
        config_path = config_paths[candidate.key]
        print(f"\n=== Running merge for {candidate.key} ===")
        run_merge(
            repo_root = repo_root,
            config_path = config_path,
            merged_root = merged_root,
            merge_models_device = args.merge_models_device,
            torch_dtype = args.torch_dtype,
        )
        artifact = find_final_artifact(merged_root, candidate.final_model_name)
        if artifact is None:
            print(f"Warning: could not find merged artifact for {candidate.key}", file = sys.stderr)
            continue
        merged_artifacts.append(artifact)
        print(f"Final artifact: {artifact}")

    if args.run_eval:
        eval_output_dir = eval_root / "cpt_eval_merge_search"
        print(f"\n=== Running evaluation for {len(merged_artifacts)} merged models ===")
        run_eval(
            eval_script = args.eval_script.resolve(),
            eval_workdir = args.eval_workdir.resolve(),
            prompt_file = args.prompt_file.resolve(),
            base_model = base_model,
            merged_models = merged_artifacts,
            eval_output_dir = eval_output_dir.resolve(),
        )
        print(f"Eval output: {eval_output_dir.resolve()}")


if __name__ == "__main__":
    main()
