from __future__ import annotations

import argparse
import csv
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import solve


SUMMARY_FIELDS = [
    "approach",
    "run_id",
    "timestamp",
    "input_file",
    "output_file",
    "seed",
    "seconds",
    "restarts",
    "top_k",
    "max_optional_candidates",
    "valid",
    "score",
    "cleaned_count",
    "message",
    "runtime_seconds",
]


@dataclass(frozen=True)
class ApproachPreset:
    seconds: float
    restarts: int
    top_k: int
    max_optional_candidates: int
    mandatory_mode: str = "standard"


PRESETS: Dict[str, ApproachPreset] = {
    "baseline": ApproachPreset(
        seconds=1.0,
        restarts=1,
        top_k=1,
        max_optional_candidates=0,
    ),
    "grasp": ApproachPreset(
        seconds=30.0,
        restarts=0,
        top_k=4,
        max_optional_candidates=20000,
    ),
    "tuned": ApproachPreset(
        seconds=90.0,
        restarts=0,
        top_k=8,
        max_optional_candidates=50000,
    ),
    "role_grasp": ApproachPreset(
        seconds=30.0,
        restarts=0,
        top_k=4,
        max_optional_candidates=20000,
        mandatory_mode="role",
    ),
    "coverage_tuned": ApproachPreset(
        seconds=90.0,
        restarts=0,
        top_k=4,
        max_optional_candidates=50000,
        mandatory_mode="role",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run structured Street Cleaning solver experiments.")
    parser.add_argument("--approach", choices=sorted(PRESETS), default="grasp", help="approach preset to run")
    parser.add_argument("--input-dir", type=Path, default=Path("training"), help="folder used when --instances is omitted")
    parser.add_argument("--output-root", type=Path, default=Path("outputs"), help="root folder for experiment outputs")
    parser.add_argument("--instances", nargs="*", type=Path, help="specific instance files; defaults to all *.txt in --input-dir")
    parser.add_argument("--seconds", type=float, help="override preset seconds per instance")
    parser.add_argument("--seed", type=int, default=1, help="base random seed")
    parser.add_argument("--restarts", type=int, help="override preset restart count; 0 means use seconds budget")
    parser.add_argument("--top-k", type=int, help="override preset randomized choice width")
    parser.add_argument("--max-optional-candidates", type=int, help="override preset optional candidate cap")
    parser.add_argument("--mandatory-mode", choices=["standard", "role"], help="override mandatory construction mode")
    parser.add_argument(
        "--solution-extension",
        default="txt",
        help="extension for generated solution files, for example txt or .out",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress per-instance console output")
    return parser.parse_args()


def resolve_config(args: argparse.Namespace) -> ApproachPreset:
    preset = PRESETS[args.approach]
    return ApproachPreset(
        seconds=args.seconds if args.seconds is not None else preset.seconds,
        restarts=args.restarts if args.restarts is not None else preset.restarts,
        top_k=args.top_k if args.top_k is not None else preset.top_k,
        max_optional_candidates=(
            args.max_optional_candidates
            if args.max_optional_candidates is not None
            else preset.max_optional_candidates
        ),
        mandatory_mode=args.mandatory_mode if args.mandatory_mode is not None else preset.mandatory_mode,
    )


def find_instances(args: argparse.Namespace) -> List[Path]:
    if args.instances:
        instances = args.instances
    else:
        instances = sorted(args.input_dir.glob("*.txt"))

    if not instances:
        raise ValueError("no input instances found")

    missing = [path for path in instances if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing instance file(s): {', '.join(str(path) for path in missing)}")

    return instances


def normalize_extension(extension: str) -> str:
    cleaned = extension.strip()
    if not cleaned:
        raise ValueError("--solution-extension cannot be empty")
    if not cleaned.startswith("."):
        cleaned = f".{cleaned}"
    return cleaned


def next_run_id(approach_dir: Path) -> str:
    approach_dir.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(r"^run_(\d+)_")
    highest = 0
    for child in approach_dir.iterdir():
        if not child.is_dir():
            continue
        match = pattern.match(child.name)
        if match:
            highest = max(highest, int(match.group(1)))

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_number = highest + 1
    run_id = f"run_{run_number:04d}_{timestamp}"
    while (approach_dir / run_id).exists():
        run_number += 1
        run_id = f"run_{run_number:04d}_{timestamp}"
    return run_id


def write_metadata(
    run_dir: Path,
    args: argparse.Namespace,
    config: ApproachPreset,
    run_id: str,
    timestamp: str,
    instances: List[Path],
) -> None:
    metadata = {
        "approach": args.approach,
        "run_id": run_id,
        "timestamp": timestamp,
        "seed": args.seed,
        "config": {
            "seconds": config.seconds,
            "restarts": config.restarts,
            "top_k": config.top_k,
            "max_optional_candidates": config.max_optional_candidates,
            "mandatory_mode": config.mandatory_mode,
        },
        "input_dir": str(args.input_dir),
        "output_root": str(args.output_root),
        "solution_extension": normalize_extension(args.solution_extension),
        "instances": [str(path) for path in instances],
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def append_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def run_instance(
    instance_path: Path,
    output_path: Path,
    args: argparse.Namespace,
    config: ApproachPreset,
    run_id: str,
    timestamp: str,
) -> Dict[str, object]:
    started = time.monotonic()
    output_text = str(output_path)

    try:
        instance = solve.parse_instance(instance_path)
        solution = solve.solve(
            instance=instance,
            seconds=config.seconds,
            seed=args.seed,
            restarts=config.restarts,
            top_k=max(1, config.top_k),
            max_optional_candidates=max(0, config.max_optional_candidates),
            mandatory_mode=config.mandatory_mode,
        )
        solve.write_solution(output_path, solution)
        cleaned_count = len({street_id for vehicle in solution.cleaned_by_vehicle for street_id in vehicle})
        valid = solution.valid
        score = solution.score
        message = solution.message
    except Exception as exc:  # Keep the run summary useful even if one instance fails.
        output_text = ""
        cleaned_count = 0
        valid = False
        score = 0.0
        message = f"error: {exc}"

    runtime = time.monotonic() - started
    return {
        "approach": args.approach,
        "run_id": run_id,
        "timestamp": timestamp,
        "input_file": str(instance_path),
        "output_file": output_text,
        "seed": args.seed,
        "seconds": config.seconds,
        "restarts": config.restarts,
        "top_k": config.top_k,
        "max_optional_candidates": config.max_optional_candidates,
        "valid": valid,
        "score": f"{score:.6f}",
        "cleaned_count": cleaned_count,
        "message": message,
        "runtime_seconds": f"{runtime:.3f}",
    }


def main() -> int:
    args = parse_args()
    config = resolve_config(args)
    instances = find_instances(args)
    solution_extension = normalize_extension(args.solution_extension)

    approach_dir = args.output_root / args.approach
    run_id = next_run_id(approach_dir)
    run_dir = approach_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    timestamp = datetime.now().isoformat(timespec="seconds")

    write_metadata(run_dir, args, config, run_id, timestamp, instances)

    rows: List[Dict[str, object]] = []
    for instance_path in instances:
        output_path = run_dir / f"{instance_path.stem}{solution_extension}"
        row = run_instance(instance_path, output_path, args, config, run_id, timestamp)
        rows.append(row)
        if not args.quiet:
            print(
                f"{instance_path.name}: valid={row['valid']} score={row['score']} "
                f"cleaned={row['cleaned_count']} output={row['output_file']}"
            )

    write_summary(run_dir / "summary.csv", rows)
    append_csv(args.output_root / "runs.csv", rows)

    if not args.quiet:
        print(f"run_dir={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
