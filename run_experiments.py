from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

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
    parser.add_argument("--seeds", nargs="+", type=int, help="run multiple seeds in one structured run folder")
    parser.add_argument("--restarts", type=int, help="override preset restart count; 0 means use seconds budget")
    parser.add_argument("--top-k", type=int, help="override preset randomized choice width")
    parser.add_argument("--max-optional-candidates", type=int, help="override preset optional candidate cap")
    parser.add_argument("--mandatory-mode", choices=["standard", "role"], help="override mandatory construction mode")
    parser.add_argument(
        "--solution-extension",
        default="txt",
        help="extension for generated solution files, for example txt or .out",
    )
    parser.add_argument(
        "--refresh-final",
        action="store_true",
        help="copy the best valid checked output per active instance from outputs/runs.csv into final/",
    )
    parser.add_argument("--final-dir", type=Path, default=Path("final"), help="folder refreshed by --refresh-final")
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
        "seeds": args.seeds if args.seeds else [args.seed],
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


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "valid"}


def resolve_existing_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def row_instance_stem(row: Dict[str, str]) -> str:
    input_file = row.get("input_file", "")
    return Path(input_file.replace("\\", "/")).stem


def clear_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def find_best_checked_rows(
    runs_csv: Path,
    target_stems: List[str],
) -> Tuple[Dict[str, Dict[str, str]], List[str]]:
    if not runs_csv.exists():
        raise FileNotFoundError(f"{runs_csv} does not exist")

    targets = set(target_stems)
    best: Dict[str, Dict[str, str]] = {}
    skipped: List[str] = []

    with runs_csv.open(newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            stem = row_instance_stem(row)
            if stem not in targets:
                continue
            if not truthy(row.get("valid", "")):
                continue

            output_text = row.get("output_file", "")
            source = resolve_existing_path(output_text)
            if not source.exists():
                skipped.append(f"{stem}: missing output {output_text}")
                continue

            try:
                score = float(row.get("score", "0") or 0)
            except ValueError:
                skipped.append(f"{stem}: invalid score {row.get('score')}")
                continue

            try:
                solve.check_submission_structure(source)
            except Exception as exc:
                skipped.append(f"{stem}: structure check failed for {output_text}: {exc}")
                continue

            current = best.get(stem)
            current_score = float(current.get("score", "0") or 0) if current is not None else -1.0
            current_timestamp = current.get("timestamp", "") if current is not None else ""
            row_timestamp = row.get("timestamp", "")
            if (
                current is None
                or score > current_score
                or (score == current_score and row_timestamp >= current_timestamp)
            ):
                row["_resolved_output_file"] = str(source)
                best[stem] = row

    return best, skipped


def refresh_final(args: argparse.Namespace) -> int:
    instances = find_instances(args)
    target_stems = [path.stem for path in instances]
    best_rows, skipped = find_best_checked_rows(args.output_root / "runs.csv", target_stems)

    missing = [stem for stem in target_stems if stem not in best_rows]
    if missing:
        print(f"missing valid checked run(s): {', '.join(missing)}")
        if skipped and not args.quiet:
            print("skipped candidates:")
            for message in skipped[-20:]:
                print(f"  {message}")
        return 1

    clear_directory(args.final_dir)

    total = 0.0
    for instance_path in instances:
        stem = instance_path.stem
        row = best_rows[stem]
        source = Path(row["_resolved_output_file"])
        target = args.final_dir / instance_path.name
        shutil.copyfile(source, target)
        solve.check_submission_structure(target)
        score = float(row.get("score", "0") or 0)
        total += score
        if not args.quiet:
            print(
                f"{target}: score={score:.6f} seed={row.get('seed')} "
                f"approach={row.get('approach')} source={row.get('output_file')}"
            )

    if not args.quiet:
        print(f"final_dir={args.final_dir}")
        print(f"total_score={total:.6f}")
    return 0


def run_instance(
    instance_path: Path,
    output_path: Path,
    args: argparse.Namespace,
    config: ApproachPreset,
    run_id: str,
    timestamp: str,
    seed: int,
) -> Dict[str, object]:
    started = time.monotonic()
    output_text = str(output_path)

    try:
        instance = solve.parse_instance(instance_path)
        solution = solve.solve(
            instance=instance,
            seconds=config.seconds,
            seed=seed,
            restarts=config.restarts,
            top_k=max(1, config.top_k),
            max_optional_candidates=max(0, config.max_optional_candidates),
            mandatory_mode=config.mandatory_mode,
        )
        solve.write_solution(output_path, solution)
        solve.check_submission_structure(output_path)
        cleaned_count = len({street_id for vehicle in solution.cleaned_by_vehicle for street_id in vehicle})
        valid = solution.valid
        score = solution.score
        message = solution.message
    except Exception as exc:  # Keep the run summary useful even if one instance fails.
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
        "seed": seed,
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
    if args.refresh_final:
        return refresh_final(args)

    config = resolve_config(args)
    instances = find_instances(args)
    solution_extension = normalize_extension(args.solution_extension)
    seeds = args.seeds if args.seeds else [args.seed]

    approach_dir = args.output_root / args.approach
    run_id = next_run_id(approach_dir)
    run_dir = approach_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    timestamp = datetime.now().isoformat(timespec="seconds")

    write_metadata(run_dir, args, config, run_id, timestamp, instances)

    rows: List[Dict[str, object]] = []
    use_seed_suffix = len(seeds) > 1
    for seed in seeds:
        for instance_path in instances:
            suffix = f"_seed{seed:03d}" if use_seed_suffix else ""
            output_path = run_dir / f"{instance_path.stem}{suffix}{solution_extension}"
            row = run_instance(instance_path, output_path, args, config, run_id, timestamp, seed)
            rows.append(row)
            if not args.quiet:
                print(
                    f"{instance_path.name} seed={seed}: valid={row['valid']} score={row['score']} "
                    f"cleaned={row['cleaned_count']} output={row['output_file']}"
                )

    write_summary(run_dir / "summary.csv", rows)
    append_csv(args.output_root / "runs.csv", rows)

    if not args.quiet:
        print(f"run_dir={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
