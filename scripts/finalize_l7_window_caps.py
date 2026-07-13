#!/usr/bin/env python3
"""Drop infeasible motions, copy aligned datasets, and build WPC labels."""

import argparse
import csv
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_l7_window_cap_labels import LABEL_FIELDS, read_rollout_csvs


def motion_key(row):
    return row["teacher_mem_path"], int(row["local_motion_id"])


def split_feasible_rows(rows):
    invalid = {motion_key(row) for row in rows if row["status"] != "success"}
    return [row for row in rows if motion_key(row) not in invalid], invalid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--filtered-dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--window-sec", type=float, default=5.0)
    parser.add_argument("--motion-fps", type=float, default=50.0)
    parser.add_argument("--max-load-kg", type=float, default=30.0)
    parser.add_argument("--load-step-kg", type=float, default=5.0)
    parser.add_argument("--expected-shards", type=int, default=8)
    parser.add_argument("--paired-dataset", action="append", default=[], metavar="TEACHER=STUDENT")
    args = parser.parse_args()
    values = (args.window_sec, args.motion_fps, args.max_load_kg, args.load_step_kg)
    if not all(map(math.isfinite, values)) or args.window_sec <= 0.0 or args.motion_fps <= 0.0 or args.max_load_kg < 0.0 or args.load_step_kg <= 0.0:
        parser.error("window/fps/load values must be finite; durations and steps positive; max load nonnegative")

    paired_datasets = {}
    for pair in args.paired_dataset:
        if "=" not in pair:
            parser.error("--paired-dataset must be TEACHER=STUDENT")
        teacher, student = pair.split("=", 1)
        paired_datasets[teacher] = student

    required = {"teacher_mem_path", "student_mem_path", "local_motion_id", "status"}
    paths, fieldnames, rows = read_rollout_csvs(
        args.input_glob,
        required | LABEL_FIELDS,
        args.expected_shards,
    )

    feasible_rows, invalid = split_feasible_rows(rows)
    all_motions = {motion_key(row) for row in rows}
    output_dir = Path(args.output_dir).resolve()
    removal_dir = output_dir / "removals"
    removal_dir.mkdir(parents=True, exist_ok=True)

    removal_plan = defaultdict(set)
    for row in rows:
        teacher_path, local_id = motion_key(row)
        removal_plan[teacher_path]
        student_path = row["student_mem_path"].strip() or paired_datasets.get(teacher_path, "")
        if student_path:
            removal_plan[student_path]
        if (teacher_path, local_id) in invalid:
            removal_plan[teacher_path].add(local_id)
            if student_path:
                removal_plan[student_path].add(local_id)

    dataset_root = Path(args.dataset_root).resolve()
    filtered_root = Path(args.filtered_dataset_root).resolve()
    filter_script = Path(__file__).with_name("filter_motion_memmap.py")
    for mem_path, remove_ids in sorted(removal_plan.items()):
        relative = Path(mem_path)
        if relative.is_absolute():
            raise ValueError(f"Dataset paths must be relative to --dataset-root: {mem_path}")
        manifest = removal_dir / ("__".join(relative.parts) + ".json")
        manifest.write_text(json.dumps(sorted(remove_ids), indent=2) + "\n", encoding="utf-8")
        subprocess.run(
            [
                sys.executable,
                str(filter_script),
                "--source",
                str(dataset_root / relative),
                "--output",
                str(filtered_root / relative),
                "--remove-ids",
                str(manifest),
            ],
            check=True,
        )

    feasible_csv = output_dir / "feasible_windows.csv"
    with feasible_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(feasible_rows)

    builder = Path(__file__).with_name("build_l7_window_cap_labels.py")
    subprocess.run(
        [
            sys.executable,
            str(builder),
            "--input-glob",
            str(feasible_csv),
            "--output",
            str(Path(args.output).resolve()),
            "--window-sec",
            str(args.window_sec),
            "--max-load-kg",
            str(args.max_load_kg),
            "--load-step-kg",
            str(args.load_step_kg),
        ],
        check=True,
    )

    window_frames = round(args.window_sec * args.motion_fps)
    summary = {
        "input_csvs": [str(path) for path in paths],
        "motions_before": len(all_motions),
        "motions_removed": len(invalid),
        "motions_kept": len(all_motions - invalid),
        "windows_before": len(rows),
        "windows_kept": len(feasible_rows),
        "partial_final_windows_kept": sum(
            int(row["window_end_frame"]) - int(row["window_start_frame"]) < window_frames
            for row in feasible_rows
        ),
        "filtered_dataset_root": str(filtered_root),
        "label_path": str(Path(args.output).resolve()),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
