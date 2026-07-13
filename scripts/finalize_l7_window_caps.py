#!/usr/bin/env python3
"""Filter infeasible motions and build the matching WPC label file."""

import argparse
import csv
import glob
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from active_adaptation.utils.motion import MotionMinimalData, _write_motion_dataset


LABEL_FIELDS = {
    "motion_id",
    "source_path",
    "window_idx",
    "window_start_frame",
    "window_end_frame",
    "max_success_load_kg",
    "status",
}


def read_rollout_csvs(input_glob, required_fields, expected_shards=None):
    paths = [Path(path) for path in sorted(glob.glob(input_glob))]
    if not paths:
        raise FileNotFoundError(f"No rollout CSV files matched {input_glob!r}")
    if expected_shards is not None:
        shard_ids = []
        for path in paths:
            match = re.fullmatch(r"shard_(\d+)\.csv", path.name)
            if not match:
                raise ValueError(f"Unexpected rollout shard filename: {path.name}")
            shard_ids.append(int(match.group(1)))
        expected = list(range(expected_shards))
        if sorted(shard_ids) != expected:
            raise ValueError(f"Expected rollout shards {expected}, got {sorted(shard_ids)}")

    rows = []
    fieldnames = None
    for path in paths:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            current_fields = reader.fieldnames or []
            missing = required_fields - set(current_fields)
            if missing:
                raise ValueError(f"{path} is missing fields: {sorted(missing)}")
            if fieldnames is not None and set(current_fields) != set(fieldnames):
                raise ValueError(f"{path} has a different CSV header")
            fieldnames = fieldnames or current_fields
            rows.extend(reader)
    if not rows:
        raise ValueError("No rollout windows found in the matched CSV files")

    windows = defaultdict(list)
    expected_counts = defaultdict(set)
    seen = set()
    has_expected_counts = "expected_window_count" in fieldnames
    for row in rows:
        try:
            motion_id = int(row["motion_id"])
            window_idx = int(row["window_idx"])
            start = int(row["window_start_frame"])
            end = int(row["window_end_frame"])
            if has_expected_counts:
                expected_counts[motion_id].add(int(row["expected_window_count"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid rollout window row: {row}") from exc
        key = (motion_id, window_idx)
        if key in seen:
            raise ValueError(f"Duplicate rollout window: motion_id={motion_id}, window_idx={window_idx}")
        if motion_id < 0 or window_idx < 0 or end <= start:
            raise ValueError(f"Invalid rollout window bounds or index: {row}")
        seen.add(key)
        windows[motion_id].append((window_idx, start, end))

    for motion_id, motion_windows in windows.items():
        motion_windows.sort()
        indices = [window_idx for window_idx, _, _ in motion_windows]
        expected_count = len(indices)
        if has_expected_counts:
            counts = expected_counts[motion_id]
            if len(counts) != 1 or next(iter(counts)) <= 0:
                raise ValueError(f"Invalid expected_window_count for motion_id={motion_id}: {sorted(counts)}")
            expected_count = next(iter(counts))
        if expected_count != len(indices) or any(index != position for position, index in enumerate(indices)):
            raise ValueError(
                f"Missing rollout windows for motion_id={motion_id}: "
                f"expected_count={expected_count}, got indices={indices}"
            )
        if any(previous[2] != current[1] for previous, current in zip(motion_windows, motion_windows[1:])):
            raise ValueError(f"Non-contiguous rollout windows for motion_id={motion_id}")
    return paths, fieldnames, rows


def motion_key(row):
    return row["teacher_mem_path"], int(row["local_motion_id"])


def split_feasible_rows(rows):
    invalid = {motion_key(row) for row in rows if row["status"] != "success"}
    return [row for row in rows if motion_key(row) not in invalid], invalid


def filter_motion_dataset(source: Path, output: Path, remove_ids: set[int]):
    if output.exists():
        raise FileExistsError(output)
    meta = json.loads((source / "meta_motion.json").read_text())
    labels = json.loads((source / "id_label.json").read_text())
    if remove_ids - set(range(len(labels))):
        raise ValueError(f"remove IDs out of range for {source}")
    data = MotionMinimalData.load_memmap(str(source / "_tensordict"))
    keep = [i for i in range(len(labels)) if i not in remove_ids]
    starts, ends = meta["starts"], meta["ends"]
    _write_motion_dataset(
        joint_names=meta["joint_names"],
        body_names=meta.get("body_names", []),
        metadata_rows=[{key: values[i] for key, values in meta.get("info", {}).items()} for i in keep],
        id_labels=[labels[i] for i in keep],
        lengths=[ends[i] - starts[i] for i in keep],
        root_pos_chunks=[data.root_pos_w[starts[i]:ends[i]] for i in keep],
        root_quat_chunks=[data.root_quat_w[starts[i]:ends[i]] for i in keep],
        joint_pos_chunks=[data.joint_pos[starts[i]:ends[i]] for i in keep],
        mem_path=str(output),
    )


def build_labels(rows, output: Path, window_sec: float, max_load_kg: float, load_step_kg: float):
    rows.sort(key=lambda row: (row["source_path"], int(row["window_start_frame"])))
    motion_files = sorted({row["source_path"] for row in rows})
    motion_ids = {path: idx for idx, path in enumerate(motion_files)}
    caps = np.asarray([float(row["max_success_load_kg"]) for row in rows], dtype=np.float32)
    if not np.isfinite(caps).all() or np.any((caps < 0.0) | (caps > max_load_kg)):
        raise ValueError(f"max_success_load_kg must be finite and within [0, {max_load_kg}]")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        motion_files=np.asarray(motion_files, dtype=str),
        bin_motion_idx=np.asarray([motion_ids[row["source_path"]] for row in rows], dtype=np.int32),
        bin_idx=np.asarray([int(row["window_idx"]) for row in rows], dtype=np.int32),
        start_frame=np.asarray([int(row["window_start_frame"]) for row in rows], dtype=np.int32),
        end_frame=np.asarray([int(row["window_end_frame"]) for row in rows], dtype=np.int32),
        window_cap_kg=caps,
        bin_size_s=np.asarray([window_sec], dtype=np.float32),
        max_load_kg=np.asarray([max_load_kg], dtype=np.float32),
        load_step_kg=np.asarray([load_step_kg], dtype=np.float32),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--filtered-dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--window-sec", type=float, default=5.0)
    parser.add_argument("--max-load-kg", type=float, default=30.0)
    parser.add_argument("--load-step-kg", type=float, default=5.0)
    parser.add_argument("--expected-shards", type=int, default=8)
    parser.add_argument("--paired-dataset", action="append", default=[], metavar="TEACHER=STUDENT")
    args = parser.parse_args()
    values = (args.window_sec, args.max_load_kg, args.load_step_kg)
    if (
        not all(map(math.isfinite, values))
        or args.window_sec <= 0
        or args.max_load_kg < 0
        or args.load_step_kg <= 0
        or args.expected_shards <= 0
    ):
        parser.error("window/load values must be finite; durations and steps positive; max load nonnegative")

    paired_datasets = {}
    for pair in args.paired_dataset:
        if "=" not in pair:
            parser.error("--paired-dataset must be TEACHER=STUDENT")
        teacher, student = pair.split("=", 1)
        paired_datasets[teacher] = student

    required = LABEL_FIELDS | {"teacher_mem_path", "student_mem_path", "local_motion_id"}
    paths, fieldnames, rows = read_rollout_csvs(args.input_glob, required, args.expected_shards)
    feasible_rows, invalid = split_feasible_rows(rows)
    if not feasible_rows:
        raise ValueError("No feasible motions remain after filtering")
    all_motions = {motion_key(row) for row in rows}

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
    for mem_path, remove_ids in sorted(removal_plan.items()):
        relative = Path(mem_path)
        if relative.is_absolute():
            raise ValueError(f"Dataset paths must be relative to --dataset-root: {mem_path}")
        filter_motion_dataset(dataset_root / relative, filtered_root / relative, remove_ids)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    feasible_csv = output_dir / "feasible_windows.csv"
    with feasible_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(feasible_rows)

    output = Path(args.output).resolve()
    build_labels(feasible_rows, output, args.window_sec, args.max_load_kg, args.load_step_kg)
    summary = {
        "input_csvs": [str(path) for path in paths],
        "motions_before": len(all_motions),
        "motions_removed": len(invalid),
        "motions_kept": len(all_motions - invalid),
        "windows_before": len(rows),
        "windows_kept": len(feasible_rows),
        "filtered_dataset_root": str(filtered_root),
        "label_path": str(output),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
