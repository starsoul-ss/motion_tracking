#!/usr/bin/env python3
"""Merge window-load-cap shard CSVs into the public WPC label format."""

import argparse
import csv
import glob
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


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
        if expected_shards <= 0:
            raise ValueError("--expected-shards must be positive")
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
    has_expected_counts = "expected_window_count" in (fieldnames or [])
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
        for previous, current in zip(motion_windows, motion_windows[1:]):
            if previous[2] != current[1]:
                raise ValueError(f"Non-contiguous rollout windows for motion_id={motion_id}")
    return paths, fieldnames, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--window-sec", type=float, default=5.0)
    parser.add_argument("--max-load-kg", type=float, default=30.0)
    parser.add_argument("--load-step-kg", type=float, default=5.0)
    parser.add_argument("--expected-shards", type=int)
    args = parser.parse_args()
    values = (args.window_sec, args.max_load_kg, args.load_step_kg)
    if not np.isfinite(values).all() or args.window_sec <= 0.0 or args.max_load_kg < 0.0 or args.load_step_kg <= 0.0:
        parser.error("--window-sec and --load-step-kg must be positive; --max-load-kg must be nonnegative")

    _, _, rows = read_rollout_csvs(args.input_glob, LABEL_FIELDS, args.expected_shards)
    invalid = [row for row in rows if row["status"] != "success"]
    if invalid:
        statuses = sorted({row["status"] for row in invalid})
        raise ValueError(
            f"Refusing to build training labels with {len(invalid)} invalid windows "
            f"(statuses={statuses}). Filter invalid motions before building labels."
        )
    rows.sort(key=lambda row: (row["source_path"], int(row["window_start_frame"])))

    motion_files = sorted({row["source_path"] for row in rows})
    motion_ids = {path: idx for idx, path in enumerate(motion_files)}
    caps = np.asarray([float(row["max_success_load_kg"]) for row in rows], dtype=np.float32)
    if not np.isfinite(caps).all() or np.any((caps < 0.0) | (caps > args.max_load_kg)):
        raise ValueError(f"max_success_load_kg must be finite and within [0, {args.max_load_kg}]")
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        motion_files=np.asarray(motion_files, dtype=str),
        bin_motion_idx=np.asarray([motion_ids[row["source_path"]] for row in rows], dtype=np.int32),
        bin_idx=np.asarray([int(row["window_idx"]) for row in rows], dtype=np.int32),
        start_frame=np.asarray([int(row["window_start_frame"]) for row in rows], dtype=np.int32),
        end_frame=np.asarray([int(row["window_end_frame"]) for row in rows], dtype=np.int32),
        window_cap_kg=caps,
        bin_size_s=np.asarray([args.window_sec], dtype=np.float32),
        max_load_kg=np.asarray([args.max_load_kg], dtype=np.float32),
        load_step_kg=np.asarray([args.load_step_kg], dtype=np.float32),
    )
    print(f"saved {output}: motions={len(motion_files)} windows={len(rows)}")


if __name__ == "__main__":
    main()
