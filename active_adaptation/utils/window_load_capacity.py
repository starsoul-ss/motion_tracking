from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _norm_path_key(path: str | os.PathLike[str]) -> str:
    return os.path.normcase(os.path.abspath(os.path.expanduser(str(path))))


def _basename_key(path: str | os.PathLike[str]) -> str:
    return Path(str(path)).name


def _resolve_label_path(label_path: str | os.PathLike[str]) -> Path:
    path = Path(label_path).expanduser()
    candidates = [path]
    if not path.is_absolute():
        root = _repo_root()
        candidates.extend(
            [
                root / path,
                root / "dataset" / path,
                root.parent / "humanoid_teleop" / "GMR" / path,
            ]
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Could not find window load capacity label path '{label_path}'. Tried: "
        + ", ".join(str(c) for c in candidates)
    )


def _first_present(data, names: Sequence[str]):
    for name in names:
        if name in data:
            return name
    return None


@dataclass
class WindowCapacityRows:
    source_paths: np.ndarray
    start_frames: np.ndarray
    end_frames: np.ndarray
    caps_kg: np.ndarray
    bin_indices: np.ndarray


def load_window_capacity_rows(
    label_path: str | os.PathLike[str],
    *,
    motion_fps: float = 50.0,
) -> tuple[WindowCapacityRows, Path]:
    path = _resolve_label_path(label_path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        meta = json.loads(path.read_text())
        npz_name = str(meta.get("npz", meta.get("label_path", "")) or "")
        if not npz_name:
            raise ValueError(f"JSON capacity label {path} must contain 'npz' or 'label_path'.")
        npz_path = Path(npz_name).expanduser()
        if not npz_path.is_absolute():
            npz_path = path.parent / npz_path
        return load_window_capacity_rows(npz_path, motion_fps=motion_fps)
    if suffix == ".npz":
        return _load_window_capacity_rows_npz(path), path
    return _load_window_capacity_rows_csv(path, motion_fps=motion_fps), path


def _load_window_capacity_rows_npz(path: Path) -> WindowCapacityRows:
    with np.load(path, allow_pickle=False) as data:
        cap_key = _first_present(
            data,
            (
                "window_cap_kg",
                "cap_window_kg",
                "load_cap_kg",
                "cap_q20_kg",
                "unit_cap_kg",
                "max_success_load_kg",
            ),
        )
        if cap_key is None:
            raise ValueError(
                f"Capacity label {path} does not contain a cap field. "
                "Expected one of window_cap_kg/cap_window_kg/load_cap_kg/cap_q20_kg/unit_cap_kg/max_success_load_kg."
            )

        if "motion_files" in data and "bin_motion_idx" in data:
            motion_files = np.asarray(data["motion_files"]).astype(str)
            motion_idx = np.asarray(data["bin_motion_idx"], dtype=np.int64)
            source_paths = motion_files[motion_idx]
            start_key = _first_present(data, ("start_frame", "window_start_frame"))
            end_key = _first_present(data, ("end_frame", "window_end_frame"))
            if start_key is None or end_key is None:
                raise ValueError(f"Capacity label {path} with motion_files must include start/end frame fields.")
            start_frames = np.asarray(data[start_key], dtype=np.int64)
            end_frames = np.asarray(data[end_key], dtype=np.int64)
            bin_indices = np.asarray(data["bin_idx"], dtype=np.int64) if "bin_idx" in data else np.arange(
                start_frames.shape[0], dtype=np.int64
            )
        else:
            source_key = _first_present(data, ("source_path", "file", "motion_file", "motion_files"))
            if source_key is None:
                raise ValueError(f"Capacity label {path} must include source_path/file or motion_files.")
            source_paths = np.asarray(data[source_key]).astype(str)
            start_key = _first_present(data, ("start_frame", "window_start_frame", "t_start_frame"))
            end_key = _first_present(data, ("end_frame", "window_end_frame", "t_end_frame"))
            if start_key is None or end_key is None:
                raise ValueError(f"Capacity label {path} must include start/end frame fields.")
            start_frames = np.asarray(data[start_key], dtype=np.int64)
            end_frames = np.asarray(data[end_key], dtype=np.int64)
            bin_indices = np.asarray(data["bin_idx"], dtype=np.int64) if "bin_idx" in data else np.arange(
                start_frames.shape[0], dtype=np.int64
            )

        caps = np.asarray(data[cap_key], dtype=np.float32)
        if source_paths.shape[0] != start_frames.shape[0] or caps.shape[0] != start_frames.shape[0]:
            raise ValueError(
                f"Capacity label {path} row count mismatch: "
                f"source_paths={source_paths.shape[0]} starts={start_frames.shape[0]} caps={caps.shape[0]}"
            )
        return WindowCapacityRows(
            source_paths=source_paths,
            start_frames=start_frames,
            end_frames=end_frames,
            caps_kg=caps,
            bin_indices=bin_indices,
        )


def _load_window_capacity_rows_csv(path: Path, *, motion_fps: float) -> WindowCapacityRows:
    source_paths: list[str] = []
    start_frames: list[int] = []
    end_frames: list[int] = []
    caps_kg: list[float] = []
    bin_indices: list[int] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader):
            source = row.get("source_path") or row.get("file") or row.get("motion_file")
            if source is None:
                raise ValueError(f"CSV capacity label {path} row {row_idx} is missing source_path/file.")
            if "start_frame" in row:
                start = int(row["start_frame"])
                end = int(row["end_frame"])
            elif "window_start_frame" in row:
                start = int(row["window_start_frame"])
                end = int(row["window_end_frame"])
            else:
                start_s = float(row["t_start_s"])
                end_s = float(row["t_end_s"])
                start = int(round(start_s * motion_fps))
                end = max(start + 1, int(round(end_s * motion_fps)))
            cap_raw = (
                row.get("window_cap_kg")
                or row.get("cap_window_kg")
                or row.get("load_cap_kg")
                or row.get("cap_q20_kg")
                or row.get("unit_cap_kg")
                or row.get("max_success_load_kg")
            )
            if cap_raw in (None, ""):
                raise ValueError(f"CSV capacity label {path} row {row_idx} is missing a cap field.")
            source_paths.append(str(source))
            start_frames.append(start)
            end_frames.append(end)
            caps_kg.append(float(cap_raw))
            bin_indices.append(int(row.get("bin_idx", row_idx)))
    return WindowCapacityRows(
        source_paths=np.asarray(source_paths, dtype=str),
        start_frames=np.asarray(start_frames, dtype=np.int64),
        end_frames=np.asarray(end_frames, dtype=np.int64),
        caps_kg=np.asarray(caps_kg, dtype=np.float32),
        bin_indices=np.asarray(bin_indices, dtype=np.int64),
    )


def _optional_int(row: dict, *names: str) -> int | None:
    for name in names:
        value = row.get(name)
        if value in (None, ""):
            continue
        try:
            return int(float(value))
        except (TypeError, ValueError):
            continue
    return None


def _unique_or_none(values: dict[str, list[int]]) -> dict[str, int]:
    return {key: ids[0] for key, ids in values.items() if len(ids) == 1}


def _to_numpy_int64(values: Sequence[int] | torch.Tensor) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy().astype(np.int64, copy=False)
    return np.asarray(values, dtype=np.int64)


@dataclass
class WindowLoadCapacityLookup:
    bin_motion_ids: torch.Tensor
    bin_indices: torch.Tensor
    bin_start_frames: torch.Tensor
    bin_end_frames: torch.Tensor
    bin_caps_kg: torch.Tensor
    key_start_frames: torch.Tensor
    key_stride: int
    num_motions: int
    matched_rows: int
    labeled_motions: int
    label_source: str

    @classmethod
    def from_label_file(
        cls,
        label_path: str | os.PathLike[str],
        *,
        motion_source_paths: Sequence[str],
        motion_labels: Sequence[dict] | None,
        motion_lengths: Sequence[int] | torch.Tensor,
        motion_fps: float,
        device: torch.device | str,
        cap_safety_scale: float = 1.0,
        missing_motion_policy: str = "error",
    ) -> "WindowLoadCapacityLookup":
        rows, resolved_path = load_window_capacity_rows(label_path, motion_fps=motion_fps)
        mapped = map_window_capacity_rows_to_motion_ids(
            rows,
            motion_source_paths=motion_source_paths,
            motion_labels=motion_labels,
            motion_lengths=motion_lengths,
            cap_safety_scale=cap_safety_scale,
        )
        if mapped.start_frames.size == 0:
            raise ValueError(
                f"No window capacity rows from {resolved_path} matched the loaded motion dataset. "
                "Check label source paths against the memmap id_label.json source_path values."
            )
        missing_policy = str(missing_motion_policy).lower()
        if missing_policy not in {"error", "zero"}:
            raise ValueError("missing_motion_policy must be 'error' or 'zero'.")
        num_motions = len(motion_source_paths)
        labeled_motion_ids = np.unique(mapped.motion_ids)
        if missing_policy == "error" and labeled_motion_ids.size != num_motions:
            missing = sorted(set(range(num_motions)) - set(map(int, labeled_motion_ids.tolist())))
            preview = missing[:10]
            raise ValueError(
                f"Capacity labels from {resolved_path} matched {labeled_motion_ids.size}/{num_motions} motions; "
                f"missing first ids={preview}. Use missing_motion_policy='zero' only if this is intentional."
            )
        motion_lengths_np = _to_numpy_int64(motion_lengths)
        # Labels generated before the half-open interval fix used length - 1
        # for the final end. Normalize them in memory without rewriting artifacts.
        for motion_id in labeled_motion_ids:
            idx = np.flatnonzero(mapped.motion_ids == motion_id)
            final_idx = idx[np.argmax(mapped.end_frames[idx])]
            if int(mapped.end_frames[final_idx]) == int(motion_lengths_np[motion_id]) - 1:
                mapped.end_frames[final_idx] = int(motion_lengths_np[motion_id])
        if missing_policy == "error":
            coverage_errors = []
            for motion_id in range(num_motions):
                idx = np.flatnonzero(mapped.motion_ids == motion_id)
                order_local = np.argsort(mapped.start_frames[idx], kind="stable")
                starts_local = mapped.start_frames[idx][order_local]
                ends_local = mapped.end_frames[idx][order_local]
                expected_end = max(int(motion_lengths_np[motion_id]), 1)
                contiguous = (
                    starts_local.size > 0
                    and int(starts_local[0]) == 0
                    and int(ends_local[-1]) == expected_end
                    and np.array_equal(ends_local[:-1], starts_local[1:])
                )
                if not contiguous:
                    coverage_errors.append(
                        (motion_id, starts_local[:4].tolist(), ends_local[-4:].tolist(), expected_end)
                    )
            if coverage_errors:
                raise ValueError(
                    f"Capacity labels from {resolved_path} do not continuously cover "
                    f"{len(coverage_errors)}/{num_motions} motions; first errors={coverage_errors[:5]}"
                )

        order = np.lexsort((mapped.end_frames, mapped.start_frames, mapped.motion_ids))
        motion_ids = mapped.motion_ids[order].astype(np.int64, copy=False)
        bin_indices = mapped.bin_indices[order].astype(np.int64, copy=False)
        starts = mapped.start_frames[order].astype(np.int64, copy=False)
        ends = mapped.end_frames[order].astype(np.int64, copy=False)
        caps = mapped.caps_kg[order].astype(np.float32, copy=False)
        motion_lengths_np = _to_numpy_int64(motion_lengths)
        max_frame = int(max(ends.max(initial=1), motion_lengths_np.max(initial=1)))
        stride = max_frame + 2
        keys = motion_ids * stride + starts
        return cls(
            bin_motion_ids=torch.as_tensor(motion_ids, device=device, dtype=torch.long),
            bin_indices=torch.as_tensor(bin_indices, device=device, dtype=torch.long),
            bin_start_frames=torch.as_tensor(starts, device=device, dtype=torch.long),
            bin_end_frames=torch.as_tensor(ends, device=device, dtype=torch.long),
            bin_caps_kg=torch.as_tensor(caps, device=device, dtype=torch.float32),
            key_start_frames=torch.as_tensor(keys, device=device, dtype=torch.long),
            key_stride=stride,
            num_motions=num_motions,
            matched_rows=int(starts.shape[0]),
            labeled_motions=int(labeled_motion_ids.size),
            label_source=str(resolved_path),
        )

    def lookup(self, motion_ids: torch.Tensor, frames: torch.Tensor) -> dict[str, torch.Tensor]:
        motion_ids = motion_ids.to(device=self.key_start_frames.device, dtype=torch.long)
        frames = frames.to(device=self.key_start_frames.device, dtype=torch.long)
        keys = motion_ids * int(self.key_stride) + frames
        raw_idx = torch.searchsorted(self.key_start_frames, keys, right=True) - 1
        in_range = (raw_idx >= 0) & (raw_idx < self.bin_motion_ids.numel())
        safe_idx = raw_idx.clamp(0, max(self.bin_motion_ids.numel() - 1, 0))
        same_motion = self.bin_motion_ids[safe_idx] == motion_ids
        contains = (self.bin_start_frames[safe_idx] <= frames) & (frames < self.bin_end_frames[safe_idx])
        valid = in_range & same_motion & contains

        bin_idx = torch.full_like(frames, -1)
        start = torch.zeros_like(frames)
        end = torch.zeros_like(frames)
        cap = torch.zeros(frames.shape, device=frames.device, dtype=torch.float32)
        flat_idx = torch.full_like(frames, -1)
        if valid.any():
            valid_idx = safe_idx[valid]
            bin_idx[valid] = self.bin_indices[valid_idx]
            start[valid] = self.bin_start_frames[valid_idx]
            end[valid] = self.bin_end_frames[valid_idx]
            cap[valid] = self.bin_caps_kg[valid_idx]
            flat_idx[valid] = valid_idx

        next_raw_idx = safe_idx + 1
        next_in_range = valid & (next_raw_idx < self.bin_motion_ids.numel())
        next_same_motion = torch.zeros_like(valid)
        if next_in_range.any():
            next_same_motion[next_in_range] = self.bin_motion_ids[next_raw_idx[next_in_range]] == motion_ids[next_in_range]
        next_valid = next_in_range & next_same_motion
        next_cap = torch.zeros_like(cap)
        next_start = torch.zeros_like(frames)
        if next_valid.any():
            valid_next_idx = next_raw_idx[next_valid]
            next_cap[next_valid] = self.bin_caps_kg[valid_next_idx]
            next_start[next_valid] = self.bin_start_frames[valid_next_idx]
        return {
            "valid": valid,
            "flat_idx": flat_idx,
            "bin_idx": bin_idx,
            "start": start,
            "end": end,
            "cap_kg": cap,
            "next_valid": next_valid,
            "next_start": next_start,
            "next_cap_kg": next_cap,
        }


@dataclass
class _MappedWindowCapacityRows:
    motion_ids: np.ndarray
    bin_indices: np.ndarray
    start_frames: np.ndarray
    end_frames: np.ndarray
    caps_kg: np.ndarray


def map_window_capacity_rows_to_motion_ids(
    rows: WindowCapacityRows,
    *,
    motion_source_paths: Sequence[str],
    motion_labels: Sequence[dict] | None,
    motion_lengths: Sequence[int] | torch.Tensor,
    cap_safety_scale: float = 1.0,
) -> _MappedWindowCapacityRows:
    labels = list(motion_labels) if motion_labels is not None else [{} for _ in motion_source_paths]
    if len(labels) != len(motion_source_paths):
        raise ValueError("motion_labels length must match motion_source_paths length.")
    lengths = _to_numpy_int64(motion_lengths)
    if lengths.shape[0] != len(motion_source_paths):
        raise ValueError("motion_lengths length must match motion_source_paths length.")

    exact_to_ids: dict[str, list[int]] = {}
    basename_to_ids: dict[str, list[int]] = {}
    path_windows: dict[str, list[tuple[int, int, int]]] = {}
    basename_windows: dict[str, list[tuple[int, int, int]]] = {}
    label_origins = np.zeros((len(motion_source_paths),), dtype=np.int64)
    label_has_window = np.zeros((len(motion_source_paths),), dtype=bool)
    for motion_id, source_path in enumerate(motion_source_paths):
        label = dict(labels[motion_id])
        path = str(label.get("source_path", source_path))
        path_key = _norm_path_key(path)
        base_key = _basename_key(path)
        exact_to_ids.setdefault(path_key, []).append(motion_id)
        basename_to_ids.setdefault(base_key, []).append(motion_id)
        window_origin = _optional_int(label, "window_start_frame")
        window_end = _optional_int(label, "window_end_frame")
        label_has_window[motion_id] = window_origin is not None or window_end is not None
        origin = window_origin if window_origin is not None else (_optional_int(label, "segment_start") or 0)
        label_end = window_end if window_end is not None else _optional_int(label, "segment_end")
        if label_end is None:
            label_end = origin + int(lengths[motion_id])
        label_origins[motion_id] = origin
        path_windows.setdefault(path_key, []).append((origin, label_end, motion_id))
        basename_windows.setdefault(base_key, []).append((origin, label_end, motion_id))

    exact_unique = _unique_or_none(exact_to_ids)
    basename_unique = _unique_or_none(basename_to_ids)

    mapped_motion_ids: list[int] = []
    mapped_bin_indices: list[int] = []
    mapped_starts: list[int] = []
    mapped_ends: list[int] = []
    mapped_caps: list[float] = []
    scale = max(float(cap_safety_scale), 0.0)

    for row_idx, source in enumerate(rows.source_paths.astype(str).tolist()):
        row_start = int(rows.start_frames[row_idx])
        row_end = int(rows.end_frames[row_idx])
        if row_end <= row_start:
            continue
        path_key = _norm_path_key(source)
        base_key = _basename_key(source)
        motion_id = None
        origin = 0

        windows = path_windows.get(path_key, [])
        contained = [item for item in windows if item[0] <= row_start and row_end <= item[1]]
        if len(contained) == 1:
            origin, _label_end, motion_id = contained[0]
        elif path_key in exact_unique:
            candidate_id = exact_unique[path_key]
            if not label_has_window[candidate_id]:
                motion_id = candidate_id
                origin = int(label_origins[motion_id])
        else:
            base_windows = basename_windows.get(base_key, [])
            contained = [item for item in base_windows if item[0] <= row_start and row_end <= item[1]]
            if len(contained) == 1:
                origin, _label_end, motion_id = contained[0]
            elif base_key in basename_unique:
                candidate_id = basename_unique[base_key]
                if not label_has_window[candidate_id]:
                    motion_id = candidate_id
                    origin = int(label_origins[motion_id])

        if motion_id is None:
            continue
        local_start = row_start - origin
        local_end = row_end - origin
        length = int(lengths[motion_id])
        local_start = max(0, min(local_start, max(length - 1, 0)))
        local_end = max(local_start + 1, min(local_end, max(length, 1)))
        cap = float(rows.caps_kg[row_idx]) * scale
        if not np.isfinite(cap) or cap < 0.0:
            continue
        mapped_motion_ids.append(int(motion_id))
        mapped_bin_indices.append(int(rows.bin_indices[row_idx]))
        mapped_starts.append(local_start)
        mapped_ends.append(local_end)
        mapped_caps.append(cap)

    return _MappedWindowCapacityRows(
        motion_ids=np.asarray(mapped_motion_ids, dtype=np.int64),
        bin_indices=np.asarray(mapped_bin_indices, dtype=np.int64),
        start_frames=np.asarray(mapped_starts, dtype=np.int64),
        end_frames=np.asarray(mapped_ends, dtype=np.int64),
        caps_kg=np.asarray(mapped_caps, dtype=np.float32),
    )
