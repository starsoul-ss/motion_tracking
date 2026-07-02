import json
import os
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from scipy.spatial.transform import Rotation as sRot
from tensordict import MemoryMappedTensor, TensorClass
from tqdm import tqdm

class MotionMinimalData(TensorClass):
    root_pos_w: torch.Tensor
    root_quat_w: torch.Tensor
    joint_pos: torch.Tensor


class MotionData(TensorClass):
    root_pos_w: torch.Tensor
    root_quat_w: torch.Tensor
    root_lin_vel_w: torch.Tensor
    root_ang_vel_w: torch.Tensor
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    body_pos_w: torch.Tensor
    body_pos_b: torch.Tensor
    body_vel_w: torch.Tensor
    body_vel_b: torch.Tensor
    body_quat_w: torch.Tensor
    body_quat_b: torch.Tensor
    body_angvel_w: torch.Tensor
    body_angvel_b: torch.Tensor


FOOT_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]
END_EFFECTOR_BODY_NAME_CANDIDATES = {
    "left_foot": ("left_ankle_roll_link",),
    "right_foot": ("right_ankle_roll_link",),
    "left_hand": ("left_hand_mimic", "left_wrist_pitch_link", "left_wrist_yaw_link", "left_rubber_hand"),
    "right_hand": ("right_hand_mimic", "right_wrist_pitch_link", "right_wrist_yaw_link", "right_rubber_hand"),
}
PAIRED_ENDPOINT_Z_DIFF_LIMIT_M = 0.2
PAIRED_ENDPOINT_Z_DIFF_CONTEXT_FRAMES = 25


def _normalize_motion_filter_result(result, length: int) -> tuple[bool, list[dict]]:
    if isinstance(result, dict):
        keep = bool(result.get("keep", True))
        if not keep:
            return False, []
        raw_intervals = result.get("bad_intervals", result.get("invalid_intervals", [])) or []
        intervals = []
        for item in raw_intervals:
            if isinstance(item, dict):
                start = item.get("start")
                end = item.get("end")
                reason = item.get("reason", "bad interval")
            else:
                if len(item) < 2:
                    raise ValueError(f"Bad interval entries need at least start/end, got {item}")
                start = item[0]
                end = item[1]
                reason = item[2] if len(item) >= 3 else "bad interval"
            start = max(0, min(length, int(start)))
            end = max(0, min(length, int(end)))
            if end > start:
                intervals.append({"start": start, "end": end, "reason": str(reason)})
        return True, intervals
    return bool(result), []


def _split_keep_intervals(
    raw_bad_intervals: list[dict],
    *,
    length: int,
    padding: int,
    min_segment_frames: int,
    max_segments: int | None = None,
) -> tuple[list[tuple[int, int]], list[dict]]:
    padded_bad_intervals = []
    for item in raw_bad_intervals:
        start = max(0, int(item["start"]) - padding)
        end = min(length, int(item["end"]) + padding)
        if end <= start:
            continue
        padded_bad_intervals.append({"start": start, "end": end, "reasons": {str(item["reason"])}})
    padded_bad_intervals.sort(key=lambda item: (item["start"], item["end"]))

    bad_intervals = []
    for item in padded_bad_intervals:
        if not bad_intervals or item["start"] > bad_intervals[-1]["end"]:
            bad_intervals.append(item)
            continue
        bad_intervals[-1]["end"] = max(bad_intervals[-1]["end"], item["end"])
        bad_intervals[-1]["reasons"].update(item["reasons"])

    for item in bad_intervals:
        item["reasons"] = sorted(item["reasons"])

    if not bad_intervals:
        return [(0, length)], []

    keep_intervals = []
    cursor = 0
    for item in bad_intervals:
        start = int(item["start"])
        end = int(item["end"])
        if start - cursor >= min_segment_frames:
            keep_intervals.append((cursor, start))
        cursor = max(cursor, end)
    if length - cursor >= min_segment_frames:
        keep_intervals.append((cursor, length))
    if max_segments is not None and max_segments > 0 and len(keep_intervals) > max_segments:
        longest_intervals = sorted(
            keep_intervals,
            key=lambda span: (span[1] - span[0], -span[0]),
            reverse=True,
        )[:max_segments]
        keep_intervals = sorted(longest_intervals)
    return keep_intervals, bad_intervals


def _mask_to_bad_intervals(mask: np.ndarray, *, context: int, length: int, reason: str) -> list[dict]:
    if not np.any(mask):
        return []
    padded = np.concatenate(([0], mask.astype(np.int8), [0]))
    edges = np.diff(padded)
    run_starts = np.where(edges == 1)[0]
    run_ends = np.where(edges == -1)[0]
    intervals = []
    for start, end in zip(run_starts, run_ends):
        interval_start = max(0, int(start) - context)
        interval_end = min(length, int(end) + context)
        if interval_end > interval_start:
            intervals.append({"start": interval_start, "end": interval_end, "reason": reason})
    return intervals


def _paired_endpoint_z_diff_intervals(
    teacher_motion: dict,
    student_motion: dict,
    body_names: Sequence[str],
    *,
    threshold_m: float = PAIRED_ENDPOINT_Z_DIFF_LIMIT_M,
    context_frames: int = PAIRED_ENDPOINT_Z_DIFF_CONTEXT_FRAMES,
) -> list[dict]:
    body_index = {name: idx for idx, name in enumerate(body_names)}
    indices = []
    missing_roles = []
    for role, candidates in END_EFFECTOR_BODY_NAME_CANDIDATES.items():
        for name in candidates:
            if name in body_index:
                indices.append(body_index[name])
                break
        else:
            missing_roles.append(f"{role}: {list(candidates)}")
    if missing_roles:
        raise ValueError(f"Paired endpoint z check could not resolve endpoint bodies: {missing_roles}")

    teacher_z = teacher_motion["xpos"][:, indices, 2]
    student_z = student_motion["xpos"][:, indices, 2]
    if teacher_z.shape != student_z.shape:
        raise ValueError(f"Paired endpoint z check shape mismatch: teacher={teacher_z.shape}, student={student_z.shape}")

    length = int(teacher_z.shape[0])
    over_limit = np.any(np.abs(student_z - teacher_z) > float(threshold_m), axis=1)
    return _mask_to_bad_intervals(
        over_limit,
        context=max(0, int(context_frames)),
        length=length,
        reason="paired endpoint z mismatch",
    )


def _reason_counts(intervals: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in intervals:
        reasons = item.get("reasons")
        if reasons is None:
            reasons = [item.get("reason", "bad interval")]
        for reason in reasons:
            reason = str(reason)
            counts[reason] = counts.get(reason, 0) + 1
    return counts


def _resolve_motion_paths(root_path: str) -> list[Path]:
    root = Path(root_path)
    if root.is_file() and root.suffix == ".npz":
        return [root]
    paths = sorted(root.rglob("*.npz"))
    if not paths:
        raise RuntimeError(f"No motions found in {root_path}")
    return paths


def _resolve_paired_motion_paths(root_path: str, student_root_path: str) -> list[tuple[Path, Path]]:
    teacher_root = Path(root_path)
    student_root = Path(student_root_path)
    if teacher_root.is_file() or student_root.is_file():
        if not (teacher_root.is_file() and student_root.is_file()):
            raise ValueError("root_path and student_root_path must both be files or both be directories")
        if teacher_root.suffix != ".npz" or student_root.suffix != ".npz":
            raise ValueError("root_path and student_root_path must point to .npz files")
        return [(teacher_root, student_root)]

    teacher_paths = sorted(teacher_root.rglob("*.npz"))
    student_paths = sorted(student_root.rglob("*.npz"))
    if not teacher_paths:
        raise RuntimeError(f"No motions found in {root_path}")
    if not student_paths:
        raise RuntimeError(f"No motions found in {student_root_path}")

    teacher_by_rel = {path.relative_to(teacher_root): path for path in teacher_paths}
    student_by_rel = {path.relative_to(student_root): path for path in student_paths}
    if set(teacher_by_rel) != set(student_by_rel):
        raise ValueError("root_path and student_root_path must contain the same .npz files")
    return [(teacher_by_rel[rel], student_by_rel[rel]) for rel in sorted(teacher_by_rel)]


def _resolve_dataset_mem_path(mem_path: str) -> str:
    path_root = os.environ.get("MEMPATH")
    if path_root is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(current_dir, "../..")
        path_root = os.path.join(project_root, "dataset")
    return os.path.join(path_root, mem_path)


def _prepare_motion_arrays(
    path: Path,
    joint_names: list[str] | None,
    body_names: list[str] | None,
    foot_idx: list[int] | None,
    target_fps: int,
    allow_body_subset: bool = False,
) -> tuple[dict, list[str], list[str], list[int]]:
    with np.load(path, allow_pickle=True) as data:
        if joint_names is None:
            joint_names = list(data["joint_names"].tolist())
            body_names = list(data["body_names"].tolist())
            foot_idx = [body_names.index(name) for name in FOOT_NAMES]
            body_indices = list(range(len(body_names)))
        else:
            current_joint_names = list(data["joint_names"].tolist())
            current_body_names = list(data["body_names"].tolist())
            if current_joint_names != joint_names:
                raise ValueError(f"joint_names mismatch in {path}")
            if current_body_names == body_names:
                body_indices = list(range(len(body_names)))
            elif allow_body_subset:
                current_body_index = {name: idx for idx, name in enumerate(current_body_names)}
                missing = [name for name in body_names if name not in current_body_index]
                if missing:
                    raise ValueError(f"body_names subset missing in {path}: {missing}")
                body_indices = [current_body_index[name] for name in body_names]
            else:
                raise ValueError(f"body_names mismatch in {path}")

        fps = int(data.get("mocap_framerate", data.get("frequency", data.get("fps", 0))))
        if fps != target_fps:
            raise ValueError(f"Expected fps={target_fps}, got {fps} for {path}")

        root_pos = np.asarray(data["root_pos"], dtype=np.float32)
        root_quat_xyzw = np.asarray(data["root_rot"], dtype=np.float32)
        joint_pos = np.asarray(data["dof_pos"][:, : len(joint_names)], dtype=np.float32)
        local_body_pos = np.asarray(data["local_body_pos"][:, body_indices], dtype=np.float32)

    root_rot = sRot.from_quat(root_quat_xyzw)
    root_rot_m = root_rot.as_matrix().astype(np.float32)
    body_pos_w = np.einsum("tij,tbj->tbi", root_rot_m, local_body_pos) + root_pos[:, None, :]

    root_quat_wxyz = np.concatenate([root_quat_xyzw[..., 3:4], root_quat_xyzw[..., :3]], axis=-1).astype(np.float32)
    qpos = np.concatenate([root_pos, root_quat_wxyz, joint_pos], axis=-1).astype(np.float32)

    motion = {
        "fps": fps,
        "qpos": qpos,
        "xpos": body_pos_w.astype(np.float32),
        "joint_names": joint_names,
        "body_names": body_names,
    }
    return motion, joint_names, body_names, foot_idx


def _read_motion_schema(path: Path) -> tuple[list[str], list[str]]:
    with np.load(path, allow_pickle=True) as data:
        return list(data["joint_names"].tolist()), list(data["body_names"].tolist())


def _resolve_body_subset_schema(paths, *, paired: bool) -> tuple[list[str], list[str], list[int]]:
    joint_names: list[str] | None = None
    ordered_body_names: list[str] | None = None
    common_body_names: set[str] | None = None

    for path_item in paths:
        item_paths = path_item if paired else (path_item,)
        for path in item_paths:
            current_joint_names, current_body_names = _read_motion_schema(path)
            if joint_names is None:
                joint_names = current_joint_names
                ordered_body_names = current_body_names
                common_body_names = set(current_body_names)
                continue
            if current_joint_names != joint_names:
                raise ValueError(f"joint_names mismatch in {path}")
            common_body_names &= set(current_body_names)

    if joint_names is None or ordered_body_names is None or common_body_names is None:
        raise RuntimeError("No motion schemas found")

    body_names = [name for name in ordered_body_names if name in common_body_names]
    missing_feet = [name for name in FOOT_NAMES if name not in body_names]
    if missing_feet:
        raise ValueError(f"Common body subset is missing required foot links: {missing_feet}")
    foot_idx = [body_names.index(name) for name in FOOT_NAMES]
    return joint_names, body_names, foot_idx


def _write_motion_dataset(
    *,
    joint_names: list[str],
    body_names: list[str],
    metadata_rows: list[dict],
    id_labels: list[dict],
    lengths: list[int],
    root_pos_chunks: list[torch.Tensor],
    root_quat_chunks: list[torch.Tensor],
    joint_pos_chunks: list[torch.Tensor],
    mem_path: str | None,
):
    info = {}
    keys = sorted({key for row in metadata_rows for key in row.keys()})
    for key in keys:
        values = [row.get(key) for row in metadata_rows]
        arr = np.asarray(values)
        info[key] = arr.reshape(len(metadata_rows), -1).tolist()

    meta = {
        "joint_names": joint_names,
        "body_names": body_names,
        "info": info,
    }

    starts = []
    ends = []
    cursor = 0
    for length in lengths:
        starts.append(cursor)
        cursor += length
        ends.append(cursor)

    data = MotionMinimalData(
        root_pos_w=torch.cat(root_pos_chunks, dim=0),
        root_quat_w=torch.cat(root_quat_chunks, dim=0),
        joint_pos=torch.cat(joint_pos_chunks, dim=0),
        batch_size=[cursor],
    )

    if mem_path is not None:
        path = Path(mem_path)
        path.mkdir(parents=True, exist_ok=True)
        data.memmap(str(path / "_tensordict"))
        dump_data = {
            "joint_names": meta["joint_names"],
            "body_names": meta["body_names"],
            "starts": starts,
            "ends": ends,
            "info": meta["info"],
        }
        with (path / "meta_motion.json").open("w") as f:
            json.dump(dump_data, f)
        with (path / "id_label.json").open("w") as f:
            json.dump(id_labels, f, ensure_ascii=True)

    return data, meta


def create_motion_dataset_from_path(
    root_path: str,
    target_fps: int = 50,
    mem_path: str | None = None,
    motion_processer: Callable | None = None,
    motion_filter: Callable | None = None,
    student_root_path: str | None = None,
    student_mem_path: str | None = None,
    *,
    storage_float_dtype: torch.dtype = torch.float16,
    storage_int_dtype: torch.dtype = torch.int32,
    allow_body_subset: bool = False,
    bad_interval_padding: int = 10,
    min_segment_frames: int = 150,
    max_segments_per_motion: int | None = None,
):
    del storage_int_dtype  # kept for call-site compatibility
    bad_interval_padding = max(0, int(bad_interval_padding))
    min_segment_frames = max(1, int(min_segment_frames))
    if max_segments_per_motion is not None:
        max_segments_per_motion = int(max_segments_per_motion)
        if max_segments_per_motion <= 0:
            max_segments_per_motion = None
    paired = student_root_path is not None
    if not paired and student_mem_path is not None:
        raise ValueError("student_mem_path requires student_root_path")
    paths = _resolve_paired_motion_paths(root_path, student_root_path) if paired else _resolve_motion_paths(root_path)
    if allow_body_subset:
        joint_names, body_names, foot_idx = _resolve_body_subset_schema(paths, paired=paired)
        first_schema_path = paths[0][0] if paired else paths[0]
        _, first_body_names = _read_motion_schema(first_schema_path)
        if len(body_names) != len(first_body_names):
            print(
                f"Using common body subset: {len(body_names)}/{len(first_body_names)} bodies "
                f"from first schema {first_schema_path}"
            )
    else:
        joint_names = None
        body_names = None
        foot_idx = None
    metadata_rows: list[dict] = []
    id_labels: list[dict] = []
    lengths: list[int] = []
    root_pos_chunks: list[torch.Tensor] = []
    root_quat_chunks: list[torch.Tensor] = []
    joint_pos_chunks: list[torch.Tensor] = []
    student_root_pos_chunks: list[torch.Tensor] = []
    student_root_quat_chunks: list[torch.Tensor] = []
    student_joint_pos_chunks: list[torch.Tensor] = []

    load_bar = tqdm(paths, desc="load")
    for path_item in load_bar:
        if paired:
            teacher_path, student_path = path_item
        else:
            teacher_path = path_item
            student_path = None

        motion, joint_names, body_names, foot_idx = _prepare_motion_arrays(
            teacher_path, joint_names, body_names, foot_idx, target_fps, allow_body_subset=allow_body_subset
        )
        start_idx = 0
        end_idx = int(motion["qpos"].shape[0])
        student_motion = None

        if motion_processer is not None:
            motion = motion_processer(motion, foot_idx, teacher_path, start_idx, end_idx)
        if paired:
            student_motion, _, _, _ = _prepare_motion_arrays(
                student_path,
                joint_names,
                body_names,
                foot_idx,
                target_fps,
                allow_body_subset=allow_body_subset,
            )
            if motion_processer is not None:
                student_motion = motion_processer(student_motion, foot_idx, student_path, start_idx, end_idx)
            if int(motion["qpos"].shape[0]) != int(student_motion["qpos"].shape[0]):
                raise ValueError(
                    f"Paired motions must have the same length after preprocessing: "
                    f"{teacher_path} vs {student_path}"
                )

        qpos = motion["qpos"]
        length = int(qpos.shape[0])
        if motion_filter is not None:
            filter_result = motion_filter(motion, foot_idx, teacher_path, start_idx, end_idx)
            keep, raw_bad_intervals = _normalize_motion_filter_result(filter_result, length)
            if not keep:
                continue
        else:
            raw_bad_intervals = []
        if paired:
            raw_bad_intervals.extend(
                _paired_endpoint_z_diff_intervals(
                    motion,
                    student_motion,
                    body_names,
                )
            )

        keep_intervals, bad_intervals = _split_keep_intervals(
            raw_bad_intervals,
            length=length,
            padding=bad_interval_padding,
            min_segment_frames=min_segment_frames,
            max_segments=max_segments_per_motion,
        )
        if bad_intervals:
            if not keep_intervals:
                load_bar.write(f"Invalid motion due to no valid segment after interval filtering: {teacher_path}")
                continue
            dropped = length - sum(end - start for start, end in keep_intervals)
            load_bar.write(
                f"Segmented motion due to {_reason_counts(bad_intervals)}: "
                f"{teacher_path} -> {len(keep_intervals)} segment(s), dropped {dropped}/{length} frames"
            )
        joint_count = len(joint_names)
        if paired:
            student_qpos = student_motion["qpos"]
            if student_qpos.shape[1] != qpos.shape[1]:
                raise ValueError(
                    f"Paired motions must have the same qpos width after preprocessing: "
                    f"{teacher_path} vs {student_path}"
                )

        segment_count = len(keep_intervals)
        for segment_index, (segment_start, segment_end) in enumerate(keep_intervals):
            qpos_segment = qpos[segment_start:segment_end]
            segment_length = int(qpos_segment.shape[0])
            lengths.append(segment_length)
            metadata_rows.append(motion.get("metadata") or {})
            label = {
                "source_path": str(teacher_path),
                "segment_start": int(start_idx + segment_start),
                "segment_end": int(start_idx + segment_end),
            }
            if bad_intervals:
                label.update(
                    {
                        "source_motion_start": int(start_idx),
                        "source_motion_end": int(end_idx),
                        "source_segment_index": int(segment_index),
                        "source_segment_count": int(segment_count),
                        "source_bad_interval_count": int(len(bad_intervals)),
                        "source_bad_interval_reasons": sorted(_reason_counts(bad_intervals).keys()),
                    }
                )
            if paired:
                student_qpos_segment = student_qpos[segment_start:segment_end]
                label["student_source_path"] = str(student_path)
            id_labels.append(label)
            root_pos_chunks.append(torch.as_tensor(qpos_segment[:, :3], dtype=storage_float_dtype))
            root_quat_chunks.append(torch.as_tensor(qpos_segment[:, 3:7], dtype=storage_float_dtype))
            joint_pos_chunks.append(torch.as_tensor(qpos_segment[:, 7:7 + joint_count], dtype=storage_float_dtype))
            if paired:
                student_root_pos_chunks.append(torch.as_tensor(student_qpos_segment[:, :3], dtype=storage_float_dtype))
                student_root_quat_chunks.append(torch.as_tensor(student_qpos_segment[:, 3:7], dtype=storage_float_dtype))
                student_joint_pos_chunks.append(
                    torch.as_tensor(student_qpos_segment[:, 7:7 + joint_count], dtype=storage_float_dtype)
                )
        if paired:
            del student_qpos
        load_bar.set_postfix(total=sum(lengths), motions=len(lengths))

    if joint_names is None or body_names is None or len(lengths) == 0:
        raise RuntimeError(f"No valid motions found in {root_path}")

    teacher = _write_motion_dataset(
        joint_names=joint_names,
        body_names=body_names,
        metadata_rows=metadata_rows,
        id_labels=id_labels,
        lengths=lengths,
        root_pos_chunks=root_pos_chunks,
        root_quat_chunks=root_quat_chunks,
        joint_pos_chunks=joint_pos_chunks,
        mem_path=mem_path,
    )
    if not paired:
        return teacher

    student = _write_motion_dataset(
        joint_names=joint_names,
        body_names=body_names,
        metadata_rows=metadata_rows,
        id_labels=id_labels,
        lengths=lengths,
        root_pos_chunks=student_root_pos_chunks,
        root_quat_chunks=student_root_quat_chunks,
        joint_pos_chunks=student_joint_pos_chunks,
        mem_path=student_mem_path,
    )
    return teacher, student


def _load_motion_memmap(
    mem_path: str,
    dataset_extra_keys: Sequence[dict] = (),
    ds_device: torch.device = torch.device("cpu"),
) -> dict:
    resolved_path = _resolve_dataset_mem_path(mem_path)
    data_mm = MotionMinimalData.load_memmap(os.path.join(resolved_path, "_tensordict"))
    data = MotionMinimalData(
        root_pos_w=data_mm.root_pos_w.to(device=ds_device, dtype=torch.float32),
        root_quat_w=data_mm.root_quat_w.to(device=ds_device, dtype=torch.float32),
        joint_pos=data_mm.joint_pos.to(device=ds_device, dtype=torch.float32),
        batch_size=list(data_mm.batch_size),
        device=ds_device,
    )
    with open(os.path.join(resolved_path, "meta_motion.json"), "r") as f:
        meta = json.load(f)

    starts = torch.tensor(meta["starts"], dtype=torch.long, device=ds_device)
    ends = torch.tensor(meta["ends"], dtype=torch.long, device=ds_device)
    id_label_path = os.path.join(resolved_path, "id_label.json")
    if os.path.exists(id_label_path):
        with open(id_label_path, "r") as f:
            id_labels = json.load(f)
        if len(id_labels) != len(meta["starts"]):
            raise ValueError(
                f"id_label.json length mismatch in {resolved_path}: "
                f"{len(id_labels)} != {len(meta['starts'])}"
            )
        source_paths = [str(row.get("source_path", "")) for row in id_labels]
    else:
        id_labels = [
            {
                "source_path": os.path.join(resolved_path, f"motion_{i:06d}.npz"),
                "segment_start": 0,
                "segment_end": int((ends[i] - starts[i]).item()),
            }
            for i in range(len(meta["starts"]))
        ]
        source_paths = [str(row["source_path"]) for row in id_labels]
    info = {}
    for k in dataset_extra_keys:
        name = k["name"]
        shape = k["shape"]
        if name not in meta["info"]:
            info[name] = torch.zeros((len(meta["starts"]), shape), dtype=k["dtype"], device=ds_device)
        else:
            info[name] = torch.tensor(meta["info"][name], dtype=k["dtype"], device=ds_device)
        if info[name].shape != (len(meta["starts"]), shape):
            raise ValueError(f"Shape of {name} does not match: {info[name].shape} != {(len(meta['starts']), shape)}")

    return {
        "joint_names": list(meta["joint_names"]),
        "body_names": list(meta.get("body_names", [])),
        "starts": starts,
        "ends": ends,
        "data": data,
        "info": info,
        "resolved_path": resolved_path,
        "source_paths": source_paths,
        "id_labels": id_labels,
    }


class GlobalMotionDataset:
    def __init__(
        self,
        *,
        joint_names: list[str],
        body_names: list[str],
        starts: torch.Tensor,
        ends: torch.Tensor,
        data: MotionMinimalData,
        info: dict[str, torch.Tensor],
        motion_to_dataset_id: torch.Tensor,
        dataset_counts: torch.Tensor,
        dataset_paths: list[str],
        motion_source_paths: list[str],
        motion_labels: list[dict],
        ds_device: torch.device = torch.device("cpu"),
    ):
        self.joint_names = joint_names
        self.body_names = body_names
        self.starts = starts.to(dtype=torch.long, device=ds_device)
        self.ends = ends.to(dtype=torch.long, device=ds_device)
        self.lengths = self.ends - self.starts
        self.data = data
        self.info = info
        self.motion_to_dataset_id = motion_to_dataset_id.to(dtype=torch.long, device=ds_device)
        self.dataset_counts = dataset_counts.to(dtype=torch.long, device=ds_device)
        self.dataset_paths = list(dataset_paths)
        self.motion_source_paths = list(motion_source_paths)
        self.motion_labels = list(motion_labels)
        self.ds_device = ds_device

    def trim_initial_frames(self, skip_initial_frames: int):
        skip = int(skip_initial_frames)
        if skip < 0:
            raise ValueError("skip_initial_frames must be non-negative")
        if skip == 0:
            return
        too_short = skip >= self.lengths
        if too_short.any():
            motion_id = int(too_short.nonzero(as_tuple=False)[0].item())
            raise ValueError(
                f"skip_initial_frames={skip} leaves no frames in motion {motion_id} "
                f"(length={int(self.lengths[motion_id].item())})"
            )
        self.starts = self.starts + skip
        self.lengths = self.ends - self.starts

    @classmethod
    def load_from_mem_paths(
        cls,
        mem_paths: Sequence[str],
        dataset_extra_keys: Sequence[dict] = (),
        ds_device: torch.device = torch.device("cpu"),
    ):
        if len(mem_paths) == 0:
            raise ValueError("mem_paths must be non-empty")
        shards = [_load_motion_memmap(p, dataset_extra_keys=dataset_extra_keys, ds_device=ds_device) for p in mem_paths]
        ref_joint_names = list(shards[0]["joint_names"])
        ref_body_names = list(shards[0]["body_names"])
        for i, shard in enumerate(shards[1:], start=1):
            if list(shard["joint_names"]) != ref_joint_names:
                raise ValueError(
                    f"joint_names mismatch between mem_paths[0]='{mem_paths[0]}' and mem_paths[{i}]='{mem_paths[i]}'"
                )

        cursor = 0
        starts = []
        ends = []
        motion_to_dataset_id = []
        dataset_counts = []
        info_chunks = {k["name"]: [] for k in dataset_extra_keys}
        root_pos_chunks = []
        root_quat_chunks = []
        joint_pos_chunks = []
        motion_source_paths = []
        motion_labels = []
        for dataset_idx, shard in enumerate(shards):
            shard_starts = shard["starts"] + cursor
            shard_ends = shard["ends"] + cursor
            starts.append(shard_starts)
            ends.append(shard_ends)
            motion_count = int(shard_starts.numel())
            dataset_counts.append(motion_count)
            motion_to_dataset_id.append(torch.full((motion_count,), dataset_idx, dtype=torch.long, device=ds_device))
            for extra in dataset_extra_keys:
                info_chunks[extra["name"]].append(shard["info"][extra["name"]])
            root_pos_chunks.append(shard["data"].root_pos_w)
            root_quat_chunks.append(shard["data"].root_quat_w)
            joint_pos_chunks.append(shard["data"].joint_pos)
            motion_source_paths.extend(shard["source_paths"])
            motion_labels.extend(shard["id_labels"])
            cursor += int(shard["data"].root_pos_w.shape[0])

        data = MotionMinimalData(
            root_pos_w=torch.cat(root_pos_chunks, dim=0),
            root_quat_w=torch.cat(root_quat_chunks, dim=0),
            joint_pos=torch.cat(joint_pos_chunks, dim=0),
            batch_size=[cursor],
            device=ds_device,
        )
        info = {
            extra["name"]: torch.cat(info_chunks[extra["name"]], dim=0) if info_chunks[extra["name"]] else torch.zeros((0, extra["shape"]), dtype=extra["dtype"], device=ds_device)
            for extra in dataset_extra_keys
        }
        return cls(
            joint_names=ref_joint_names,
            body_names=ref_body_names,
            starts=torch.cat(starts, dim=0),
            ends=torch.cat(ends, dim=0),
            data=data,
            info=info,
            motion_to_dataset_id=torch.cat(motion_to_dataset_id, dim=0),
            dataset_counts=torch.tensor(dataset_counts, dtype=torch.long, device=ds_device),
            dataset_paths=list(mem_paths),
            motion_source_paths=motion_source_paths,
            motion_labels=motion_labels,
            ds_device=ds_device,
        )

    @property
    def num_motions(self):
        return int(self.starts.numel())

    @property
    def num_steps(self):
        return int(self.data.root_pos_w.shape[0])

    def get_slice(
        self,
        motion_ids: torch.Tensor,
        starts: torch.Tensor,
        steps: int | torch.Tensor = 1,
        target_device: torch.device | None = None,
    ) -> MotionMinimalData:
        motion_ids = motion_ids.to(device=self.ds_device, dtype=torch.long)
        starts = starts.to(device=self.ds_device, dtype=torch.long)
        starts_per_motion = self.starts[motion_ids].unsqueeze(1)
        if isinstance(steps, int):
            idx = starts_per_motion + starts.unsqueeze(1) + torch.arange(steps, device=self.ds_device, dtype=torch.long)
        else:
            idx = starts_per_motion + starts.unsqueeze(1) + steps.to(device=self.ds_device, dtype=torch.long).unsqueeze(0)
        ends_per_motion = (self.ends[motion_ids] - 1).unsqueeze(1)
        idx.clamp_min_(starts_per_motion).clamp_max_(ends_per_motion)
        out = self.data[idx]
        return out.to(target_device)
