import argparse
from functools import partial
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as sRot

from active_adaptation.utils.motion import (
    create_motion_dataset_from_path,
)

EXCLUDE_LABEL_PATH = Path(__file__).parent / "label.txt"
SEED_KEEP_FILENAMES_PATH = Path("/home/axell/Desktop/dataset_new/retarget_g1/seed/keep_filenames.txt")
EXCLUDED_PATHS = set()
SEED_KEEP_FILENAMES = set()
ENABLE_AMASS_FILTER = False
ENABLE_SEED_FILTER = False
VELOCITY_CHECK_FPS = 10.0
ROOT_LINEAR_VEL_LIMIT = 8.0
ROOT_ANGULAR_VEL_LIMIT = 15.0
JOINT_VEL_LIMIT = 20.0
MIN_LINK_Z_LIMIT = -0.10
MIN_MOTION_FRAMES = 150
EXCLUDED_SUBSTRINGS = [
    "CMU/94",
    "CMU/126",
    "chair",
    "HDM05/tr",
    "HDM05/bk",
    "SShapeRL",
    "SShapeLR",
    "CircleCCW",
    "KIT/1226",
]


def preprocess_motion(motion, foot_idx, _path, _start_idx, _end_idx, always_on_ground: bool = False):
    root_pos = motion["qpos"][:, :3]  # (T,3)
    offset_xy = root_pos[0, :2].copy()  # 首帧 x,y
    motion["qpos"][:, 0] -= offset_xy[0]
    motion["qpos"][:, 1] -= offset_xy[1]
    motion["xpos"][:, :, 0] -= offset_xy[0]
    motion["xpos"][:, :, 1] -= offset_xy[1]

    # z_l = motion["xpos"][:, foot_idx[0], 2]
    # z_r = motion["xpos"][:, foot_idx[1], 2]

    # if not always_on_ground:
    #     z_min = float(min(z_l.min(), z_r.min()))
    #     target_z0 = 0.0
    #     dz = target_z0 - z_min
    #     motion["qpos"][:, 2] += dz
    #     motion["xpos"][:, :, 2] += dz
    # else:
    #     z_min = np.min(
    #         np.concatenate([z_l.reshape(-1, 1), z_r.reshape(-1, 1)], axis=1),
    #         axis=-1,
    #         keepdims=True,
    #     )
    #     target_z0 = 0.0
    #     dz = target_z0 - z_min
    #     motion["qpos"][:, 2] += dz.reshape(-1)
    #     motion["xpos"][:, :, 2] += dz
    return motion


def load_excluded_paths(label_path: Path) -> set[str]:
    if not label_path.exists():
        return set()
    paths = set()
    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            token = line.split("\t", 1)[0]
            token = token.split(" ", 1)[0]
            if token:
                paths.add(token)
    return paths

def load_keep_filenames(list_path: Path) -> set[str]:
    if not list_path.exists():
        return set()
    names = set()
    with list_path.open("r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if name:
                names.add(name)
    return names


def _strip_seed_segment_suffix(stem: str) -> str:
    base, sep, suffix = stem.rpartition("_")
    if sep and suffix.isdigit():
        return base
    return stem


def _mask_to_velocity_intervals(mask: np.ndarray, frame_step: int, frame_count: int, reason: str) -> list[dict]:
    if not np.any(mask):
        return []
    padded = np.concatenate(([0], mask.astype(np.int8), [0]))
    edges = np.diff(padded)
    run_starts = np.where(edges == 1)[0]
    run_ends = np.where(edges == -1)[0]
    return [
        {
            "start": int(start),
            "end": int(min(frame_count, end + frame_step)),
            "reason": reason,
        }
        for start, end in zip(run_starts, run_ends)
    ]


def _mask_to_frame_intervals(mask: np.ndarray, frame_count: int, reason: str) -> list[dict]:
    if not np.any(mask):
        return []
    padded = np.concatenate(([0], mask.astype(np.int8), [0]))
    edges = np.diff(padded)
    run_starts = np.where(edges == 1)[0]
    run_ends = np.where(edges == -1)[0]
    return [
        {
            "start": int(start),
            "end": int(min(frame_count, end)),
            "reason": reason,
        }
        for start, end in zip(run_starts, run_ends)
    ]


def _low_link_z_intervals(motion) -> list[dict]:
    if MIN_LINK_Z_LIMIT is None:
        return []
    xpos = motion["xpos"]
    frame_count = int(xpos.shape[0])
    min_link_z = np.min(xpos[:, :, 2], axis=1)
    return _mask_to_frame_intervals(
        min_link_z < float(MIN_LINK_Z_LIMIT),
        frame_count=frame_count,
        reason=f"min link z < {float(MIN_LINK_Z_LIMIT):.2f}m",
    )


def _window_linear_velocity(values: np.ndarray, fps: int, frame_step: int) -> np.ndarray:
    if values.shape[0] <= frame_step:
        return np.zeros((0,) + values.shape[1:], dtype=np.float32)
    return ((values[frame_step:] - values[:-frame_step]) * (float(fps) / float(frame_step))).astype(np.float32)


def _window_root_angular_velocity(root_quat_wxyz: np.ndarray, fps: int, frame_step: int) -> np.ndarray:
    if root_quat_wxyz.shape[0] <= frame_step:
        return np.zeros((0, 3), dtype=np.float32)
    root_quat_xyzw = np.concatenate([root_quat_wxyz[:, 1:], root_quat_wxyz[:, :1]], axis=-1)
    root_rot = sRot.from_quat(root_quat_xyzw)
    delta_rot = root_rot[frame_step:] * root_rot[:-frame_step].inv()
    return (delta_rot.as_rotvec() * (float(fps) / float(frame_step))).astype(np.float32)


def _velocity_spike_intervals(motion) -> list[dict]:
    qpos = motion["qpos"]
    fps = int(motion.get("fps", 0))
    if fps <= 0:
        fps = 50
    frame_step = 1
    if VELOCITY_CHECK_FPS > 0:
        frame_step = max(1, int(round(float(fps) / float(VELOCITY_CHECK_FPS))))

    root_lin_vel = _window_linear_velocity(qpos[:, :3], fps, frame_step)
    root_ang_vel = _window_root_angular_velocity(qpos[:, 3:7], fps, frame_step)
    joint_vel = _window_linear_velocity(qpos[:, 7:], fps, frame_step)

    frame_count = int(qpos.shape[0])
    intervals = []
    intervals.extend(
        _mask_to_velocity_intervals(
            np.linalg.norm(root_lin_vel, axis=1) > ROOT_LINEAR_VEL_LIMIT,
            frame_step,
            frame_count,
            "high root linear velocity spike",
        )
    )
    intervals.extend(
        _mask_to_velocity_intervals(
            np.linalg.norm(root_ang_vel, axis=1) > ROOT_ANGULAR_VEL_LIMIT,
            frame_step,
            frame_count,
            "high root angular velocity spike",
        )
    )
    intervals.extend(
        _mask_to_velocity_intervals(
            np.any(np.abs(joint_vel) > JOINT_VEL_LIMIT, axis=1),
            frame_step,
            frame_count,
            "high joint velocity spike",
        )
    )
    return intervals


def check_motion(motion, foot_idx, path, start_idx, end_idx):
    """Return False when the motion violates basic physical sanity checks."""

    def reject(reason: str) -> bool:
        print(f"Invalid motion due to {reason}: {path}")
        return False

    qpos = motion["qpos"]
    xpos = motion["xpos"]

    path_str = str(path)
    if ENABLE_AMASS_FILTER:
        if EXCLUDED_PATHS and path_str in EXCLUDED_PATHS:
            return reject("excluded path")
        if any(s in path_str for s in EXCLUDED_SUBSTRINGS):
            return reject("excluded substring")
    seed_stem = _strip_seed_segment_suffix(path.stem)
    if ENABLE_SEED_FILTER and seed_stem.endswith("_M"):
        return reject("mirrored seed motion")
    if ENABLE_SEED_FILTER and SEED_KEEP_FILENAMES and seed_stem not in SEED_KEEP_FILENAMES:
        return reject("missing keep filename")

    if qpos.shape[0] < MIN_MOTION_FRAMES:
        return reject("short length")

    min_body_z = np.min(xpos[:, :, 2], axis=1)
    all_off = min_body_z > 0.2
    fps = int(motion.get("fps", 0))
    if fps <= 0:
        fps = 50
    if np.any(all_off):
        padded = np.concatenate(([0], all_off.astype(np.int8), [0]))
        edges = np.diff(padded)
        run_starts = np.where(edges == 1)[0]
        run_ends = np.where(edges == -1)[0]
        max_run = (run_ends - run_starts).max() if run_starts.size else 0
        if max_run > fps:
            return reject("all bodies off ground > 1s")
    max_body_z = float(np.max(xpos[:, :, 2]))
    if max_body_z <= 0.2:
        return reject("low max body height")

    bad_intervals = []
    bad_intervals.extend(_velocity_spike_intervals(motion))
    bad_intervals.extend(_low_link_z_intervals(motion))
    if bad_intervals:
        return {"keep": True, "bad_intervals": bad_intervals}
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", help="NPZ file or directory to convert")
    ap.add_argument("--mem-path", help="Output memmap directory")
    ap.add_argument("--student-root", help="Optional paired student NPZ file or directory")
    ap.add_argument("--student-mem-path", help="Output student memmap directory for paired data")
    ap.add_argument("--amass-filter", action="store_true", help="Enable AMASS-specific path/name filters")
    ap.add_argument("--seed-filter", action="store_true", help="Enable seed keep_filenames allowlist filter")
    ap.add_argument("--velocity-check-fps", type=float, default=10.0, help="FPS used for segmentable velocity checks")
    ap.add_argument(
        "--min-link-z-threshold",
        type=float,
        default=-0.10,
        help="Mark frames with any link z below this threshold as bad intervals",
    )
    ap.add_argument("--bad-interval-padding", type=int, default=10, help="Frames to drop before/after bad intervals")
    ap.add_argument("--min-segment-frames", type=int, default=150, help="Minimum kept segment length after filtering")
    ap.add_argument(
        "--max-segments-per-motion",
        type=int,
        default=2,
        help="Keep at most this many longest valid segments after interval filtering",
    )
    ap.add_argument(
        "--allow-body-subset",
        action="store_true",
        help="Allow motions with different body_names by using their common named body subset for checks.",
    )
    args = ap.parse_args()

    global EXCLUDED_PATHS, SEED_KEEP_FILENAMES, ENABLE_AMASS_FILTER, ENABLE_SEED_FILTER, VELOCITY_CHECK_FPS
    global MIN_LINK_Z_LIMIT
    global MIN_MOTION_FRAMES
    ENABLE_AMASS_FILTER = args.amass_filter
    ENABLE_SEED_FILTER = args.seed_filter
    VELOCITY_CHECK_FPS = args.velocity_check_fps
    MIN_LINK_Z_LIMIT = args.min_link_z_threshold
    MIN_MOTION_FRAMES = args.min_segment_frames
    EXCLUDED_PATHS = load_excluded_paths(EXCLUDE_LABEL_PATH) if ENABLE_AMASS_FILTER else set()
    SEED_KEEP_FILENAMES = load_keep_filenames(SEED_KEEP_FILENAMES_PATH) if ENABLE_SEED_FILTER else set()

    preprocess = partial(preprocess_motion, always_on_ground=False)

    if args.dataset_root is None or args.mem_path is None:
        raise ValueError("Dataset generation requires both --dataset-root and --mem-path")
    if args.student_root is None and args.student_mem_path is not None:
        raise ValueError("--student-mem-path requires --student-root")
    if args.student_root is not None and args.student_mem_path is None:
        raise ValueError("--student-root requires --student-mem-path")

    dataset_root = Path(args.dataset_root)
    create_motion_dataset_from_path(
        str(dataset_root),
        target_fps=50,
        mem_path=args.mem_path,
        motion_processer=preprocess,
        motion_filter=check_motion,
        student_root_path=args.student_root,
        student_mem_path=args.student_mem_path,
        storage_float_dtype=torch.float16,
        storage_int_dtype=torch.int32,
        allow_body_subset=args.allow_body_subset,
        bad_interval_padding=args.bad_interval_padding,
        min_segment_frames=args.min_segment_frames,
        max_segments_per_motion=args.max_segments_per_motion,
    )


if __name__ == "__main__":
    main()
