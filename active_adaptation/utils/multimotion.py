import re
from typing import Any, List, Sequence

import torch

from active_adaptation.utils.fk_helper import MotionFKHelper
from active_adaptation.utils.motion import GlobalMotionDataset, MotionData


SAMPLING_MODES = {"motion_uniform", "length_uniform"}


def _select_required_body_names(asset_body_names: Sequence[str], required_motion_body_patterns: Sequence[str]) -> list[str]:
    if len(required_motion_body_patterns) == 0:
        raise ValueError("required_motion_body_patterns must be provided and non-empty.")
    body_names = [
        name for name in asset_body_names
        if any(re.match(pattern, name) for pattern in required_motion_body_patterns)
    ]
    if len(body_names) == 0:
        raise ValueError(
            "required_motion_body_patterns did not match any asset bodies: "
            f"patterns={list(required_motion_body_patterns)}, asset_body_names={list(asset_body_names)}"
        )
    return body_names


def _normalize_sampling_mode(raw_mode: Any) -> str:
    mode = str(raw_mode)
    if mode not in SAMPLING_MODES:
        raise ValueError(f"Unknown dataset sampling_mode {mode!r}; expected one of {sorted(SAMPLING_MODES)}")
    return mode


def _normalize_dataset_groups(
    *,
    groups: Sequence[dict],
) -> list[dict]:
    normalized_groups = []
    for i, raw_group in enumerate(groups):
        group = dict(raw_group)
        if "teacher_mem_path" not in group:
            raise ValueError(f"dataset.groups[{i}].teacher_mem_path is required")
        if "weight" not in group:
            raise ValueError(f"dataset.groups[{i}].weight is required")
        teacher_mem_path = str(group["teacher_mem_path"])
        student_mem_path = group.get("student_mem_path")
        normalized_groups.append(
            {
                "name": str(group.get("name", teacher_mem_path)),
                "teacher_mem_path": teacher_mem_path,
                "student_mem_path": (None if student_mem_path is None else str(student_mem_path)),
                "weight": float(group["weight"]),
                "sampling_mode": _normalize_sampling_mode(group.get("sampling_mode", "motion_uniform")),
            }
        )
    if len(normalized_groups) == 0:
        raise ValueError("dataset.groups must be non-empty")
    return normalized_groups


def _build_motion_sample_probs(
    *,
    groups: Sequence[dict],
    teacher_dataset: GlobalMotionDataset,
    device: torch.device,
) -> torch.Tensor:
    weights = torch.tensor([group["weight"] for group in groups], dtype=torch.float32, device=device)
    dataset_probs = (weights / weights.sum()).float()
    motion_to_dataset_id = teacher_dataset.motion_to_dataset_id.to(device)
    global_lengths = teacher_dataset.lengths.to(device)
    motion_lengths_f = global_lengths.float().clamp_min(1.0)

    motion_sample_probs = torch.zeros_like(motion_lengths_f)
    for dataset_id, group in enumerate(groups):
        mask = motion_to_dataset_id == dataset_id
        if not torch.any(mask):
            continue
        if group["sampling_mode"] == "motion_uniform":
            motion_sample_probs[mask] = dataset_probs[dataset_id] / mask.sum().float().clamp_min(1.0)
        elif group["sampling_mode"] == "length_uniform":
            group_lengths = motion_lengths_f[mask]
            motion_sample_probs[mask] = dataset_probs[dataset_id] * (
                group_lengths / group_lengths.sum().clamp_min(1.0)
            )
        else:
            raise ValueError(f"Unknown dataset sampling_mode {group['sampling_mode']!r}")
    return motion_sample_probs / motion_sample_probs.sum()


def _build_student_pair_mapping(
    *,
    groups: Sequence[dict],
    teacher_dataset: GlobalMotionDataset,
    student_dataset: GlobalMotionDataset | None,
    device: torch.device,
    ds_device: torch.device,
) -> torch.Tensor:
    teacher_to_student_motion_id = torch.full(
        (teacher_dataset.num_motions,),
        -1,
        dtype=torch.long,
        device=device,
    )
    if student_dataset is None:
        return teacher_to_student_motion_id

    if list(student_dataset.joint_names) != list(teacher_dataset.joint_names):
        raise ValueError("Paired student dataset joint_names must match teacher dataset joint_names")

    teacher_motion_offset = 0
    student_motion_offset = 0
    student_dataset_idx = 0
    teacher_lengths = teacher_dataset.lengths.to(ds_device)
    student_lengths = student_dataset.lengths.to(ds_device)

    for group_idx, group in enumerate(groups):
        teacher_count = int(teacher_dataset.dataset_counts[group_idx].item())
        teacher_slice = slice(teacher_motion_offset, teacher_motion_offset + teacher_count)
        if group["student_mem_path"] is not None:
            if student_dataset_idx >= int(student_dataset.dataset_counts.numel()):
                raise ValueError(f"Missing paired student shard for group '{group['name']}'")
            student_count = int(student_dataset.dataset_counts[student_dataset_idx].item())
            if teacher_count != student_count:
                raise ValueError(
                    f"Paired group '{group['name']}' motion count mismatch: teacher={teacher_count} student={student_count}"
                )
            student_slice = slice(student_motion_offset, student_motion_offset + student_count)
            if not torch.equal(teacher_lengths[teacher_slice], student_lengths[student_slice]):
                raise ValueError(
                    f"Paired group '{group['name']}' motion lengths must match between teacher and student datasets"
                )
            teacher_to_student_motion_id[teacher_slice] = torch.arange(
                student_slice.start,
                student_slice.stop,
                dtype=torch.long,
                device=device,
            )
            student_motion_offset += student_count
            student_dataset_idx += 1
        teacher_motion_offset += teacher_count

    if student_dataset_idx != int(student_dataset.dataset_counts.numel()):
        raise ValueError("Unused paired student shards remain after building teacher/student mapping")
    return teacher_to_student_motion_id


def _motion_data_field_shape(
    field_name: str,
    *,
    env_size: int,
    window_len: int,
    joint_count: int,
    body_count: int,
) -> tuple[int, ...]:
    if field_name.startswith("root_"):
        last_dim = 4 if "_quat_" in field_name else 3
        return (env_size, window_len, last_dim)
    if field_name.startswith("joint_"):
        return (env_size, window_len, joint_count)
    if field_name.startswith("body_"):
        last_dim = 4 if "_quat_" in field_name else 3
        return (env_size, window_len, body_count, last_dim)
    raise ValueError(f"Unsupported MotionData field '{field_name}' for cache allocation")


def _init_motion_data_field_(field_name: str, tensor: torch.Tensor):
    if "_quat_" in field_name:
        tensor[..., 0] = 1.0


class ProgressiveMultiMotionDataset:
    def __init__(
        self,
        *,
        groups: List[dict],
        env_size: int,
        cache_request_range: Sequence[int],
        cache_fill_range: Sequence[int],
        required_motion_body_patterns: List[str],
        fk_asset: Any,
        dataset_extra_keys: List[dict] = [],
        device: torch.device = torch.device("cpu"),
        ds_device: torch.device | None = None,
        motion_fps: float = 50.0,
        offset_pos_z: float = 0.0,
        skip_initial_frames: int = 0,
    ):
        self.device = device
        self.ds_device = device if ds_device is None else ds_device
        self.env_size = env_size
        self.dataset_extra_keys = dataset_extra_keys
        self.required_motion_body_patterns = list(required_motion_body_patterns)
        self.motion_fps = float(motion_fps)
        self.offset_pos_z = float(offset_pos_z)
        self.skip_initial_frames = int(skip_initial_frames)
        if self.skip_initial_frames < 0:
            raise ValueError("dataset.skip_initial_frames must be non-negative")
        self.groups = _normalize_dataset_groups(groups=groups)

        # init and validate cache ranges
        if len(cache_request_range) != 2 or len(cache_fill_range) != 2:
            raise ValueError("cache ranges must be [start, end]")
        self.cache_request_start, self.cache_request_end = map(int, cache_request_range)
        self.cache_fill_start, self.cache_fill_end = map(int, cache_fill_range)
        if not (self.cache_request_start < self.cache_request_end and self.cache_fill_start < self.cache_fill_end):
            raise ValueError("cache ranges must satisfy start < end")
        if not (self.cache_fill_start <= self.cache_request_start <= 0 < self.cache_request_end <= self.cache_fill_end):
            raise ValueError("cache_fill_range must cover cache_request_range and cache_request_range must include 0")
        self.cache_window_len = self.cache_fill_end - self.cache_fill_start
        self.cache_fill_offsets = torch.arange(self.cache_fill_start, self.cache_fill_end, device=self.device, dtype=torch.long)

        # Teacher dataset and related info
        teacher_mem_paths = [group["teacher_mem_path"] for group in self.groups]
        self.teacher_dataset = GlobalMotionDataset.load_from_mem_paths(teacher_mem_paths, dataset_extra_keys=dataset_extra_keys, ds_device=self.ds_device)
        self.teacher_dataset.trim_initial_frames(self.skip_initial_frames)
        self.joint_names = self.teacher_dataset.joint_names
        self.body_names = _select_required_body_names(fk_asset.body_names, self.required_motion_body_patterns)
        self._fk_helper = MotionFKHelper.from_mjlab_asset(asset=fk_asset, dataset_joint_names=self.joint_names, output_body_names=self.body_names)

        # Student dataset is optional since not all teacher datasets need to have paired student datasets
        self.student_dataset: GlobalMotionDataset | None = None
        paired_student_groups = [group for group in self.groups if group["student_mem_path"] is not None]
        paired_student_mem_paths = [group["student_mem_path"] for group in paired_student_groups]
        if paired_student_mem_paths:
            self.student_dataset = GlobalMotionDataset.load_from_mem_paths(paired_student_mem_paths, ds_device=self.ds_device)
            self.student_dataset.trim_initial_frames(self.skip_initial_frames)
        self._teacher_motion_to_student_motion_id = _build_student_pair_mapping(groups=self.groups, teacher_dataset=self.teacher_dataset, student_dataset=self.student_dataset, device=self.device, ds_device=self.ds_device)

        self.global_lengths = self.teacher_dataset.lengths.to(self.device)
        self.motion_to_dataset_id = self.teacher_dataset.motion_to_dataset_id.to(self.device)
        self.motion_source_paths = list(self.teacher_dataset.motion_source_paths)
        self.motion_labels = list(self.teacher_dataset.motion_labels)
        self.motion_sample_probs = _build_motion_sample_probs(groups=self.groups, teacher_dataset=self.teacher_dataset, device=self.device)

        self._teacher_cache = self._allocate_cache_buffer(self.cache_window_len)
        self._student_cache = self._allocate_cache_buffer(self.cache_window_len)
        self._student_cache_has_override = torch.zeros(env_size, dtype=torch.bool, device=self.device)
        self._cache_motion_id = torch.full((env_size,), -1, dtype=torch.long, device=self.device)
        self._cache_start = torch.zeros(env_size, dtype=torch.long, device=self.device)
        self._cache_end = torch.zeros(env_size, dtype=torch.long, device=self.device)
        self._cache_valid = torch.zeros(env_size, dtype=torch.bool, device=self.device)

        self._motion_ids_A = torch.zeros(env_size, dtype=torch.long, device=self.device)
        self._dataset_ids_A = torch.zeros(env_size, dtype=torch.long, device=self.device)
        self._len_A = torch.ones(env_size, dtype=torch.int32, device=self.device)

        self.joint_pos_limit: torch.Tensor | None = None
        self.joint_vel_limit: torch.Tensor | None = None

        self._all_env_ids = torch.arange(self.env_size, device=self.device, dtype=torch.long)
        self._resample_motion_ids(self._all_env_ids)
        self._invalidate_cache(self._all_env_ids)

    def reset(self, env_ids: torch.Tensor) -> torch.Tensor:
        env_ids = env_ids.long()
        # do not resample motion ids or invalidate cache here
        # self._resample_motion_ids(env_ids)
        # self._invalidate_cache(env_ids)
        return self._len_A[env_ids]

    @property
    def env_dataset_ids(self) -> torch.Tensor:
        return self._dataset_ids_A

    def get_teacher_student_slice(
        self,
        env_ids: torch.Tensor | None,
        starts: torch.Tensor,
        *,
        teacher_steps: int | torch.Tensor,
        student_steps: int | torch.Tensor | None = None,
    ) -> MotionData | tuple[MotionData, MotionData]:
        if env_ids is None:
            env_ids = self._all_env_ids
        else:
            env_ids = env_ids.long()
        starts = starts.long()
        self._ensure_cache(env_ids, starts)
        teacher_req_idx = self._build_slice_indices(starts, teacher_steps)
        teacher_cache_idx = teacher_req_idx - self._cache_start[env_ids].unsqueeze(1)
        teacher_motion = self._slice_cache(self._teacher_cache, env_ids, teacher_cache_idx)
        if student_steps is None:
            return teacher_motion

        student_req_idx = self._build_slice_indices(starts, student_steps)
        student_cache_idx = student_req_idx - self._cache_start[env_ids].unsqueeze(1)
        student_motion = self._slice_cache(self._teacher_cache, env_ids, student_cache_idx)
        if not self._student_cache_has_override[env_ids].any():
            return teacher_motion, student_motion

        paired_local_idx = self._student_cache_has_override[env_ids].nonzero(as_tuple=False).squeeze(-1)
        paired_env_ids = env_ids[paired_local_idx]
        paired_student_cache_idx = student_cache_idx[paired_local_idx]
        paired_student_motion = self._slice_cache(self._student_cache, paired_env_ids, paired_student_cache_idx)
        student_motion[paired_local_idx] = paired_student_motion
        return teacher_motion, student_motion

    def set_limit(self, joint_pos_limit: torch.Tensor, joint_vel_limit: torch.Tensor, joint_names: List[str]):
        self.joint_pos_limit = torch.zeros(1, len(self.joint_names), 2, device=self.device)
        self.joint_vel_limit = torch.zeros(1, len(self.joint_names), 2, device=self.device)
        self.joint_pos_limit[:, :, 0] = -3.14
        self.joint_pos_limit[:, :, 1] = 3.14
        self.joint_vel_limit[:, :, 0] = -10.0
        self.joint_vel_limit[:, :, 1] = 10.0
        for asset_idx, name in enumerate(joint_names):
            if name in self.joint_names:
                motion_idx = self.joint_names.index(name)
                self.joint_pos_limit[:, motion_idx] = joint_pos_limit[0, asset_idx]
                self.joint_vel_limit[:, motion_idx] = joint_vel_limit[0, asset_idx]

    def _build_slice_indices(self, starts: torch.Tensor, steps: int | torch.Tensor):
        if isinstance(steps, int):
            req_idx = starts.unsqueeze(1) + torch.arange(steps, device=self.device, dtype=torch.long)
        else:
            step_tensor = steps.long()
            req_idx = starts.unsqueeze(1) + step_tensor.unsqueeze(0)
        return req_idx

    def _ensure_cache(self, env_ids: torch.Tensor, starts: torch.Tensor):
        req_low = starts + self.cache_request_start
        req_high = starts + self.cache_request_end
        current_motion_ids = self._motion_ids_A[env_ids]
        hit_mask = (
            self._cache_valid[env_ids]
            & (self._cache_motion_id[env_ids] == current_motion_ids)
            & (req_low >= self._cache_start[env_ids])
            & (req_high <= self._cache_end[env_ids])
        )
        if hit_mask.all():
            return
        miss_env_ids = env_ids[~hit_mask]
        miss_starts = starts[~hit_mask]
        self._refill_cache(miss_env_ids, miss_starts)

    def _resample_motion_ids(self, env_ids: torch.Tensor):
        num_envs = int(env_ids.numel())
        if num_envs == 0:
            return
        global_ids = torch.multinomial(self.motion_sample_probs, num_envs, replacement=True)
        self._motion_ids_A[env_ids] = global_ids
        self._dataset_ids_A[env_ids] = self.motion_to_dataset_id[global_ids]
        self._len_A[env_ids] = self.global_lengths[global_ids].to(dtype=torch.int32)

    def _invalidate_cache(self, env_ids: torch.Tensor):
        self._cache_valid[env_ids] = False
        self._cache_motion_id[env_ids] = -1
        self._cache_start[env_ids] = 0
        self._cache_end[env_ids] = 0
        self._student_cache_has_override[env_ids] = False

    def _fetch_minimal_global_slice(self, dataset: GlobalMotionDataset, global_motion_ids: torch.Tensor, starts: torch.Tensor, steps: int | torch.Tensor):
        return dataset.get_slice(global_motion_ids, starts, steps=steps, target_device=self.device)

    def _refill_cache(self, env_ids: torch.Tensor, starts: torch.Tensor):
        if env_ids.numel() == 0:
            return
        teacher_motion_ids = self._motion_ids_A[env_ids]
        teacher_minimal = self._fetch_minimal_global_slice(
            self.teacher_dataset,
            teacher_motion_ids,
            starts,
            steps=self.cache_fill_offsets,
        )
        teacher_full = self._post_process(self._fk_helper.expand_minimal_motion(teacher_minimal, fps=self.motion_fps))
        self._teacher_cache[env_ids] = teacher_full

        self._student_cache_has_override[env_ids] = False
        paired_student_motion_ids = self._teacher_motion_to_student_motion_id[teacher_motion_ids]
        paired_mask = paired_student_motion_ids >= 0
        if paired_mask.any():
            if self.student_dataset is None:
                raise RuntimeError("student_dataset must exist when paired student motion ids are present")
            paired_env_ids = env_ids[paired_mask]
            paired_starts = starts[paired_mask]
            student_minimal = self._fetch_minimal_global_slice(
                self.student_dataset,
                paired_student_motion_ids[paired_mask],
                paired_starts,
                steps=self.cache_fill_offsets,
            )
            student_full = self._post_process(self._fk_helper.expand_minimal_motion(student_minimal, fps=self.motion_fps))
            self._student_cache[paired_env_ids] = student_full
            self._student_cache_has_override[paired_env_ids] = True

        self._cache_motion_id[env_ids] = teacher_motion_ids
        self._cache_start[env_ids] = starts + self.cache_fill_start
        self._cache_end[env_ids] = starts + self.cache_fill_end
        self._cache_valid[env_ids] = True

    def _slice_cache(self, cache: MotionData, env_ids: torch.Tensor, cache_idx: torch.Tensor) -> MotionData:
        return cache[env_ids.unsqueeze(-1), cache_idx]

    def _allocate_cache_buffer(self, window_len: int) -> MotionData:
        float_dtype = torch.float32
        body_count = len(self.body_names)
        joint_count = len(self.joint_names)
        data = {}
        for field_name in MotionData.__annotations__:
            tensor = torch.zeros(
                _motion_data_field_shape(field_name, env_size=self.env_size, window_len=window_len, joint_count=joint_count, body_count=body_count),
                dtype=float_dtype,
                device=self.device,
            )
            _init_motion_data_field_(field_name, tensor)
            data[field_name] = tensor
        return MotionData(**data, batch_size=[self.env_size, window_len], device=self.device)

    def _post_process(self, data: MotionData) -> MotionData:
        if self.joint_pos_limit is not None:
            data.joint_pos[:] = torch.clamp(data.joint_pos, self.joint_pos_limit[:, :, 0], self.joint_pos_limit[:, :, 1])
            data.joint_vel[:] = torch.clamp(data.joint_vel, self.joint_vel_limit[:, :, 0], self.joint_vel_limit[:, :, 1])
        data.root_pos_w[..., 2] += self.offset_pos_z
        data.body_pos_w[..., 2] += self.offset_pos_z
        return data
