import torch
from typing import List

from active_adaptation.utils.math import wrap_to_pi
from active_adaptation.utils.motion import MotionDataset, MotionData


class SimpleSequentialMultiMotionDataset:
    def __init__(
        self,
        mem_paths: List[str],
        path_weights: List[float] | None = None,
        env_size: int = 1,
        max_step_size: int = 1000,
        dataset_extra_keys: List[dict] = [],
        device: torch.device = torch.device("cpu"),
        ds_device: torch.device = torch.device("cpu"),
        sequential_start: int = 0,
        sequential_wrap: bool = True,
        **_ignored,
    ):
        if len(mem_paths) != 1:
            raise ValueError("SimpleSequentialMultiMotionDataset expects exactly one mem_path.")

        self.device = device
        self.ds_device = ds_device
        self.env_size = env_size
        self.max_step_size = max_step_size
        self.dataset_extra_keys = dataset_extra_keys

        self.ds = MotionDataset.create_from_path_lazy(mem_paths[0], dataset_extra_keys, device=ds_device)
        self.body_names = self.ds.body_names
        self.joint_names = self.ds.joint_names

        self._total_motions = self.ds.num_motions
        if self._total_motions <= 0:
            raise ValueError("No motions available in dataset.")

        self._next_motion = int(sequential_start)
        self._sequential_wrap = sequential_wrap

        self.motion_ids = torch.full((env_size,), -1, dtype=torch.int32, device=device)
        self.lengths = torch.zeros(env_size, dtype=torch.int32, device=device)

        self.joint_pos_limit: torch.Tensor | None = None
        self.joint_vel_limit: torch.Tensor | None = None
        self.resample_all()

    def update(self):
        return None

    def reset(self, env_ids: torch.Tensor) -> torch.Tensor:
        env_ids = env_ids.to(self.device)
        if (self.motion_ids[env_ids] < 0).any():
            self.resample(env_ids)
        return self.lengths[env_ids]

    def get_slice(
        self,
        env_ids: torch.Tensor | None,
        starts: torch.Tensor,
        steps: int | torch.Tensor = 1,
    ) -> MotionData:
        if env_ids is None:
            motion_ids = self.motion_ids
        else:
            env_ids = env_ids.to(self.device)
            motion_ids = self.motion_ids[env_ids]

        motion_ids_ds = motion_ids.to(self.ds_device, dtype=torch.long)
        starts_ds = starts.to(self.ds_device, dtype=torch.long)
        data = self.ds.get_slice(motion_ids_ds, starts_ds, steps=steps)
        if self.device != self.ds_device:
            data = data.to(self.device)
        data = self._to_float(data, dtype=torch.float32)
        return self._post_process(data)

    def get_slice_info(self, env_ids: torch.Tensor):
        if not self.dataset_extra_keys:
            return {}
        env_ids = env_ids.to(self.device)
        motion_ids = self.motion_ids[env_ids].to(self.ds_device, dtype=torch.long)
        ret = {}
        for k in self.dataset_extra_keys:
            name = k["name"]
            ret[name] = self.ds.info[name][motion_ids].to(self.device)
        return ret

    def get_current_motion_ids(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            return self.motion_ids
        env_ids = env_ids.to(self.device)
        return self.motion_ids[env_ids]

    def resample(self, env_ids: torch.Tensor):
        env_ids = env_ids.to(self.device)
        num = env_ids.numel()
        motion_ids = self._next_motion_ids(num).to(self.device)
        self.motion_ids[env_ids] = motion_ids

        motion_ids_ds = motion_ids.to(self.ds_device, dtype=torch.long)
        lengths = self.ds.lengths[motion_ids_ds].to(self.device)
        self.lengths[env_ids] = lengths

    def resample_all(self):
        env_ids = torch.arange(self.env_size, device=self.device, dtype=torch.long)
        self.resample(env_ids)

    def set_limit(self, joint_pos_limit: torch.Tensor, joint_vel_limit: torch.Tensor, joint_names: List[str]):
        self.joint_pos_limit = torch.zeros(1, len(self.joint_names), 2, device=self.device)
        self.joint_vel_limit = torch.zeros(1, len(self.joint_names), 2, device=self.device)

        self.joint_pos_limit[:, :, 0] = -3.14
        self.joint_pos_limit[:, :, 1] = 3.14
        self.joint_vel_limit[:, :, 0] = -10.0
        self.joint_vel_limit[:, :, 1] = 10.0

        for id_asset, name in enumerate(joint_names):
            if name in self.joint_names:
                id_motion = self.joint_names.index(name)
                self.joint_pos_limit[:, id_motion] = joint_pos_limit[0, id_asset]
            else:
                print(f"[warning] joint {name} not found in motion dataset")

    def _next_motion_ids(self, num: int) -> torch.Tensor:
        ids = torch.arange(num, device=self.device, dtype=torch.long) + self._next_motion
        if self._sequential_wrap:
            ids = ids % self._total_motions
            self._next_motion = int((self._next_motion + num) % self._total_motions)
        else:
            ids = ids.clamp(max=self._total_motions - 1)
            self._next_motion = int(min(self._next_motion + num, self._total_motions))
        return ids.to(torch.int32)

    def _post_process(self, data: MotionData) -> MotionData:
        data = self._clamp_joint_pos_vel(data)
        data = self._offset_pos_z(data)
        return data

    def _offset_pos_z(self, data: MotionData, z_offset: float = 0.035) -> MotionData:
        data.root_pos_w[..., 2] += z_offset
        data.body_pos_w[..., 2] += z_offset
        return data

    def _clamp_joint_pos_vel(self, data: MotionData) -> MotionData:
        if self.joint_pos_limit is None:
            return data
        joint_pos = wrap_to_pi(data.joint_pos)
        data.joint_pos[:] = torch.clamp(
            joint_pos,
            self.joint_pos_limit[:, :, 0],
            self.joint_pos_limit[:, :, 1],
        )
        data.joint_vel[:] = torch.clamp(
            data.joint_vel,
            self.joint_vel_limit[:, :, 0],
            self.joint_vel_limit[:, :, 1],
        )
        return data

    @staticmethod
    def _to_float(data: MotionData, dtype=torch.float32):
        for f in data.__dataclass_fields__:
            v = getattr(data, f)
            if torch.is_floating_point(v):
                setattr(data, f, v.to(dtype=dtype))
        return data
