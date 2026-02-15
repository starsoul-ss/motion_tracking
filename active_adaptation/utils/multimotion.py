import torch
from typing import Any, List, Sequence
import re

from active_adaptation.utils.math import wrap_to_pi  # noqa: F401
from active_adaptation.utils.motion import MotionDataset, MotionData, MotionOriginalData
from active_adaptation.utils.fk_helper import UpperBodyFKHelper
from active_adaptation.utils.joint_modifier import apply_joint_abc_modification_

class ProgressiveMultiMotionDataset:
    def __init__(self,
                 mem_paths: List[str],
                 path_weights: List[float],
                 env_size: int,
                 max_step_size: int,
                 dataset_extra_keys: List[dict] = [],
                 device: torch.device = torch.device("cpu"),
                 ds_device: torch.device = torch.device("cpu"),
                 fix_ds: int = None,
                 fix_motion_id: int = None):
        self.device = device
        self.ds_device = ds_device
        self.env_size = env_size
        self.max_step_size = max_step_size
        self.dataset_extra_keys = dataset_extra_keys

        self.fix_ds = fix_ds
        self.fix_motion_id = fix_motion_id
        self.enable_modify_joint = False

        self.datasets = [
            MotionDataset.create_from_path_lazy(p, dataset_extra_keys, device=ds_device)
            for p in mem_paths
        ]
        assert len(self.datasets) == len(path_weights)

        body0, joint0 = self.datasets[0].body_names, self.datasets[0].joint_names
        for ds in self.datasets[1:]:
            assert ds.body_names == body0 and ds.joint_names == joint0
        self.body_names = body0
        self.joint_names = joint0

        w = torch.tensor(path_weights, dtype=torch.double)
        self.probs = (w / w.sum()).float().to(device)
        self.counts = [ds.num_motions for ds in self.datasets]

        self._buf_A = self._allocate_empty_buffer()
        self._len_A = torch.zeros(env_size, dtype=torch.int32, device=device)
        self._info_A = self._allocate_info_buffer()
        self._modified_mask_A = torch.zeros((env_size, max_step_size), dtype=torch.bool, device=device)

        self._populate_buffer_full()

        self.joint_pos_limit: torch.Tensor | None = None
        self.joint_vel_limit: torch.Tensor | None = None

    def update(self):
        pass

    def reset(self, env_ids: torch.Tensor) -> torch.Tensor:
        env_ids = env_ids.to(self.device)

        return self._len_A[env_ids]

    def get_slice(self,
                  env_ids: torch.Tensor | None,
                  starts: torch.Tensor,
                  steps: int | torch.Tensor = 1) -> "MotionData":
        if env_ids is not None:
            env_ids = env_ids.to(self.device)
        starts = starts.to(self.device)

        if isinstance(steps, int):
            idx = starts.unsqueeze(1) + torch.arange(steps, device=self.device)
        else:
            idx = starts.unsqueeze(1) + steps.to(device=self.device, dtype=torch.long)

        if env_ids is not None:
            idx = idx.clamp(max=(self._len_A[env_ids] - 1).unsqueeze(1))
            sub = self._buf_A[env_ids.unsqueeze(-1), idx]
        else:
            idx = idx.clamp(max=(self._len_A[:] - 1).unsqueeze(1))
            sub = self._buf_A.gather(1, idx)
        sub = self._to_float(sub, dtype=torch.float32)
        return self._post_process(sub)

    def get_slice_original(
        self,
        env_ids: torch.Tensor | None,
        starts: torch.Tensor,
        steps: int | torch.Tensor = 1,
    ) -> "MotionOriginalData":
        original_joint_pos = getattr(self, "original_joint_pos", None)
        original_joint_vel = getattr(self, "original_joint_vel", None)
        original_body_pos_w = getattr(self, "original_body_pos_w", None)
        if (not self.enable_modify_joint) or original_joint_pos is None or original_joint_vel is None:
            raise RuntimeError("get_slice_original requires enabled joint modification and initialized backups")
        if original_body_pos_w is None:
            raise RuntimeError("original_body_pos_w backup is not initialized")

        if env_ids is not None:
            env_ids = env_ids.to(self.device, dtype=torch.long)
        starts = starts.to(self.device, dtype=torch.long)

        if isinstance(steps, int):
            idx = starts.unsqueeze(1) + torch.arange(steps, device=self.device, dtype=torch.long)
        else:
            idx = starts.unsqueeze(1) + steps.to(device=self.device, dtype=torch.long)

        if env_ids is not None:
            idx = idx.clamp(max=(self._len_A[env_ids] - 1).unsqueeze(1))
            joint_pos = original_joint_pos[env_ids.unsqueeze(-1), idx]
            joint_vel = original_joint_vel[env_ids.unsqueeze(-1), idx]
            body_pos_w = original_body_pos_w[env_ids.unsqueeze(-1), idx]
        else:
            idx = idx.clamp(max=(self._len_A[:] - 1).unsqueeze(1))
            jp_idx = idx.unsqueeze(-1).expand(-1, -1, original_joint_pos.shape[-1])
            jv_idx = idx.unsqueeze(-1).expand(-1, -1, original_joint_vel.shape[-1])
            bp_idx = idx.unsqueeze(-1).unsqueeze(-1).expand(
                -1, -1, original_body_pos_w.shape[-2], original_body_pos_w.shape[-1]
            )
            joint_pos = torch.gather(original_joint_pos, dim=1, index=jp_idx)
            joint_vel = torch.gather(original_joint_vel, dim=1, index=jv_idx)
            body_pos_w = torch.gather(original_body_pos_w, dim=1, index=bp_idx)
        body_pos_w[..., 2] += 0.035

        ret = MotionOriginalData(
            joint_pos=joint_pos.to(dtype=torch.float32),
            joint_vel=joint_vel.to(dtype=torch.float32),
            body_pos_w=body_pos_w.to(dtype=torch.float32),
            batch_size=[joint_pos.shape[0], joint_pos.shape[1]],
            device=self.device,
        )
        return ret

    def get_slice_info(self, env_ids: torch.Tensor):
        env_ids = env_ids.to(self.device)
        ret = {}
        for k in self.dataset_extra_keys:
            ret[k['name']] = self._info_A[k['name']][env_ids]
        return ret
    
    def get_slice_modified_mask(
        self,
        env_ids: torch.Tensor | None,
        starts: torch.Tensor,
        steps: int | torch.Tensor = 1,
    ) -> torch.Tensor:
        if env_ids is not None:
            env_ids = env_ids.to(self.device, dtype=torch.long)
        starts = starts.to(self.device, dtype=torch.long)

        if isinstance(steps, int):
            idx = starts.unsqueeze(1) + torch.arange(steps, device=self.device, dtype=torch.long)
        else:
            idx = starts.unsqueeze(1) + steps.to(device=self.device, dtype=torch.long)

        if env_ids is not None:
            idx = idx.clamp(max=(self._len_A[env_ids] - 1).unsqueeze(1))
            return self._modified_mask_A[env_ids.unsqueeze(-1), idx]
        idx = idx.clamp(max=(self._len_A[:] - 1).unsqueeze(1))
        return self._modified_mask_A.gather(1, idx)

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

    def _allocate_empty_buffer(self) -> "MotionData":
        tpl = self.datasets[0].data
        mm = {}
        for field in tpl.__dataclass_fields__:
            t = getattr(tpl, field)
            if torch.is_floating_point(t):
                mm[field] = torch.zeros(
                    (self.env_size, self.max_step_size) + t.shape[1:],
                    dtype=torch.float16,
                    device=self.device
                )
            else:
                mm[field] = torch.zeros(
                    (self.env_size, self.max_step_size) + t.shape[1:],
                    dtype=t.dtype,
                    device=self.device
                )
        return MotionData(**mm, batch_size=[self.env_size, self.max_step_size], device=self.device)

    def _allocate_info_buffer(self):
        ret = {}
        for k in self.dataset_extra_keys:
            ret[k['name']] = torch.zeros((self.env_size, k['shape']), dtype=k['dtype'], device=self.device)
        return ret

    @torch.no_grad()
    def _populate_buffer_full(self):
        buf, len_buf, info_buf = self._buf_A, self._len_A, self._info_A

        path_samples = torch.multinomial(self.probs, self.env_size, replacement=True).to(torch.int32)

        if self.fix_ds is not None:
            path_samples[:] = self.fix_ds

        for pi, ds in enumerate(self.datasets):
            mask = (path_samples == pi)
            if not mask.any():
                continue
            cnt = self.counts[pi]
            mids = (torch.rand(mask.sum(), device=self.ds_device) * cnt).floor().to(torch.int32)
            
            if self.fix_motion_id is not None:
                mids[:] = self.fix_motion_id

            mids_long = mids.to(torch.long)
            local_starts = ds.starts[mids_long]
            local_ends = ds.ends[mids_long] - 1
            steps = torch.arange(self.max_step_size, device=self.ds_device, dtype=torch.long)
            local_idx = local_starts.unsqueeze(1) + steps  # (k, max_step)
            local_idx = local_idx.clamp(max=local_ends.unsqueeze(1))

            buf[mask, :self.max_step_size] = self._to_float(ds.data[local_idx].to(self.device), dtype=torch.float16)
            len_buf[mask] = ds.lengths[mids_long].clamp_max(self.max_step_size).to(self.device)

            for k in self.dataset_extra_keys:
                name = k['name']
                info_buf[name][mask] = ds.info[name][mids_long].to(self.device)

    def _post_process(self, data: "MotionData") -> "MotionData":
        data = self._clamp_joint_pos_vel(data)
        data = self._offset_pos_z(data)
        return data

    def _offset_pos_z(self, data: "MotionData", z_offset: float = 0.035):
        data.root_pos_w[..., 2] += z_offset
        data.body_pos_w[..., 2] += z_offset
        return data

    def _clamp_joint_pos_vel(self, data: "MotionData"):
        if self.joint_pos_limit is None:
            return data
        joint_pos = wrap_to_pi(data.joint_pos)
        data.joint_pos[:] = torch.clamp(joint_pos,
                                        self.joint_pos_limit[:, :, 0],
                                        self.joint_pos_limit[:, :, 1])
        data.joint_vel[:] = torch.clamp(data.joint_vel,
                                        self.joint_vel_limit[:, :, 0],
                                        self.joint_vel_limit[:, :, 1])
        return data

    @staticmethod
    def _to_float(data, dtype=torch.float32):
        for f in data.__dataclass_fields__:
            v = getattr(data, f)
            if torch.is_floating_point(v):
                setattr(data, f, v.to(dtype=dtype))
        return data

    def setup_joint_modification(
        self,
        *,
        ac_len_range: Sequence[int],
        b_ratio_range: Sequence[float],
        fps: float,
        modify_b_tmid_prob: float,
        modify_b_dataset_prob: float,
        modify_joint_pos_bank: torch.Tensor | None,
        modify_joint_left_patterns: List[str],
        modify_joint_right_patterns: List[str],
        fk_asset: Any,
        fk_base_body_name: str,
        fk_ee_link_names: Sequence[str],
        backup_body_idx_motion: Sequence[int] | torch.Tensor,
    ):
        self.enable_modify_joint = True
        self.modify_joint_left_prob = 0.7
        self.modify_joint_right_prob = 0.7
        self.modify_ac_len_range = tuple(int(x) for x in ac_len_range)
        self.modify_b_ratio_range = tuple(float(x) for x in b_ratio_range)
        self.modify_fps = float(fps)
        self.modify_b_tmid_prob = float(modify_b_tmid_prob)
        self.modify_b_dataset_prob = float(modify_b_dataset_prob)
        self.modify_joint_left_patterns = list(modify_joint_left_patterns)
        self.modify_joint_right_patterns = list(modify_joint_right_patterns)
        self.modify_fk_asset = fk_asset
        self.modify_fk_base_body_name = str(fk_base_body_name)
        self.modify_fk_ee_link_names = tuple(fk_ee_link_names)
        self.modify_backup_body_idx_motion = torch.tensor(
            list(backup_body_idx_motion), device=self.device, dtype=torch.long
        )
        self.modify_joint_left_ids = None
        self.modify_joint_right_ids = None
        self.modify_joint_pos_bank = None
        self.original_joint_pos = None
        self.original_joint_vel = None
        self.original_body_pos_w = None
        self._fk_helper = None
        left_ids = []
        right_ids = []
        for i, name in enumerate(self.joint_names):
            if any(re.match(p, name) for p in self.modify_joint_left_patterns):
                left_ids.append(i)
            if any(re.match(p, name) for p in self.modify_joint_right_patterns):
                right_ids.append(i)
        if len(left_ids) == 0 and len(right_ids) == 0:
            raise ValueError("No joints matched modify_joint_left_patterns/modify_joint_right_patterns")
        self.modify_joint_left_ids = torch.tensor(left_ids, device=self.device, dtype=torch.long)
        self.modify_joint_right_ids = torch.tensor(right_ids, device=self.device, dtype=torch.long)
        if modify_joint_pos_bank is not None:
            self.modify_joint_pos_bank = modify_joint_pos_bank.to(
                device=self.device, dtype=self._buf_A.joint_pos.dtype
            ).contiguous()

        self._fk_helper = UpperBodyFKHelper.from_mjlab_asset(
            asset=self.modify_fk_asset,
            dataset_joint_names=self.joint_names,
            dataset_body_names=self.body_names,
            ee_link_names=self.modify_fk_ee_link_names,
            base_body_name=self.modify_fk_base_body_name,
        )

        self.original_joint_pos = self._buf_A.joint_pos.clone()
        self.original_joint_vel = self._buf_A.joint_vel.clone()
        self.original_body_pos_w = self._buf_A.body_pos_w[:, :, self.modify_backup_body_idx_motion].clone()

    @torch.no_grad()
    def modify_joint(self, env_ids_restore: torch.Tensor, env_ids_modify: torch.Tensor):
        if (not self.enable_modify_joint) or env_ids_restore.numel() == 0:
            return
        original_joint_pos = getattr(self, "original_joint_pos", None)
        original_joint_vel = getattr(self, "original_joint_vel", None)
        modify_joint_left_ids = getattr(self, "modify_joint_left_ids", None)
        modify_joint_right_ids = getattr(self, "modify_joint_right_ids", None)
        modify_joint_pos_bank = getattr(self, "modify_joint_pos_bank", None)
        fk_helper = getattr(self, "_fk_helper", None)
        if original_joint_pos is None or original_joint_vel is None:
            raise RuntimeError("original_joint_pos/vel are not initialized")
        if modify_joint_left_ids is None or modify_joint_right_ids is None:
            raise RuntimeError("modify_joint_left_ids/modify_joint_right_ids are not initialized")
        if fk_helper is None:
            raise RuntimeError("FK helper is not initialized")

        env_ids_restore = env_ids_restore.to(self.device, dtype=torch.long)
        env_ids_modify = env_ids_modify.to(self.device, dtype=torch.long)

        # Restore original joint track first, then apply a new perturbation.
        self._buf_A.joint_pos[env_ids_restore] = original_joint_pos[env_ids_restore]
        self._buf_A.joint_vel[env_ids_restore] = original_joint_vel[env_ids_restore]
        self._modified_mask_A[env_ids_restore] = False

        if env_ids_modify.numel() > 0:
            sub_joint_pos = self._buf_A.joint_pos[env_ids_modify].clone()
            sub_joint_vel = self._buf_A.joint_vel[env_ids_modify].clone()
            sub_lengths = self._len_A[env_ids_modify].to(dtype=torch.long)

            sub_modified_mask = apply_joint_abc_modification_(
                sub_joint_pos,
                sub_joint_vel,
                sub_lengths,
                left_joint_ids=modify_joint_left_ids,
                right_joint_ids=modify_joint_right_ids,
                left_prob=self.modify_joint_left_prob,
                right_prob=self.modify_joint_right_prob,
                b_tmid_prob=self.modify_b_tmid_prob,
                b_dataset_prob=self.modify_b_dataset_prob,
                joint_pos_bank=modify_joint_pos_bank,
                ac_len_range=self.modify_ac_len_range,
                b_ratio_range=self.modify_b_ratio_range,
                fps=self.modify_fps,
            )

            self._buf_A.joint_pos[env_ids_modify] = sub_joint_pos
            self._buf_A.joint_vel[env_ids_modify] = sub_joint_vel
            self._modified_mask_A[env_ids_modify] = sub_modified_mask

        # Sync restore/modify results back to body-level trajectories via FK.
        sub_motion = self._buf_A[env_ids_restore].clone()
        fk_helper.forward(sub_motion)
        self._buf_A.body_pos_w[env_ids_restore] = sub_motion.body_pos_w
        self._buf_A.body_pos_b[env_ids_restore] = sub_motion.body_pos_b
        self._buf_A.body_vel_w[env_ids_restore] = sub_motion.body_vel_w
        self._buf_A.body_vel_b[env_ids_restore] = sub_motion.body_vel_b
        self._buf_A.body_quat_w[env_ids_restore] = sub_motion.body_quat_w
        self._buf_A.body_quat_b[env_ids_restore] = sub_motion.body_quat_b
        self._buf_A.body_angvel_w[env_ids_restore] = sub_motion.body_angvel_w
        self._buf_A.body_angvel_b[env_ids_restore] = sub_motion.body_angvel_b
