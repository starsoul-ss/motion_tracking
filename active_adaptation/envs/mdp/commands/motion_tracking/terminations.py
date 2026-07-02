import torch

from active_adaptation.envs.mdp.observations import observation_method
from active_adaptation.envs.mdp.terminations import termination_method
from active_adaptation.utils import symmetry as sym_utils
from active_adaptation.utils.math import quat_apply_inverse

from .utils import _match_indices


class MotionTrackingTerminationMixin:
    TERMINATION_KILL_FRAMES = 5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._body_z_termination_buffer = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self._gravity_dir_termination_buffer = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

    def _continuous_termination(self, trigger: torch.Tensor, buffer: torch.Tensor):
        trigger = trigger.reshape(self.num_envs, -1).any(dim=1)
        buffer.add_(trigger.to(buffer.dtype))
        buffer.masked_fill_(~trigger, 0)
        buffer.clamp_(max=max(int(self.TERMINATION_KILL_FRAMES), 1))
        return (buffer >= int(self.TERMINATION_KILL_FRAMES)).unsqueeze(1)

    def reset(self, env_ids):
        super().reset(env_ids)
        self._body_z_termination_buffer[env_ids] = 0
        self._gravity_dir_termination_buffer[env_ids] = 0

    def _ensure_body_z_termination_indices(self, body_z_terminate_patterns: list[str] | None = None):
        body_z_terminate_patterns = tuple([] if body_z_terminate_patterns is None else body_z_terminate_patterns)
        if not hasattr(self, "_body_z_termination_cache"):
            self._body_z_termination_cache = True
            self._body_z_termination_patterns = body_z_terminate_patterns
            self._body_z_idx_motion, self._body_z_idx_asset = _match_indices(
                self.dataset.body_names,
                self.asset.body_names,
                body_z_terminate_patterns,
                device=self.device,
            )
            self._body_z_names_asset = tuple(self.asset.body_names[int(i)] for i in self._body_z_idx_asset.tolist())
        elif body_z_terminate_patterns != self._body_z_termination_patterns:
            raise ValueError(
                "body_z_termination and body_z_termination_obs must use the same "
                f"body_z_terminate_patterns, got {body_z_terminate_patterns!r} after "
                f"{self._body_z_termination_patterns!r}"
            )
        return self._body_z_idx_motion, self._body_z_idx_asset

    def _body_z_values(self, body_z_terminate_patterns: list[str] | None = None):
        body_z_idx_motion, body_z_idx_asset = self._ensure_body_z_termination_indices(body_z_terminate_patterns)
        target_z = self._motion.body_pos_w[:, 0, body_z_idx_motion, 2]
        current_z = self.asset.data.body_link_pos_w[:, body_z_idx_asset, 2]
        return target_z, current_z

    def _gravity_dir_values(self):
        motion_quat = self._motion.root_quat_w[:, 0]
        motion_g_b = quat_apply_inverse(motion_quat, self._gravity_vec_w)
        current_quat = self.asset.data.root_link_quat_w
        robot_g_b = quat_apply_inverse(current_quat, self._gravity_vec_w)
        return motion_g_b, robot_g_b

    @termination_method
    def motion_xy_range_termination(
        self,
        motion_xy_max_offset: float | None = None,
    ):
        motion_xy_max_offset = float("inf") if motion_xy_max_offset is None else float(motion_xy_max_offset)
        target_xy = self._motion.root_pos_w[:, 0, :2]
        exceed = (target_xy.abs() > motion_xy_max_offset).any(dim=1, keepdim=True)
        return exceed

    @termination_method
    def body_z_termination(
        self,
        body_z_terminate_thres: list[float] | tuple[float, float] | None = None,
        body_z_terminate_patterns: list[str] | None = None,
    ):
        body_z_terminate_min, body_z_terminate_max = (
            float(body_z_terminate_thres[0]),
            float(body_z_terminate_thres[1]),
        )
        target_z, current_z = self._body_z_values(body_z_terminate_patterns)
        target_z_min_thres = target_z + body_z_terminate_min  # [N, B]
        target_z_max_thres = target_z + body_z_terminate_max  # [N, B]
        target_z_min = target_z.amin(dim=1, keepdim=True)
        # Relax lower bound for airborne motions:
        # target_z_min <= 0.1: no relax
        # target_z_min >= 0.3: max relax = 0.2
        # in-between: linear interpolation
        lower_relax = ((target_z_min - 0.1) / 0.2).clamp(0.0, 1.0) * 0.2
        target_z_min_thres = target_z_min_thres - lower_relax

        exceed = ((current_z < target_z_min_thres) | (current_z > target_z_max_thres)).any(dim=1, keepdim=True)
        terminate = self._continuous_termination(exceed, self._body_z_termination_buffer)
        self._reinit_requested.logical_or_(terminate.view(-1))
        return terminate

    @termination_method
    def gravity_dir_termination(
        self,
        gravity_terminate_thres: float | None = None,
    ):
        gravity_terminate_thres = 0.0 if gravity_terminate_thres is None else float(gravity_terminate_thres)
        motion_g_b, robot_g_b = self._gravity_dir_values()
        exceed = torch.linalg.norm(motion_g_b - robot_g_b, dim=-1, keepdim=True) > gravity_terminate_thres
        terminate = self._continuous_termination(exceed, self._gravity_dir_termination_buffer)
        self._reinit_requested.logical_or_(terminate.view(-1))
        return terminate

    @observation_method
    def body_z_termination_obs(self, body_z_terminate_patterns: list[str] | None = None):
        target_z, current_z = self._body_z_values(body_z_terminate_patterns)
        buffer_size = self._body_z_termination_buffer.float().unsqueeze(1)
        return torch.cat([current_z, target_z, buffer_size], dim=-1)

    def body_z_termination_obs_sym(self, body_z_terminate_patterns: list[str] | None = None):
        self._ensure_body_z_termination_indices(body_z_terminate_patterns)
        z_transform = sym_utils.cartesian_space_symmetry(self.asset, self._body_z_names_asset, sign=(1,))
        scalar_transform = sym_utils.SymmetryTransform(torch.arange(1), [1.0])
        return sym_utils.SymmetryTransform.cat([z_transform, z_transform, scalar_transform])

    @observation_method
    def gravity_dir_termination_obs(self):
        motion_g_b, robot_g_b = self._gravity_dir_values()
        buffer_size = self._gravity_dir_termination_buffer.float().unsqueeze(1)
        return torch.cat([robot_g_b, motion_g_b, buffer_size], dim=-1)

    def gravity_dir_termination_obs_sym(self):
        gravity_transform = sym_utils.SymmetryTransform(torch.arange(3), [1.0, -1.0, 1.0])
        scalar_transform = sym_utils.SymmetryTransform(torch.arange(1), [1.0])
        return sym_utils.SymmetryTransform.cat([gravity_transform, gravity_transform, scalar_transform])

    @termination_method
    def motion_timeout(self):
        return (self.t >= self.lengths - 1).unsqueeze(1)
