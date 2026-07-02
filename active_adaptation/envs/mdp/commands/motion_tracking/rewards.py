import torch

from active_adaptation.envs.mdp.rewards import reward_method
from active_adaptation.utils.math import axis_angle_from_quat, quat_apply_inverse, quat_delta

from .utils import _calc_exp_sigma


class MotionTrackingRewardMixin:
    @reward_method
    def root_pos_tracking(self, sigma: list[float] | None = None):
        current_pos = self.asset.data.root_link_pos_w
        target_pos = self.reward_root_pos_w
        diff = target_pos - current_pos
        error = diff.norm(dim=-1, keepdim=True)
        return _calc_exp_sigma(error, sigma)

    @reward_method
    def root_vel_tracking(self, sigma: list[float] | None = None):
        current_linvel_w = self.asset.data.root_link_lin_vel_w
        current_quat = self.asset.data.root_link_quat_w

        current_linvel_b = quat_apply_inverse(current_quat, current_linvel_w)
        ref_linvel_b = quat_apply_inverse(self._motion.root_quat_w[:, 0], self._motion.root_lin_vel_w[:, 0])
        diff = ref_linvel_b - current_linvel_b

        error = diff.norm(dim=-1, keepdim=True)
        return _calc_exp_sigma(error, sigma)

    @reward_method
    def root_rot_tracking(self, sigma: list[float] | None = None):
        current_quat = self.asset.data.root_link_quat_w
        target_quat = self.reward_root_quat_w
        diff = axis_angle_from_quat(quat_delta(current_quat, target_quat))
        error = torch.norm(diff, dim=-1, keepdim=True)
        return _calc_exp_sigma(error, sigma)
    
    @reward_method
    def root_ang_vel_tracking(self, sigma: list[float] | None = None):
        current_angvel_w = self.asset.data.root_link_ang_vel_w
        current_quat = self.asset.data.root_link_quat_w

        current_angvel_b = quat_apply_inverse(current_quat, current_angvel_w)
        ref_angvel_b = quat_apply_inverse(self._motion.root_quat_w[:, 0], self._motion.root_ang_vel_w[:, 0])
        diff = ref_angvel_b - current_angvel_b

        error = diff.norm(dim=-1, keepdim=True)
        return _calc_exp_sigma(error, sigma)

    @reward_method
    def keypoint_pos_tracking(self, sigma: list[float] | None = None):
        target_b = self._motion.body_pos_b[:, 0, self.keypoint_idx_motion]
        error = (target_b - self._current_keypoint_pos_b).norm(dim=-1).mean(dim=-1, keepdim=True)
        return _calc_exp_sigma(error, sigma)

    @reward_method
    def keypoint_vel_tracking(self, sigma: list[float] | None = None):
        target_vel_b = self._motion.body_vel_b[:, 0, self.keypoint_idx_motion]
        error = (target_vel_b - self._current_keypoint_linvel_b).norm(dim=-1).mean(dim=-1, keepdim=True)
        return _calc_exp_sigma(error, sigma)

    @reward_method
    def keypoint_rot_tracking(self, sigma: list[float] | None = None):
        target_quat_b = self._motion.body_quat_b[:, 0, self.keypoint_idx_motion]
        diff = axis_angle_from_quat(quat_delta(self._current_keypoint_quat_b, target_quat_b))
        error = diff.norm(dim=-1).mean(dim=-1, keepdim=True)
        return _calc_exp_sigma(error, sigma)

    @reward_method
    def keypoint_angvel_tracking(self, sigma: list[float] | None = None):
        target_angvel_b = self._motion.body_angvel_b[:, 0, self.keypoint_idx_motion]
        error = (target_angvel_b - self._current_keypoint_angvel_b).norm(dim=-1).mean(dim=-1, keepdim=True)
        return _calc_exp_sigma(error, sigma)

    @reward_method
    def joint_pos_tracking(self, sigma: list[float] | None = None):
        actual = self.asset.data.joint_pos[:, self.joint_idx_asset]
        target = self._motion.joint_pos[:, 0, self.joint_idx_motion]
        error = (target - actual).abs().mean(dim=-1, keepdim=True)
        return _calc_exp_sigma(error, sigma)

    @reward_method
    def joint_vel_tracking(self, sigma: list[float] | None = None):
        actual = self.asset.data.joint_vel[:, self.joint_idx_asset]
        target = self._motion.joint_vel[:, 0, self.joint_idx_motion]
        error = (target - actual).abs().mean(dim=-1, keepdim=True)
        return _calc_exp_sigma(error, sigma)
