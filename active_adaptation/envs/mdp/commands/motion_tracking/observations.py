import torch

from active_adaptation.envs.mdp.observations import observation_method
from active_adaptation.envs.mdp.utils import perturb_quaternion
from active_adaptation.utils import symmetry as sym_utils
from active_adaptation.utils.math import quat_apply_inverse, quat_delta, quat_in_frame, quat_to_rot6d

from .utils import get_items_by_index


class MotionTrackingObservationMixin:
    ### student only observations ###
    @observation_method
    def command_obs(self, noise_std: float = 0.0):
        motion = self._motion_student

        root_quat = self.asset.data.root_link_quat_w
        if noise_std > 0.0:
            root_quat = perturb_quaternion(root_quat, noise_std)
        root_quat = root_quat.unsqueeze(1)

        root_quat_future = motion.root_quat_w[:, 0:, :]
        root_quat_future0 = root_quat_future[:, 0, :].unsqueeze(1)

        root_pos_future = motion.root_pos_w[:, 1:, :]
        root_pos_future0 = motion.root_pos_w[:, 0, :].unsqueeze(1)

        # pos diff is applied in expected root frame
        pos_diff_b = quat_apply_inverse(root_quat_future0, root_pos_future - root_pos_future0)
        
        # quat diff is applied in current root frame
        # because we can get reliable quat from real robot IMU
        root_quat = root_quat.expand(-1, root_quat_future.shape[1], -1)
        quat_diff = quat_in_frame(root_quat, root_quat_future)
        rot6d_diff = quat_to_rot6d(quat_diff)

        return torch.cat([
            pos_diff_b.reshape(self.num_envs, -1),
            rot6d_diff.reshape(self.num_envs, -1),
        ], dim=-1)
    def command_obs_sym(self):
        steps = self.student_future_steps
        return sym_utils.SymmetryTransform.cat([
            sym_utils.SymmetryTransform(perm=torch.arange(3), signs=[1, -1, 1]).repeat(len(steps) - 1),
            sym_utils.SymmetryTransform(
                perm=torch.arange(6),
                signs=[1, -1, 1, -1, 1, -1]
            ).repeat(len(steps)),
        ])

    @observation_method
    def boot_indicator_state_obs(self):
        return self.boot_indicator / self.boot_indicator_max
    def boot_indicator_state_obs_sym(self):
        return sym_utils.SymmetryTransform(perm=torch.arange(1), signs=[1.])

    ### teacher only observations ###
    @observation_method
    def target_pos_b_obs(self):
        current_pos = self.asset.data.root_link_pos_w.unsqueeze(1) - self.env.scene.env_origins.unsqueeze(1)
        current_quat = self.asset.data.root_link_quat_w.unsqueeze(1)
        target_pos_b = quat_apply_inverse(current_quat, (self._motion.root_pos_w - current_pos))
        return target_pos_b.reshape(self.num_envs, -1)
    def target_pos_b_obs_sym(self):
        return sym_utils.SymmetryTransform(perm=torch.arange(3), signs=[1., -1., 1.] ).repeat(len(self.future_steps))

    @observation_method
    def target_rot_b_obs(self):
        relative_quat = quat_in_frame(self.asset.data.root_link_quat_w.unsqueeze(1), self._motion.root_quat_w)
        rot6d = quat_to_rot6d(relative_quat)
        return rot6d.reshape(self.num_envs, -1)
    def target_rot_b_obs_sym(self):
        return sym_utils.SymmetryTransform(perm=torch.arange(6), signs=[1, -1, 1, -1, 1, -1] ).repeat(len(self.future_steps))
    
    @observation_method
    def target_linvel_b_obs(self):
        target_linvel_b = quat_apply_inverse(self.asset.data.root_link_quat_w.unsqueeze(1), self._motion.root_lin_vel_w)
        return target_linvel_b.reshape(self.num_envs, -1)
    def target_linvel_b_obs_sym(self):
        return sym_utils.SymmetryTransform(perm=torch.arange(3), signs=[1., -1., 1.] ).repeat(len(self.future_steps))

    @observation_method
    def target_angvel_b_obs(self):
        target_angvel_b = quat_apply_inverse(self.asset.data.root_link_quat_w.unsqueeze(1), self._motion.root_ang_vel_w)
        return target_angvel_b.reshape(self.num_envs, -1)
    def target_angvel_b_obs_sym(self):
        return sym_utils.SymmetryTransform(perm=torch.arange(3), signs=[-1., 1., -1.] ).repeat(len(self.future_steps))

    @observation_method
    def target_keypoints_pos_b_obs(self, required_steps: int | None = None, include_diff: bool = False):
        if required_steps is None:
            required_steps = self._motion.body_pos_b.shape[1]
        target_keypoints_b = self._motion.body_pos_b[:, :required_steps, self.keypoint_idx_motion]
        if not include_diff:
            return target_keypoints_b.reshape(self.num_envs, -1)

        actual_b = self._current_keypoint_pos_b
        target_w = (self._motion.body_pos_w[:, :required_steps, self.keypoint_idx_motion] - self._motion.root_pos_w[:, 0:1, :].unsqueeze(2))
        target_b = quat_apply_inverse(self._motion.root_quat_w[:, 0:1, :].unsqueeze(2), target_w)
        diff_b = target_b - actual_b.unsqueeze(1)
        return torch.cat([target_keypoints_b.reshape(self.num_envs, -1), diff_b.reshape(self.num_envs, -1)], dim=-1)
    def target_keypoints_pos_b_obs_sym(self, required_steps: int | None = None, include_diff: bool = False):
        if required_steps is None:
            required_steps = len(self.future_steps)
        transform = sym_utils.cartesian_space_symmetry(
            self.asset,
            get_items_by_index(self.asset.body_names, self.keypoint_idx_asset),
            sign=[1, -1, 1],
        ).repeat(required_steps)
        if not include_diff:
            return transform
        return sym_utils.SymmetryTransform.cat([transform, transform])

    @observation_method
    def target_keypoints_rot_b_obs(self, required_steps: int | None = None, include_diff: bool = False):
        if required_steps is None:
            required_steps = self._motion.body_quat_b.shape[1]
        target_quat_b = self._motion.body_quat_b[:, :required_steps, self.keypoint_idx_motion]
        target_rot6d_b = quat_to_rot6d(target_quat_b)
        if not include_diff:
            return target_rot6d_b.reshape(self.num_envs, -1)

        actual_quat_b = self._current_keypoint_quat_b
        target_quat_ref = quat_in_frame(self._motion.root_quat_w[:, 0:1, :].unsqueeze(2), self._motion.body_quat_w[:, :required_steps, self.keypoint_idx_motion])
        diff_quat_b = quat_delta(actual_quat_b.unsqueeze(1), target_quat_ref)
        diff_rot6d_b = quat_to_rot6d(diff_quat_b)
        return torch.cat([target_rot6d_b.reshape(self.num_envs, -1), diff_rot6d_b.reshape(self.num_envs, -1)], dim=-1)
    def target_keypoints_rot_b_obs_sym(self, required_steps: int | None = None, include_diff: bool = False):
        if required_steps is None:
            required_steps = len(self.future_steps)
        transform = sym_utils.cartesian_space_symmetry(
            self.asset,
            get_items_by_index(self.asset.body_names, self.keypoint_idx_asset),
            sign=[1, -1, 1, -1, 1, -1],
        ).repeat(required_steps)
        if not include_diff:
            return transform
        return sym_utils.SymmetryTransform.cat([transform, transform])

    @observation_method
    def target_keypoints_linvel_b_obs(self, required_steps: int | None = None, include_diff: bool = False):
        if required_steps is None:
            required_steps = self._motion.body_vel_b.shape[1]
        target_vel_b = self._motion.body_vel_b[:, :required_steps, self.keypoint_idx_motion]
        if not include_diff:
            return target_vel_b.reshape(self.num_envs, -1)

        actual_vel_b = self._current_keypoint_linvel_b
        diff_vel_b = target_vel_b - actual_vel_b.unsqueeze(1)
        return torch.cat(
            [
                target_vel_b.reshape(self.num_envs, -1),
                diff_vel_b.reshape(self.num_envs, -1),
            ],
            dim=-1,
        )
    def target_keypoints_linvel_b_obs_sym(self, required_steps: int | None = None, include_diff: bool = False):
        if required_steps is None:
            required_steps = len(self.future_steps)
        transform = sym_utils.cartesian_space_symmetry(
            self.asset,
            get_items_by_index(self.asset.body_names, self.keypoint_idx_asset),
            sign=[1, -1, 1],
        ).repeat(required_steps)
        if not include_diff:
            return transform
        return sym_utils.SymmetryTransform.cat([transform, transform])

    @observation_method
    def target_keypoints_angvel_b_obs(self, required_steps: int | None = None, include_diff: bool = False):
        if required_steps is None:
            required_steps = self._motion.body_angvel_b.shape[1]
        target_angvel_b = self._motion.body_angvel_b[:, :required_steps, self.keypoint_idx_motion]
        if not include_diff:
            return target_angvel_b.reshape(self.num_envs, -1)

        actual_angvel_b = self._current_keypoint_angvel_b
        diff_angvel_b = target_angvel_b - actual_angvel_b.unsqueeze(1)
        return torch.cat(
            [
                target_angvel_b.reshape(self.num_envs, -1),
                diff_angvel_b.reshape(self.num_envs, -1),
            ],
            dim=-1,
        )
    def target_keypoints_angvel_b_obs_sym(self, required_steps: int | None = None, include_diff: bool = False):
        if required_steps is None:
            required_steps = len(self.future_steps)
        transform = sym_utils.cartesian_space_symmetry(
            self.asset,
            get_items_by_index(self.asset.body_names, self.keypoint_idx_asset),
            sign=[-1, 1, -1],
        ).repeat(required_steps)
        if not include_diff:
            return transform
        return sym_utils.SymmetryTransform.cat([transform, transform])


    @observation_method
    def current_keypoint_pos_b_obs(self):
        return self._current_keypoint_pos_b.reshape(self.num_envs, -1)
    def current_keypoint_pos_b_obs_sym(self):
        return sym_utils.cartesian_space_symmetry(self.asset, get_items_by_index(self.asset.body_names, self.keypoint_idx_asset), sign=[1, -1, 1])

    @observation_method
    def current_keypoint_rot_b_obs(self):
        return self._current_keypoint_rot6d_b.reshape(self.num_envs, -1)
    def current_keypoint_rot_b_obs_sym(self):
        return sym_utils.cartesian_space_symmetry(
            self.asset,
            get_items_by_index(self.asset.body_names, self.keypoint_idx_asset),
            sign=[1, -1, 1, -1, 1, -1],
        )

    @observation_method
    def current_keypoint_linvel_b_obs(self):
        return self._current_keypoint_linvel_b.reshape(self.num_envs, -1)
    def current_keypoint_linvel_b_obs_sym(self):
        return sym_utils.cartesian_space_symmetry(self.asset, get_items_by_index(self.asset.body_names, self.keypoint_idx_asset), sign=[1, -1, 1])

    @observation_method
    def current_keypoint_angvel_b_obs(self):
        return self._current_keypoint_angvel_b.reshape(self.num_envs, -1)
    def current_keypoint_angvel_b_obs_sym(self):
        return sym_utils.cartesian_space_symmetry(
            self.asset,
            get_items_by_index(self.asset.body_names, self.keypoint_idx_asset),
            sign=[-1, 1, -1],
        )

    @observation_method
    def target_feet_contact_state_obs(self):
        return self.feet_standing.float()
    def target_feet_contact_state_obs_sym(self):
        return sym_utils.cartesian_space_symmetry(
            self.asset,
            get_items_by_index(self.asset.body_names, self.feet_idx_asset),
            sign=(1,),
        )

    ### shared observations (teacher + student) ###
    @observation_method
    def target_root_z_obs(self, horizon: str):
        motion = self._motion_for_horizon(horizon)
        return motion.root_pos_w[:, :, 2].reshape(self.num_envs, -1)
    def target_root_z_obs_sym(self, horizon: str):
        return sym_utils.SymmetryTransform(perm=torch.arange(1), signs=[1]).repeat(len(self._steps_for_horizon(horizon)))

    @observation_method
    def target_projected_gravity_b_obs(self, horizon: str):
        motion = self._motion_for_horizon(horizon)
        gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=torch.float32).reshape(1, 1, 3)
        g_b = quat_apply_inverse(motion.root_quat_w, gravity)  # [N, S, 3]
        return g_b.reshape(self.num_envs, -1)
    def target_projected_gravity_b_obs_sym(self, horizon: str):
        steps = self._steps_for_horizon(horizon)
        return sym_utils.SymmetryTransform(perm=torch.arange(3), signs=[1., -1., 1.] ).repeat(len(steps))

    @observation_method
    def target_joint_pos_obs(self, horizon: str, noise_std: float = 0.0):
        target_joint_pos = self._motion_for_horizon(horizon).joint_pos[:, :, self.joint_idx_motion]
        if float(noise_std) > 0.0:
            current_joint_pos = self.get_shared_noisy_joint_pos(self.joint_idx_asset)
        else:
            current_joint_pos = self.asset.data.joint_pos[:, self.joint_idx_asset]
        current_joint_pos = current_joint_pos - self.env.action_manager.offset[:, self.joint_idx_asset]
        current_joint_pos = current_joint_pos.unsqueeze(1) # N, 1, J
        target_minus_current = target_joint_pos - current_joint_pos # N, T, J
        return torch.cat(
            [
                target_joint_pos.reshape(self.num_envs, -1),
                target_minus_current.reshape(self.num_envs, -1),
            ],
            dim=-1,
        )
    def target_joint_pos_obs_sym(self, horizon: str):
        steps = self._steps_for_horizon(horizon)
        transform = sym_utils.joint_space_symmetry(self.asset, self.joint_names).repeat(len(steps))
        return sym_utils.SymmetryTransform.cat([transform, transform])
