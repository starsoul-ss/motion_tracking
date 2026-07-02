import torch
from typing import Sequence

from active_adaptation.envs.mdp.utils import perturb_quaternion
from active_adaptation.utils import joint_order as joint_order_utils
from active_adaptation.utils.math import quat_apply, quat_apply_inverse, quat_delta, quat_in_frame, quat_to_rot6d
from active_adaptation.utils.multimotion import ProgressiveMultiMotionDataset

from ..base import Command
from .observations import MotionTrackingObservationMixin
from .randomizations import MotionTrackingRandomizationMixin
from .rewards import MotionTrackingRewardMixin
from .terminations import MotionTrackingTerminationMixin
from .utils import _match_indices, _resolve_joint_indices, convert_dtype


class MotionTrackingCommand(
    MotionTrackingRandomizationMixin,
    MotionTrackingTerminationMixin,
    MotionTrackingObservationMixin,
    MotionTrackingRewardMixin,
    Command,
):
    def __init__(self, env, dataset: dict,
                dataset_extra_keys: list[dict] = [],
                required_motion_body_patterns: list[str] = [],
                keypoint_patterns: list[str] = [],
                feet_patterns: list[str] = [],
                feet_standing: dict[str, float] = {},
                student_motion_randomization: dict = {},
                init_noise: dict[str, float] = {},
                future_steps: list[int] = [],
                student_future_steps: list[int] | None = None,
                boot_indicator_max: int = 0,
                reinit_prob: float = 0.0,
                reinit_min_steps: int = 0,
                reinit_max_steps: int = 0,
                shared_joint_pos: dict | None = None,
                shared_joint_vel: dict | None = None,):
        super().__init__(env)
        
        # future steps
        self.future_steps = torch.tensor(future_steps, device=self.device, dtype=torch.long)
        if student_future_steps is None:
            student_future_steps = future_steps
        self.student_future_steps = torch.tensor(
            student_future_steps, device=self.device, dtype=torch.long
        )
        if self.future_steps.numel() == 0:
            raise ValueError("future_steps must contain at least one element, and should include 0.")
        if self.student_future_steps.numel() == 0:
            raise ValueError("student_future_steps must contain at least one element, and should include 0.")
        if self.future_steps[0].item() != 0:
            raise ValueError(f"future_steps[0] must be 0, got {self.future_steps.tolist()}")
        if self.student_future_steps[0].item() != 0:
            raise ValueError(f"student_future_steps[0] must be 0, got {self.student_future_steps.tolist()}")
        # Used for init-time upper bound on t. Negative steps do not constrain the future horizon.
        self.max_future_step = torch.clamp_min(torch.cat([self.future_steps, self.student_future_steps]), 0).max()

        # setup dataset
        dataset_extra_keys = [{**k, 'dtype': convert_dtype(k['dtype'])} for k in dataset_extra_keys]
        self.dataset = ProgressiveMultiMotionDataset(
            **dataset,
            env_size=self.num_envs,
            required_motion_body_patterns=required_motion_body_patterns,
            fk_asset=self.asset,
            dataset_extra_keys=dataset_extra_keys,
            device=self.device,
            ds_device=(torch.device('cpu') if self.num_envs < 1024 else self.device),
        )
        joint_pos_limits = self.asset.data.joint_pos_limits
        joint_vel_limits = getattr(self.asset.data, "soft_joint_vel_limits", None)
        if joint_vel_limits is None:
            joint_vel_limits = torch.zeros_like(joint_pos_limits)
            joint_vel_limits[..., 0] = -10.0
            joint_vel_limits[..., 1] = 10.0
        self.dataset.set_limit(joint_pos_limits, joint_vel_limits, self.asset.joint_names)

        # bodies for full‑body keypoint tracking
        self.keypoint_idx_motion, self.keypoint_idx_asset = _match_indices(self.dataset.body_names, self.asset.body_names, keypoint_patterns, device=self.device)

        # feet bodies for standing detection
        self.feet_idx_motion, self.feet_idx_asset = _match_indices( self.dataset.body_names, self.asset.body_names, feet_patterns, device=self.device )
        self.feet_names_asset = tuple(self.asset.body_names[int(i)] for i in self.feet_idx_asset.tolist())
        self.feet_standing_cfg = feet_standing
        self.feet_standing = torch.zeros(self.num_envs, int(self.feet_idx_motion.numel()), dtype=torch.bool, device=self.device)

        # joint indices: follow asset-configured canonical order.
        self.joint_names, self.joint_idx_motion, self.joint_idx_asset = _resolve_joint_indices(
            self.dataset.joint_names,
            self.asset.joint_names,
            joint_order_utils.get_joint_name_order(self.asset),
            device=self.device,
            context="asset canonical order",
        )

        # bookkeeping for termination
        self._gravity_vec_w = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=torch.float32).unsqueeze(0)

        # bookkeeping for sliding root reward
        self.reward_root_history_len = max(int(round(1.0 / float(self.env.step_dt))), 1)
        self.reward_root_ref_xy_history_w = torch.zeros(
            self.num_envs, self.reward_root_history_len, 2, device=self.device, dtype=torch.float32
        )
        self.reward_root_actual_xy_history_w = torch.zeros(
            self.num_envs, self.reward_root_history_len, 2, device=self.device, dtype=torch.float32
        )
        self._reward_root_history_slot = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.reward_root_pos_w = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32)
        self.reward_root_quat_w = torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.float32)
        self.reward_root_quat_w[:, 0] = 1.0
        # Episode-level translations that align teacher/student motions to env origin.
        # Paired teacher/student motions may not share the same XY root at the same timestep.
        self.teacher_motion_origin_offset_w = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32)
        self.student_motion_origin_offset_w = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32)

        # bookkeeping for current keypoint cache (valid after before_update)
        self._current_keypoint_pos_b = torch.zeros(self.num_envs, int(self.keypoint_idx_asset.numel()), 3, device=self.device, dtype=torch.float32)
        self._current_keypoint_quat_b = torch.zeros(self.num_envs, int(self.keypoint_idx_asset.numel()), 4, device=self.device, dtype=torch.float32)
        self._current_keypoint_rot6d_b = torch.zeros(self.num_envs, int(self.keypoint_idx_asset.numel()), 2, 3, device=self.device, dtype=torch.float32)
        self._current_keypoint_linvel_b = torch.zeros(self.num_envs, int(self.keypoint_idx_asset.numel()), 3, device=self.device, dtype=torch.float32)
        self._current_keypoint_angvel_b = torch.zeros(self.num_envs, int(self.keypoint_idx_asset.numel()), 3, device=self.device, dtype=torch.float32)

        # reinit params
        self.reinit_prob = reinit_prob
        self.reinit_min_steps = reinit_min_steps
        self.reinit_max_steps = reinit_max_steps
        self._reinit_requested = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # life cycle state
        self.lengths = torch.full((self.num_envs,), 1, dtype=torch.int32, device=self.device)
        self.t = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.boot_indicator = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device)
        self.boot_indicator_max = boot_indicator_max
        self.joint_pos_boot_protect = self.asset.data.default_joint_pos.clone()

        ## init noise
        self.init_noise_params = init_noise
        self.student_motion_randomization_cfg = student_motion_randomization
        self.shared_joint_pos_cfg = shared_joint_pos
        self.shared_joint_vel_cfg = shared_joint_vel
        self._init_randomization_state()

    ### helper methods for handling different horizons (teacher vs student) ###
    def _is_student_horizon(self, horizon: str) -> bool:
        horizon_key = str(horizon).lower()
        if horizon_key == "teacher":
            return False
        if horizon_key == "student":
            return True
        raise ValueError(f"Invalid horizon '{horizon}', expected 'teacher' or 'student'.")

    def _steps_for_horizon(self, horizon: str) -> torch.Tensor:
        return self.student_future_steps if self._is_student_horizon(horizon) else self.future_steps

    def _motion_for_horizon(self, horizon: str):
        return self._motion_student if self._is_student_horizon(horizon) else self._motion

    def _set_motion_origin_offset(self, env_ids: torch.Tensor, teacher_motion, student_motion):
        self.teacher_motion_origin_offset_w[env_ids] = 0.0
        self.student_motion_origin_offset_w[env_ids] = 0.0
        # Keep motion height unchanged; only re-center XY onto the local env origin.
        self.teacher_motion_origin_offset_w[env_ids, :2] = -teacher_motion.root_pos_w[:, 0, :2]
        self.student_motion_origin_offset_w[env_ids, :2] = -student_motion.root_pos_w[:, 0, :2]

    def _apply_motion_origin_offset(self, teacher_motion, student_motion, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            teacher_offset = self.teacher_motion_origin_offset_w
            student_offset = self.student_motion_origin_offset_w
        else:
            teacher_offset = self.teacher_motion_origin_offset_w[env_ids]
            student_offset = self.student_motion_origin_offset_w[env_ids]

        teacher_motion = teacher_motion.clone()
        teacher_motion.root_pos_w = teacher_motion.root_pos_w + teacher_offset.unsqueeze(1)
        teacher_motion.body_pos_w = teacher_motion.body_pos_w + teacher_offset.unsqueeze(1).unsqueeze(1)

        student_motion = student_motion.clone()
        student_motion.root_pos_w = student_motion.root_pos_w + student_offset.unsqueeze(1)
        student_motion.body_pos_w = student_motion.body_pos_w + student_offset.unsqueeze(1).unsqueeze(1)
        return teacher_motion, student_motion

    ### sample init -> sample init robot -> reset
    def sample_init(self, env_ids: torch.Tensor):
        t = self.t[env_ids]
        # resample motion
        lengths = self.dataset.reset(env_ids)

        n = env_ids.shape[0]

        # --- 1) reinit: rewind from previous t when requested ---
        use_reinit = self._reinit_requested[env_ids] & (torch.rand(n, device=self.device) < self.reinit_prob)
        rewind = torch.randint(self.reinit_min_steps, self.reinit_max_steps + 1, (n,), device=self.device, dtype=self.t.dtype)
        reinit_t = (t - rewind).clamp(min=0)
        self._reinit_requested[env_ids] = False

        # --- 2) random sample: uniform in [0, sample_interval) ---
        sample_interval = (lengths - self.max_future_step.to(lengths.dtype) - 100).clamp_min(0)
        t_rand = (torch.rand(n, device=self.device) * sample_interval.to(torch.float32)).floor().to(self.t.dtype)

        # --- merge: reinit takes priority over random ---
        t[:] = torch.where(use_reinit, reinit_t, t_rand)

        self.lengths[env_ids] = lengths
        self.t[env_ids] = t

        teacher_motion, student_motion = self.dataset.get_teacher_student_slice(
            env_ids,
            self.t[env_ids],
            teacher_steps=1,
            student_steps=1,
        )
        self._set_motion_origin_offset(env_ids, teacher_motion, student_motion)
        motion, _ = self._apply_motion_origin_offset(teacher_motion, student_motion, env_ids)

        # set robot state
        self.sample_init_robot(env_ids, motion)
        return None

    def sample_init_robot(self, env_ids: Sequence[int], motion, lift_height: float = 0.04):
        # Get subsets for the current envs
        init_root_state = self.init_root_state[env_ids].clone()
        init_joint_pos = self.init_joint_pos[env_ids].clone()
        init_joint_vel = self.init_joint_vel[env_ids].clone()
        env_origins = self.env.scene.env_origins[env_ids]

        # Extract motion data
        motion_root_pos = motion.root_pos_w[:, 0]
        motion_root_quat = motion.root_quat_w[:, 0]
        motion_root_lin_vel = motion.root_lin_vel_w[:, 0]
        motion_root_ang_vel = motion.root_ang_vel_w[:, 0]
        motion_body_pos = motion.body_pos_w[:, 0]
        motion_joint_pos = motion.joint_pos[:, 0]
        motion_joint_vel = motion.joint_vel[:, 0]

        # -------- root state ----------------------------------------------------
        init_root_state[:, :3] = env_origins + motion_root_pos
        init_root_state[:, 2] += lift_height
        root_pos_noise = torch.randn_like(init_root_state[:, :3]).clamp(-1, 1) * self.init_noise_params["root_pos"]
        root_pos_noise[:, 2].clamp_min_(0.0)
        init_root_state[:, :3] += root_pos_noise
        body_min_z_after_lift = (
            env_origins[:, 2]
            + motion_body_pos[:, :, 2].amin(dim=1)
            + lift_height
            + root_pos_noise[:, 2]
        )
        init_root_state[:, 2] += (-body_min_z_after_lift).clamp_min(0.0)

        init_root_state[:, 3:7] = motion_root_quat
        init_root_state[:, 3:7] = perturb_quaternion(init_root_state[:, 3:7], self.init_noise_params["root_ori"])

        init_root_state[:, 7:10] = motion_root_lin_vel
        lin_vel_noise = torch.randn_like(init_root_state[:, 7:10]).clamp(-1, 1) * self.init_noise_params["root_lin_vel"]
        init_root_state[:, 7:10] += lin_vel_noise
        
        init_root_state[:, 10:13] = motion_root_ang_vel
        ang_vel_noise = torch.randn_like(init_root_state[:, 10:13]).clamp(-1, 1) * self.init_noise_params["root_ang_vel"]
        init_root_state[:, 10:13] += ang_vel_noise

        # -------- joint state ----------------------------------------------------
        init_joint_pos[:, self.joint_idx_asset] = motion_joint_pos[:, self.joint_idx_motion]
        self.joint_pos_boot_protect[env_ids] = init_joint_pos

        init_joint_vel[:, self.joint_idx_asset] = motion_joint_vel[:, self.joint_idx_motion]
        joint_pos_noise = torch.randn_like(init_joint_pos).clamp(-1, 1) * self.init_noise_params["joint_pos"]
        joint_vel_noise = torch.randn_like(init_joint_vel).clamp(-1, 1) * self.init_noise_params["joint_vel"]
        init_joint_pos += joint_pos_noise
        init_joint_vel += joint_vel_noise

        ref_root_xy_w = env_origins[:, :2] + motion_root_pos[:, :2]
        self.reward_root_ref_xy_history_w[env_ids] = ref_root_xy_w.unsqueeze(1)
        self.reward_root_actual_xy_history_w[env_ids] = init_root_state[:, :2].unsqueeze(1)
        self._reward_root_history_slot[env_ids] = 0

        # Apply the calculated states to the simulation
        self.asset.write_root_state_to_sim(init_root_state, env_ids=env_ids)
        self.asset.write_joint_position_to_sim(init_joint_pos, env_ids=env_ids)
        self.asset.write_joint_velocity_to_sim(init_joint_vel, env_ids=env_ids)
        self.asset.set_joint_position_target(init_joint_pos, env_ids=env_ids)
    
    def reset(self, env_ids):
        super().reset(env_ids)
        self.boot_indicator[env_ids] = self.boot_indicator_max
        self.feet_standing[env_ids] = False

    def update_current_keypoint_cache(self):
        root_pos_w = self.asset.data.root_link_pos_w
        root_quat_w = self.asset.data.root_link_quat_w
        root_lin_vel_w = self.asset.data.root_link_lin_vel_w
        root_ang_vel_w = self.asset.data.root_link_ang_vel_w

        body_pos_w = self.asset.data.body_link_pos_w[:, self.keypoint_idx_asset]
        body_quat_w = self.asset.data.body_link_quat_w[:, self.keypoint_idx_asset]
        body_lin_vel_w = self.asset.data.body_link_lin_vel_w[:, self.keypoint_idx_asset]
        body_ang_vel_w = self.asset.data.body_link_ang_vel_w[:, self.keypoint_idx_asset]

        current_quat_b = quat_in_frame(root_quat_w.unsqueeze(1), body_quat_w)
        self._current_keypoint_pos_b[:] = quat_apply_inverse(root_quat_w.unsqueeze(1), body_pos_w - root_pos_w.unsqueeze(1))
        self._current_keypoint_quat_b[:] = current_quat_b
        self._current_keypoint_rot6d_b[:] = quat_to_rot6d(current_quat_b)
        self._current_keypoint_linvel_b[:] = quat_apply_inverse(root_quat_w.unsqueeze(1), body_lin_vel_w - root_lin_vel_w.unsqueeze(1))
        self._current_keypoint_angvel_b[:] = quat_apply_inverse(root_quat_w.unsqueeze(1), body_ang_vel_w - root_ang_vel_w.unsqueeze(1))

    def update_feet_standing(self):
        cfg = self.feet_standing_cfg
        feet_vel_w = self._motion.body_vel_w[:, 0, self.feet_idx_motion]
        feet_pos_w = self._motion.body_pos_w[:, 0, self.feet_idx_motion]
        root_vxy = self._motion.root_lin_vel_w[:, 0, :2].norm(dim=-1, keepdim=True).clamp_min(1.0)

        feet_vxy = feet_vel_w[..., :2].norm(dim=-1)
        feet_vz_abs = feet_vel_w[..., 2].abs()
        feet_z = feet_pos_w[..., 2]

        enter_contact = (
            (feet_z < float(cfg["z_enter"]))
            & (feet_vxy < float(cfg["vxy_enter"]) * root_vxy)
            & (feet_vz_abs < float(cfg["vz_enter"]) * root_vxy)
        )
        exit_contact = (
            (feet_z > float(cfg["z_exit"]))
            | (feet_vxy > float(cfg["vxy_exit"]) * root_vxy)
            | (feet_vz_abs > float(cfg["vz_exit"]) * root_vxy)
        )
        self.feet_standing[:] = (self.feet_standing & (~exit_contact)) | enter_contact

    def update_reward_target(self):
        raw_ref_root_pos_w = self._motion.root_pos_w[:, 0] + self.env.scene.env_origins
        raw_ref_root_quat_w = self._motion.root_quat_w[:, 0]
        current_root_pos_w = self.asset.data.root_link_pos_w
        raw_ref_root_xy_w = raw_ref_root_pos_w[:, :2]
        history_slot = self._reward_root_history_slot
        env_ids = self.dataset._all_env_ids
        history_ref_xy_w = self.reward_root_ref_xy_history_w[env_ids, history_slot]
        history_actual_xy_w = self.reward_root_actual_xy_history_w[env_ids, history_slot]
        ref_delta_xy_w = raw_ref_root_xy_w - history_ref_xy_w
        sliding_expected_xy_w = history_actual_xy_w + ref_delta_xy_w

        self.reward_root_pos_w[:] = raw_ref_root_pos_w
        self.reward_root_quat_w[:] = raw_ref_root_quat_w
        if self.env.student_train:
            self.reward_root_pos_w[:, :2] = sliding_expected_xy_w

        self.reward_root_ref_xy_history_w[env_ids, history_slot] = raw_ref_root_xy_w
        self.reward_root_actual_xy_history_w[env_ids, history_slot] = current_root_pos_w[:, :2]
        self._reward_root_history_slot.add_(1)
        self._reward_root_history_slot.remainder_(self.reward_root_history_len)

    def before_update(self):
        self.t = torch.clamp_max(self.t + 1, self.lengths - 1)
        self.boot_indicator[:] = torch.clamp_min(self.boot_indicator - 1, 0)
        teacher_motion_raw, student_motion_raw = self.dataset.get_teacher_student_slice(
            None,
            self.t,
            teacher_steps=self.future_steps,
            student_steps=self.student_future_steps,
        )
        self._motion, raw_student_motion = self._apply_motion_origin_offset(teacher_motion_raw, student_motion_raw)
        self._motion_student = self._transform_student_motion(raw_student_motion)

        self.update_feet_standing()
        self.update_current_keypoint_cache()
        self.update_reward_target()
        self.update_shared_noisy_joint_pos()
        self.update_shared_noisy_joint_vel()
        # self.sample_init_robot(torch.arange(self.num_envs, device=self.device), self._motion, lift_height=0.0)

    def update(self):
        pass

    def _motion_qpos_for_debug(self, motion, env_ids: torch.Tensor, root_offset_w=None) -> torch.Tensor:
        root_pos_w = motion.root_pos_w[env_ids, 0] + self.env.scene.env_origins[env_ids]
        if root_offset_w is not None:
            root_pos_w = root_pos_w + root_pos_w.new_tensor(root_offset_w)
        root_quat_w = motion.root_quat_w[env_ids, 0]
        joint_pos = self.asset.data.default_joint_pos[env_ids].clone()
        joint_pos[:, self.joint_idx_asset] = motion.joint_pos[env_ids, 0][:, self.joint_idx_motion]
        qpos = torch.cat([root_pos_w, root_quat_w, joint_pos], dim=-1)
        if qpos.shape[-1] != self.env.sim.mj_model.nq:
            raise RuntimeError(
                f"Ghost qpos width {qpos.shape[-1]} does not match model.nq {self.env.sim.mj_model.nq}."
            )
        return qpos

    def debug_draw(self):
        root_pos = self.asset.data.root_link_pos_w    # [N,1,3]
        root_quat = self.asset.data.root_link_quat_w  # [N,1,4]
        target_root_quat = self.reward_root_quat_w  # [N,1,4]
        heading_rel = torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(self.num_envs, 3)
        heading_world = quat_apply(root_quat, heading_rel)
        heading_world_target = quat_apply(target_root_quat, heading_rel)

        # Draw the robot's heading
        self.env.debug_draw.vector(
            root_pos.reshape(-1, 3),
            heading_world.reshape(-1, 3),
            color=(0, 0, 1, 2)
        )
        # Draw the target heading
        self.env.debug_draw.vector(
            self.reward_root_pos_w.reshape(-1, 3),
            heading_world_target.reshape(-1, 3),
            color=(1, 0, 0, 2)
        )

        # target_keypoints_w align the target root with the robot root
        tgt_rel = self._motion.body_pos_w[:, 0] - self._motion.root_pos_w[:, 0].unsqueeze(1)
        target_keypoints_w_all = tgt_rel + self.asset.data.root_link_pos_w.unsqueeze(1)
        target_keypoints_w = target_keypoints_w_all[:, self.keypoint_idx_motion]
        robot_keypoints_w = self.asset.data.body_link_pos_w[:, self.keypoint_idx_asset]

        # draw points and error vectors
        # self.env.debug_draw.point( target_keypoints_w.reshape(-1, 3), color=(1, 0, 0, 1), size=40.0)
        self.env.debug_draw.vector(
            robot_keypoints_w.reshape(-1, 3),
            (target_keypoints_w - robot_keypoints_w).reshape(-1, 3),
            color=(0, 0, 1, 1)
        )

        # Draw teacher and student motions as translucent full-body ghosts.
        ghost_env_ids = torch.as_tensor(
            tuple(self.env.debug_draw.env_indices(self.num_envs)),
            device=self.device,
            dtype=torch.long,
        )
        if ghost_env_ids.numel() > 0:
            self.env.debug_draw.ghost(
                self._motion_qpos_for_debug(
                    self._motion,
                    ghost_env_ids,
                    root_offset_w=(0.3, 0.0, 0.0),
                ),
                self.env.sim.mj_model,
                color=(0.0, 1.0, 0.0),
                alpha=0.5,
            )
            self.env.debug_draw.ghost(
                self._motion_qpos_for_debug(
                    self._motion_student,
                    ghost_env_ids,
                    root_offset_w=(0.6, 0.0, 0.0),
                ),
                self.env.sim.mj_model,
                color=(1.0, 0.5, 0.0),
                alpha=0.5,
            )
