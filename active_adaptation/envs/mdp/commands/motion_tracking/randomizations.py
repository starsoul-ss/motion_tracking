import math

import torch
from active_adaptation.utils.math import quat_from_angle_axis, quat_mul

from active_adaptation.envs.mdp.utils import (
    _rand_unit_vectors,
    add_spherical_noise,
    perturb_quaternion,
    random_noise,
    resolve_named_std,
    sample_uniform,
)


class SinusoidalScalarDrift:
    def __init__(self, num_envs: int, *, device, dtype=torch.float32):
        self.device = device
        self.amplitude = torch.zeros(num_envs, device=device, dtype=dtype)
        self.omega = torch.zeros(num_envs, device=device, dtype=dtype)
        self.phase = torch.zeros(num_envs, device=device, dtype=dtype)

    def _range_bounds(self, value_range: tuple[float | torch.Tensor, float | torch.Tensor], shape: tuple[int, ...]):
        low = torch.as_tensor(value_range[0], device=self.device, dtype=self.amplitude.dtype)
        high = torch.as_tensor(value_range[1], device=self.device, dtype=self.amplitude.dtype)
        return low.expand(shape), high.expand(shape)

    def _sample_uniform_range(self, shape: tuple[int, ...], value_range: tuple[float | torch.Tensor, float | torch.Tensor]):
        low, high = self._range_bounds(value_range, shape)
        return torch.rand(shape, device=self.device, dtype=self.amplitude.dtype) * (high - low) + low

    def _sample_log_uniform_range(self, shape: tuple[int, ...], value_range: tuple[float | torch.Tensor, float | torch.Tensor]):
        low, high = self._range_bounds(value_range, shape)
        return (torch.rand(shape, device=self.device, dtype=self.amplitude.dtype) * (high.log() - low.log()) + low.log()).exp()

    def reset_from_amplitude_range(self, env_ids: torch.Tensor, *, amplitude_range: tuple[float, float], freq_range: tuple[float, float]):
        n = env_ids.numel()
        self.amplitude[env_ids] = self._sample_uniform_range((n,), amplitude_range)
        freq = self._sample_log_uniform_range((n,), freq_range)
        self.omega[env_ids] = freq * (2.0 * math.pi)
        self.phase[env_ids] = sample_uniform((n,), 0.0, 2.0 * math.pi, device=self.device)

    def reset_from_speed_range(self, env_ids: torch.Tensor, *, speed_range: tuple[float, float], freq_range: tuple[float, float]):
        n = env_ids.numel()
        freq = self._sample_log_uniform_range((n,), freq_range)
        omega = freq * (2.0 * math.pi)
        speed = self._sample_uniform_range((n,), speed_range)
        self.amplitude[env_ids] = speed / omega
        self.omega[env_ids] = omega
        self.phase[env_ids] = sample_uniform((n,), 0.0, 2.0 * math.pi, device=self.device)

    def value(self, time_s: torch.Tensor):
        return self.amplitude.unsqueeze(1) * torch.sin(self.omega.unsqueeze(1) * time_s + self.phase.unsqueeze(1))


class MotionTrackingRandomizationMixin:
    def _init_student_motion_randomization(self):
        cfg = self.student_motion_randomization_cfg
        root_pos_drift = cfg["root_pos_drift"]
        root_rot_drift = cfg["root_rot_drift"]
        as_range = lambda value: (float(value[0]), float(value[1]))

        self._student_root_pos_noise_std = float(cfg["root_pos_noise_std"])
        self._student_root_ori_noise_std = float(cfg["root_ori_noise_std"])
        self._student_joint_pos_noise_std = float(cfg["joint_pos_noise_std"])
        self._student_root_z_offset_range_m = as_range(root_pos_drift["root_z_offset_range_m"])
        self._student_root_xy_speed_range = as_range(root_pos_drift["xy_speed_range"])
        self._student_root_xy_freq_range_hz = as_range(root_pos_drift["xy_freq_range_hz"])
        self._student_root_z_amplitude_range_m = as_range(root_pos_drift["z_amplitude_range_m"])
        self._student_root_z_freq_range_hz = as_range(root_pos_drift["z_freq_range_hz"])
        self._student_root_rot_amplitude_range_rad = as_range(root_rot_drift["amplitude_range_rad"])
        self._student_root_rot_freq_range_hz = as_range(root_rot_drift["freq_range_hz"])
        self._student_joint_pos_bias_std = resolve_named_std(
            cfg["joint_pos_bias_std"],
            self.joint_names,
            self.device,
            torch.float32,
            "student_motion_randomization.joint_pos_bias_std",
        )
        if self._student_joint_pos_bias_std is None:
            self._student_joint_pos_bias_std = torch.zeros(len(self.joint_names), device=self.device, dtype=torch.float32)

    @staticmethod
    def _parse_shared_joint_pos_cfg(cfg: dict | None):
        cfg = {} if cfg is None else dict(cfg)
        return {
            "std": cfg.get("std", 0.0),
            "bias": cfg.get("bias", 0.0),
        }

    @staticmethod
    def _parse_shared_joint_vel_cfg(cfg: dict | None):
        cfg = {} if cfg is None else dict(cfg)
        return {
            "std": cfg.get("std", 0.0),
        }

    def _init_randomization_state(self):
        self.shared_joint_pos_cfg = self._parse_shared_joint_pos_cfg(self.shared_joint_pos_cfg)
        self.shared_joint_vel_cfg = self._parse_shared_joint_vel_cfg(self.shared_joint_vel_cfg)
        names = list(self.asset.joint_names)
        dtype = self.asset.data.joint_pos.dtype
        self._shared_joint_pos_std = resolve_named_std(
            self.shared_joint_pos_cfg["std"],
            names,
            self.device,
            dtype,
            "shared_joint_pos.std",
        )
        self._shared_joint_pos_bias_std = resolve_named_std(
            self.shared_joint_pos_cfg["bias"],
            names,
            self.device,
            dtype,
            "shared_joint_pos.bias",
        )
        if self._shared_joint_pos_std is None:
            self._shared_joint_pos_std = torch.zeros(len(names), device=self.device, dtype=dtype)
        if self._shared_joint_pos_bias_std is None:
            self._shared_joint_pos_bias_std = torch.zeros(len(names), device=self.device, dtype=dtype)
        self._shared_joint_pos_std_enabled = bool((self._shared_joint_pos_std > 0.0).any())
        self._shared_joint_pos_bias_enabled = bool((self._shared_joint_pos_bias_std > 0.0).any())
        self.shared_joint_pos_noise_enabled = (
            self._shared_joint_pos_std_enabled or self._shared_joint_pos_bias_enabled
        )
        self._shared_noisy_joint_pos = torch.empty_like(self.asset.data.joint_pos)
        self._shared_joint_pos_bias = torch.zeros_like(self.asset.data.joint_pos)

        vel_dtype = self.asset.data.joint_vel.dtype
        self._shared_joint_vel_std = resolve_named_std(
            self.shared_joint_vel_cfg["std"],
            names,
            self.device,
            vel_dtype,
            "shared_joint_vel.std",
        )
        if self._shared_joint_vel_std is None:
            self._shared_joint_vel_std = torch.zeros(len(names), device=self.device, dtype=vel_dtype)
        self._shared_joint_vel_std_enabled = bool((self._shared_joint_vel_std > 0.0).any())
        self.shared_joint_vel_noise_enabled = self._shared_joint_vel_std_enabled
        self._shared_noisy_joint_vel = torch.empty_like(self.asset.data.joint_vel)

        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._resample_shared_joint_pos_bias(env_ids)
        self.update_shared_noisy_joint_pos()
        self.update_shared_noisy_joint_vel()

        self.student_motion_randomization_cfg = dict(self.student_motion_randomization_cfg)
        if not self.student_motion_randomization_cfg["enable"]:
            return
        self._init_student_motion_randomization()
        # Root position drift state
        self._root_pos_z_offset_m = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self._root_pos_drift_xy_dir_w = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.float32)
        self._root_pos_xy_drift = SinusoidalScalarDrift(self.num_envs, device=self.device, dtype=torch.float32)
        self._root_pos_z_drift = SinusoidalScalarDrift(self.num_envs, device=self.device, dtype=torch.float32)
        # Root rotation drift state
        self._root_rot_drift_axis_w = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32)
        self._root_rot_angle_drift = SinusoidalScalarDrift(self.num_envs, device=self.device, dtype=torch.float32)
        # Joint position bias state
        self._student_joint_pos_bias = torch.zeros(self.num_envs, len(self.joint_names), device=self.device, dtype=torch.float32)

    def reset(self, env_ids):
        super().reset(env_ids)
        if env_ids.numel() == 0:
            return
        self._resample_shared_joint_pos_bias(env_ids)
        self.update_shared_noisy_joint_pos()
        self.update_shared_noisy_joint_vel()
        if not self.student_motion_randomization_cfg["enable"]:
            return
        self._resample_student_joint_pos_bias(env_ids)
        self._resample_student_root_pos_drift(env_ids)
        self._resample_student_root_rot_drift(env_ids)

    def _resample_shared_joint_pos_bias(self, env_ids: torch.Tensor):
        self._shared_joint_pos_bias[env_ids] = 0.0
        if not self._shared_joint_pos_bias_enabled:
            return
        n = env_ids.numel()
        joint_noise = sample_uniform((n, self._shared_joint_pos_bias.shape[1]), -1.0, 1.0, device=self.device)
        self._shared_joint_pos_bias[env_ids] = (
            joint_noise.to(self._shared_joint_pos_bias.dtype) * self._shared_joint_pos_bias_std.unsqueeze(0)
        )

    def update_shared_noisy_joint_pos(self):
        if not self.shared_joint_pos_noise_enabled:
            return
        joint_pos = self.asset.data.joint_pos + self._shared_joint_pos_bias
        if self._shared_joint_pos_std_enabled:
            joint_pos = random_noise(joint_pos, self._shared_joint_pos_std)
        self._shared_noisy_joint_pos.copy_(joint_pos)

    def get_shared_noisy_joint_pos(self, joint_ids: torch.Tensor | None = None) -> torch.Tensor:
        source = self._shared_noisy_joint_pos if self.shared_joint_pos_noise_enabled else self.asset.data.joint_pos
        if joint_ids is None:
            return source
        return source[:, joint_ids]

    def update_shared_noisy_joint_vel(self):
        if not self.shared_joint_vel_noise_enabled:
            return
        joint_vel = random_noise(self.asset.data.joint_vel, self._shared_joint_vel_std)
        self._shared_noisy_joint_vel.copy_(joint_vel)

    def get_shared_noisy_joint_vel(self, joint_ids: torch.Tensor | None = None) -> torch.Tensor:
        source = self._shared_noisy_joint_vel if self.shared_joint_vel_noise_enabled else self.asset.data.joint_vel
        if joint_ids is None:
            return source
        return source[:, joint_ids]

    def _resample_student_joint_pos_bias(self, env_ids: torch.Tensor):
        n = env_ids.numel()
        joint_noise = sample_uniform((n, self._student_joint_pos_bias.shape[1]), -1.0, 1.0, device=self.device)
        self._student_joint_pos_bias[env_ids] = joint_noise * self._student_joint_pos_bias_std.unsqueeze(0)

    def _resample_student_root_pos_drift(self, env_ids: torch.Tensor):
        n = env_ids.numel()

        self._root_pos_z_offset_m[env_ids] = sample_uniform(
            (n,),
            self._student_root_z_offset_range_m[0],
            self._student_root_z_offset_range_m[1],
            device=self.device,
        )
        self._root_pos_drift_xy_dir_w[env_ids] = _rand_unit_vectors((n, 2), device=self.device, dtype=torch.float32)
        self._root_pos_xy_drift.reset_from_speed_range(
            env_ids,
            speed_range=self._student_root_xy_speed_range,
            freq_range=self._student_root_xy_freq_range_hz,
        )
        self._root_pos_z_drift.reset_from_amplitude_range(
            env_ids,
            amplitude_range=self._student_root_z_amplitude_range_m,
            freq_range=self._student_root_z_freq_range_hz,
        )

    def _resample_student_root_rot_drift(self, env_ids: torch.Tensor):
        self._root_rot_drift_axis_w[env_ids] = _rand_unit_vectors((env_ids.numel(), 3), device=self.device, dtype=torch.float32)
        self._root_rot_angle_drift.reset_from_amplitude_range(
            env_ids,
            amplitude_range=self._student_root_rot_amplitude_range_rad,
            freq_range=self._student_root_rot_freq_range_hz,
        )

    def _apply_root_pos_drift(self, root_pos_w: torch.Tensor, time_s: torch.Tensor):
        xy_scalar = self._root_pos_xy_drift.value(time_s)
        xy_offset = xy_scalar.unsqueeze(-1) * self._root_pos_drift_xy_dir_w.unsqueeze(1)
        z_offset = self._root_pos_z_drift.value(time_s)

        offsets = root_pos_w.new_zeros(root_pos_w.shape)
        offsets[..., :2] = xy_offset
        offsets[..., 2] = z_offset + self._root_pos_z_offset_m.unsqueeze(1)
        return root_pos_w + offsets

    def _apply_root_rot_drift(self, root_quat_w: torch.Tensor, time_s: torch.Tensor):
        num_steps = root_quat_w.shape[1]
        angle = self._root_rot_angle_drift.value(time_s)
        axis_w = self._root_rot_drift_axis_w.unsqueeze(1).expand(-1, num_steps, -1)
        delta_quat = quat_from_angle_axis(angle, axis_w)
        return quat_mul(delta_quat, root_quat_w)

    def _transform_student_motion(self, motion):
        motion = motion.clone()
        if not self.student_motion_randomization_cfg["enable"]:
            return motion

        time_s = (self.t.unsqueeze(1) + self.student_future_steps.unsqueeze(0)) * float(self.env.step_dt)
        root_pos_w = motion.root_pos_w
        root_pos_w = self._apply_root_pos_drift(root_pos_w, time_s)
        root_pos_w = add_spherical_noise(root_pos_w, self._student_root_pos_noise_std)
        motion.root_pos_w = root_pos_w

        root_quat_w = motion.root_quat_w
        root_quat_w = self._apply_root_rot_drift(root_quat_w, time_s)
        root_quat_w = perturb_quaternion(root_quat_w, self._student_root_ori_noise_std)
        motion.root_quat_w = root_quat_w

        joint_pos = motion.joint_pos + self._student_joint_pos_bias.unsqueeze(1).to(motion.joint_pos.dtype)
        joint_pos = random_noise(joint_pos, self._student_joint_pos_noise_std)
        motion.joint_pos = joint_pos

        return motion
