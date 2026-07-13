import torch
import numpy as np
import logging
from typing import TYPE_CHECKING, Dict, Tuple
import warp as wp

import mjlab.utils.lab_api.string as string_utils
from mjlab.managers.event_manager import RecomputeLevel
from active_adaptation.envs.mdp.utils import (
    _rand_unit_vectors,
    add_spherical_noise,
    log_uniform,
    sample_log_uniform,
    sample_uniform,
    uniform,
    uniform_like,
)
import active_adaptation.utils.symmetry as sym_utils
from active_adaptation.utils.window_load_capacity import WindowLoadCapacityLookup


if TYPE_CHECKING:
    from mjlab.entity import Entity as Articulation

from .base import Randomization

class motor_params_implicit(Randomization):
    def __init__(
        self,
        env,
        stiffness_range,
        damping_range,
        armature_range,
        frictionloss_range=None,
        mode: str = "log_uniform",
    ):
        super().__init__(env)
        self.ensure_model_fields_expanded(
            "actuator_gainprm",
            "actuator_biasprm",
            "dof_armature",
            "dof_frictionloss",
        )
        self.ensure_recompute_fields_expanded(RecomputeLevel.set_const_0)
        self.asset: Articulation = self.env.scene["robot"]
        self.mode = str(mode).strip().lower()
        if self.mode not in {"uniform", "log_uniform"}:
            raise ValueError(
                f"motor_params_implicit.mode must be 'uniform' or 'log_uniform', got {mode!r}."
            )

        # 存下区间字典
        self.stiffness_range = dict(stiffness_range)
        self.damping_range   = dict(damping_range)
        self.armature_range  = dict(armature_range)
        self.frictionloss_range = {} if frictionloss_range is None else dict(frictionloss_range)
        self.model = self.env.sim.model
        # ------- stiffness / damping via actuator gains (mjlab randomize_pd_gains style) -------
        kp_ids, kp_names, kp_ranges = string_utils.resolve_matching_names_values(
            self.stiffness_range, self.asset.actuator_names
        )
        kd_ids, kd_names, kd_ranges = string_utils.resolve_matching_names_values(
            self.damping_range, self.asset.actuator_names
        )

        self.kp_ctrl_ids = self.asset.indexing.ctrl_ids[
            torch.tensor(kp_ids, device=self.device, dtype=torch.long)
        ]
        self.kd_ctrl_ids = self.asset.indexing.ctrl_ids[
            torch.tensor(kd_ids, device=self.device, dtype=torch.long)
        ]

        default_gainprm = self.env.sim.get_default_field("actuator_gainprm")
        default_biasprm = self.env.sim.get_default_field("actuator_biasprm")

        self.kp_gain_def = default_gainprm[self.kp_ctrl_ids, 0]
        self.kp_bias_def = default_biasprm[self.kp_ctrl_ids, 1]
        self.kd_bias_def = default_biasprm[self.kd_ctrl_ids, 2]

        kp_low, kp_high = torch.tensor(kp_ranges, device=self.device).unbind(1)
        kd_low, kd_high = torch.tensor(kd_ranges, device=self.device).unbind(1)
        self._validate_sampling_range("stiffness_range", kp_low, kp_high)
        self._validate_sampling_range("damping_range", kd_low, kd_high)

        self.kp_low = kp_low
        self.kp_high = kp_high
        self.kd_low = kd_low
        self.kd_high = kd_high

        # ------- armature (改为相对值) -------
        arm_ids, arm_names, arm_ranges = string_utils.resolve_matching_names_values(
            self.armature_range, self.asset.joint_names
        )
        self.arm_ids = torch.tensor(arm_ids, device=self.device, dtype=torch.long)
        self.arm_dof_ids = self.asset.indexing.joint_v_adr[self.arm_ids]
        default_armature = self.env.sim.get_default_field("dof_armature")
        self.arm_def = default_armature[self.arm_dof_ids]

        arm_low, arm_high = torch.tensor(arm_ranges, device=self.device).unbind(1)
        self._validate_sampling_range("armature_range", arm_low, arm_high)
        self.arm_low = arm_low
        self.arm_high = arm_high

        # dry friction loss is an absolute torque/force value, not a scale.
        fric_ids, fric_names, fric_ranges = string_utils.resolve_matching_names_values(
            self.frictionloss_range, self.asset.joint_names
        )
        self.fric_ids = torch.tensor(fric_ids, device=self.device, dtype=torch.long)
        self.fric_dof_ids = self.asset.indexing.joint_v_adr[self.fric_ids]
        if len(fric_ranges) > 0:
            fric_low, fric_high = torch.tensor(fric_ranges, device=self.device).unbind(1)
        else:
            fric_low = torch.empty(0, device=self.device)
            fric_high = torch.empty(0, device=self.device)
        self._validate_absolute_range("frictionloss_range", fric_low, fric_high)
        self.fric_low = fric_low
        self.fric_high = fric_high

        self.kp_names = kp_names
        self.kd_names = kd_names
        self.arm_names = arm_names
        self.fric_names = fric_names
        self._obs_kp_scale = torch.ones((self.num_envs, len(kp_ids)), device=self.device, dtype=torch.float32)
        self._obs_kd_scale = torch.ones((self.num_envs, len(kd_ids)), device=self.device, dtype=torch.float32)
        self._obs_arm_scale = torch.ones((self.num_envs, len(arm_ids)), device=self.device, dtype=torch.float32)
        self._obs_frictionloss = torch.zeros((self.num_envs, len(fric_ids)), device=self.device, dtype=torch.float32)

    def _validate_sampling_range(self, range_name: str, low: torch.Tensor, high: torch.Tensor):
        if torch.any(high < low):
            raise ValueError(f"{range_name} must satisfy low <= high, got low={low.tolist()}, high={high.tolist()}")
        if self.mode == "log_uniform" and (torch.any(low <= 0.0) or torch.any(high <= 0.0)):
            raise ValueError(
                f"{range_name} must be strictly positive for log-uniform sampling, "
                f"got low={low.tolist()}, high={high.tolist()}"
            )

    @staticmethod
    def _validate_absolute_range(range_name: str, low: torch.Tensor, high: torch.Tensor):
        if torch.any(high < low):
            raise ValueError(f"{range_name} must satisfy low <= high, got low={low.tolist()}, high={high.tolist()}")
        if torch.any(low < 0.0):
            raise ValueError(f"{range_name} must be non-negative, got low={low.tolist()}")

    def _rand_uniform(self, n_env: int, low: torch.Tensor, high: torch.Tensor):
        low_expand = low.unsqueeze(0).expand(n_env, -1)
        high_expand = high.unsqueeze(0).expand(n_env, -1)
        return uniform(low_expand, high_expand)

    def _rand_log_uniform(self, n_env: int, low: torch.Tensor, high: torch.Tensor):
        low_expand = low.unsqueeze(0).expand(n_env, -1)
        high_expand = high.unsqueeze(0).expand(n_env, -1)
        return log_uniform(low_expand, high_expand)

    def _rand_range(self, n_env: int, low: torch.Tensor, high: torch.Tensor):
        if self.mode == "uniform":
            return self._rand_uniform(n_env, low, high)
        if self.mode == "log_uniform":
            return self._rand_log_uniform(n_env, low, high)
        raise RuntimeError(f"Unsupported motor_params_implicit.mode: {self.mode!r}")

    def _randomize_pd_gain(self, env_ids: torch.Tensor):
        n_env = env_ids.numel()
        if n_env == 0:
            return

        if self.kp_ctrl_ids.numel() > 0:
            kp_samples = self._rand_range(n_env, self.kp_low, self.kp_high)
            kp_gain = self.kp_gain_def.unsqueeze(0) * kp_samples
            kp_bias = self.kp_bias_def.unsqueeze(0) * kp_samples
            self.model.actuator_gainprm[env_ids.unsqueeze(1), self.kp_ctrl_ids, 0] = kp_gain
            self.model.actuator_biasprm[env_ids.unsqueeze(1), self.kp_ctrl_ids, 1] = kp_bias
            self._obs_kp_scale[env_ids] = kp_samples

        if self.kd_ctrl_ids.numel() > 0:
            kd_samples = self._rand_range(n_env, self.kd_low, self.kd_high)
            kd_bias = self.kd_bias_def.unsqueeze(0) * kd_samples
            self.model.actuator_biasprm[env_ids.unsqueeze(1), self.kd_ctrl_ids, 2] = kd_bias
            self._obs_kd_scale[env_ids] = kd_samples

    def startup(self):
        n_env = self.num_envs

        # armature
        if self.arm_dof_ids.numel() > 0:
            arma = self._rand_range(n_env, self.arm_low, self.arm_high)
            self.model.dof_armature[:, self.arm_dof_ids] = (
                self.arm_def.unsqueeze(0) * arma
            )
            self._obs_arm_scale[:] = arma
            self.env.sim.recompute_constants(RecomputeLevel.set_const_0)

        if self.arm_dof_ids.numel() > 0:
            assert torch.allclose(self.model.dof_armature[:, self.arm_dof_ids], self.arm_def.unsqueeze(0) * arma)

        if self.fric_dof_ids.numel() > 0:
            frictionloss = self._rand_uniform(n_env, self.fric_low, self.fric_high)
            self.model.dof_frictionloss[:, self.fric_dof_ids] = frictionloss
            self._obs_frictionloss[:] = frictionloss
            assert torch.allclose(self.model.dof_frictionloss[:, self.fric_dof_ids], frictionloss)

    # ----------------------------------------------------------
    def reset(self, env_ids):
        self._randomize_pd_gain(env_ids)

    def has_observation(self) -> bool:
        return True

    def observe(self, **kwargs) -> torch.Tensor:
        return torch.cat(
            [self._obs_kp_scale, self._obs_kd_scale, self._obs_arm_scale, self._obs_frictionloss],
            dim=-1,
        )

    def observe_sym(self, **kwargs):
        transforms = []
        if len(self.kp_names) > 0:
            transforms.append(sym_utils.joint_space_symmetry(self.asset, self.kp_names))
        if len(self.kd_names) > 0:
            transforms.append(sym_utils.joint_space_symmetry(self.asset, self.kd_names))
        if len(self.arm_names) > 0:
            transforms.append(sym_utils.joint_space_symmetry(self.asset, self.arm_names))
        if len(self.fric_names) > 0:
            transforms.append(sym_utils.joint_space_symmetry(self.asset, self.fric_names))
        return sym_utils.SymmetryTransform.cat(transforms)


class perturb_body_materials(Randomization):
    def __init__(
        self,
        env,
        body_names,
        static_friction_range=(0.6, 1.0),
        solref_time_constant_range=(0.02, 0.02),
        solref_dampratio_range=(1.0, 1.0),
        homogeneous: bool = False,
    ):
        super().__init__(env)
        self.ensure_model_fields_expanded("geom_friction", "geom_solref")
        self.asset: Articulation = self.env.scene["robot"]
        self.body_ids, self.body_names = self.asset.find_bodies(body_names)

        self.static_friction_range = static_friction_range
        self.solref_time_constant_range = solref_time_constant_range
        self.solref_dampratio_range = solref_dampratio_range
        self.homogeneous = homogeneous

        if len(self.body_ids) == 0:
            raise ValueError(
                "No bodies matched the provided names for material perturbation."
            )

        local_body_ids = torch.as_tensor(
            self.body_ids, device=self.device, dtype=torch.long
        )
        self.global_body_ids = self.asset.indexing.body_ids[local_body_ids]
        selected_body_set = set(self.global_body_ids.cpu().tolist())

        geom_global_ids = self.asset.indexing.geom_ids.cpu().tolist()
        geom_names = self.asset.geom_names
        selected_geom_local: list[int] = []
        selected_geom_global: list[int] = []
        selected_geom_names: list[str] = []

        cpu_model = self.env.sim.mj_model
        for local_idx, global_idx in enumerate(geom_global_ids):
            body_id = int(cpu_model.geom_bodyid[global_idx])
            if body_id in selected_body_set:
                selected_geom_local.append(local_idx)
                selected_geom_global.append(global_idx)
                selected_geom_names.append(geom_names[local_idx])

        if not selected_geom_global:
            raise ValueError(
                "No geoms found for the specified bodies when configuring material perturbation."
            )

        self.geom_local_ids = torch.as_tensor(
            selected_geom_local, device=self.device, dtype=torch.long
        )
        self.geom_global_ids = torch.as_tensor(
            selected_geom_global, device=self.device, dtype=torch.long
        )
        self.geom_names = selected_geom_names
        self._obs_static_friction = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.float32)
        self._obs_solref_time_constant = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.float32)
        self._obs_solref_dampratio = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.float32)

    def startup(self):
        logging.info(f"Randomize body materials of {self.geom_names} upon startup.")

        num_geoms = self.geom_global_ids.numel()
        sample_cols = 1 if self.homogeneous else num_geoms
        shape = (self.num_envs, sample_cols)

        sf = sample_uniform(shape, *self.static_friction_range, device=self.device)
        tc = sample_uniform(shape, *self.solref_time_constant_range, device=self.device)
        dr = sample_log_uniform(shape, *self.solref_dampratio_range, device=self.device)

        if sample_cols == 1:
            sf = sf.expand(-1, num_geoms)
            tc = tc.expand(-1, num_geoms)
            dr = dr.expand(-1, num_geoms)

        model = self.env.sim.model
        model.geom_friction[:, self.geom_global_ids, 0] = sf
        model.geom_solref[:, self.geom_global_ids, 0] = tc
        model.geom_solref[:, self.geom_global_ids, 1] = dr
        self._obs_static_friction[:] = sf[:, :1]
        self._obs_solref_time_constant[:] = tc[:, :1]
        self._obs_solref_dampratio[:] = dr[:, :1]

        assert torch.allclose(model.geom_friction[:, self.geom_global_ids, 0], sf)
        assert torch.allclose(model.geom_solref[:, self.geom_global_ids, 0], tc)
        assert torch.allclose(model.geom_solref[:, self.geom_global_ids, 1], dr)

    def has_observation(self) -> bool:
        return self.homogeneous

    def observe(self, **kwargs) -> torch.Tensor:
        if not self.homogeneous:
            raise NotImplementedError(
                "domain_perturb_body_materials only supports observation when homogeneous=True."
            )
        return torch.cat(
            [
                self._obs_static_friction,
                self._obs_solref_time_constant,
                self._obs_solref_dampratio,
            ],
            dim=-1,
        )

    def observe_sym(self, **kwargs):
        if self.homogeneous:
            dim = self.observe().shape[-1]
            return sym_utils.SymmetryTransform(torch.arange(dim), torch.ones(dim))
        raise NotImplementedError(
            "domain_perturb_body_materials does not support symmetry when homogeneous=False."
        )

class perturb_body_mass(Randomization):
    def __init__(
        self, env, **perturb_ranges: Tuple[float, float]
    ):
        super().__init__(env)
        self.ensure_model_fields_expanded("body_mass", "body_inertia")
        self.ensure_recompute_fields_expanded(RecomputeLevel.set_const)
        self.asset: Articulation = self.env.scene["robot"]
        if not perturb_ranges:
            raise ValueError("perturb_body_mass requires at least one body range entry.")

        body_ids, body_names, values = string_utils.resolve_matching_names_values(
            perturb_ranges, self.asset.body_names
        )
        if len(body_ids) == 0:
            raise ValueError(
                "No bodies matched the provided patterns for mass perturbation."
            )

        self.body_names = body_names
        self.local_body_ids = torch.as_tensor(body_ids, device=self.device, dtype=torch.long)
        self.global_body_ids = self.asset.indexing.body_ids[self.local_body_ids]
        self.mass_ranges = torch.as_tensor(values, device=self.device, dtype=torch.float32)

        model = self.env.sim.model
        self._default_mass = model.body_mass[:, self.global_body_ids].clone()
        self._default_inertia = model.body_inertia[:, self.global_body_ids].clone()


    def startup(self):
        logging.info(f"Randomize body masses of {self.body_names} upon startup.")
        low = self.mass_ranges[:, 0].unsqueeze(0)
        high = self.mass_ranges[:, 1].unsqueeze(0)
        scale = uniform_like(self._default_mass, low, high)

        model = self.env.sim.model
        new_mass = self._default_mass * scale
        new_inertia = self._default_inertia * scale.unsqueeze(-1)
        model.body_mass[:, self.global_body_ids] = new_mass
        model.body_inertia[:, self.global_body_ids] = new_inertia
        self.env.sim.recompute_constants(RecomputeLevel.set_const)

        assert torch.allclose(model.body_mass[:, self.global_body_ids], new_mass)
        assert torch.allclose(model.body_inertia[:, self.global_body_ids], new_inertia)


class perturb_body_com(Randomization):
    def __init__(self, env, body_names = ".*", com_range=(-0.05, 0.05)):
        super().__init__(env)
        self.ensure_model_fields_expanded("body_ipos")
        self.ensure_recompute_fields_expanded(RecomputeLevel.set_const)
        self.asset: Articulation = self.env.scene["robot"]
        self.body_ids, self.body_names = self.asset.find_bodies(body_names)
        if len(self.body_ids) == 0:
            raise ValueError("No bodies matched the provided names for COM perturbation.")

        self.com_range = tuple(com_range)
        self.local_body_ids = torch.as_tensor(self.body_ids, device=self.device, dtype=torch.long)
        self.global_body_ids = self.asset.indexing.body_ids[self.local_body_ids]

        model = self.env.sim.model
        self._default_body_ipos = model.body_ipos[:, self.global_body_ids].clone()
    
    def startup(self):
        num_bodies = self.global_body_ids.numel()
        low, high = self.com_range
        offsets = sample_uniform((self.num_envs, num_bodies, 3), low, high, device=self.device)

        model = self.env.sim.model
        new_ipos = self._default_body_ipos + offsets
        model.body_ipos[:, self.global_body_ids] = new_ipos
        self.env.sim.recompute_constants(RecomputeLevel.set_const)

        assert torch.allclose(model.body_ipos[:, self.global_body_ids], new_ipos)


class random_joint_offset(Randomization):
    def __init__(self, env, **offset_range: Tuple[float, float]):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids, self.joint_names, self.offset_range = string_utils.resolve_matching_names_values(dict(offset_range), self.asset.joint_names)
        
        self.joint_ids = torch.tensor(self.joint_ids, device=self.device)
        self.offset_range = torch.tensor(self.offset_range, device=self.device)

        self.action_manager = self.env.action_manager

    def reset(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return
        low = self.offset_range[:, 0].unsqueeze(0)
        high = self.offset_range[:, 1].unsqueeze(0)
        offset = uniform(
            low.expand(env_ids.numel(), -1),
            high.expand(env_ids.numel(), -1),
        )
        self.action_manager.offset[env_ids.unsqueeze(1), self.joint_ids] = offset

    def has_observation(self) -> bool:
        return True

    def observe(self, **kwargs) -> torch.Tensor:
        return self.action_manager.offset[:, self.joint_ids]

    def observe_sym(self, **kwargs):
        return sym_utils.joint_space_symmetry(self.asset, self.joint_names)


class perturb_body_wrench(Randomization):
    def __init__(
        self,
        env,
        body_name: str,
        total_duration_range_s: Tuple[float, float],
        active_duration_range_s: Tuple[float, float],
        force_magnitude_range_n: Tuple[float, float],
        lever_arm_length_range_m: Tuple[float, float],
        enabled: bool = True,
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.enabled = bool(enabled)
        self.body_name = str(body_name)
        if not self.enabled:
            self.body_id = -1
            self.body_ids = torch.empty(0, device=self.device, dtype=torch.long)
            return

        body_matches = [i for i, name in enumerate(self.asset.body_names) if name == self.body_name]
        if len(body_matches) != 1:
            raise ValueError(
                f"perturb_body_wrench.body_name must match exactly one body, got {self.body_name!r} "
                f"with matches={body_matches}."
            )
        self.body_id = int(body_matches[0])
        self.body_ids = torch.tensor([self.body_id], device=self.device, dtype=torch.long)

        self.total_duration_range_s = self._validate_nonnegative_range(
            "total_duration_range_s", total_duration_range_s
        )
        self.active_duration_range_s = self._validate_nonnegative_range(
            "active_duration_range_s", active_duration_range_s
        )
        self.force_magnitude_range_n = self._validate_nonnegative_range(
            "force_magnitude_range_n", force_magnitude_range_n
        )
        self.lever_arm_length_range_m = self._validate_nonnegative_range(
            "lever_arm_length_range_m", lever_arm_length_range_m
        )

        self.total_time_left_s = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.active_time_left_s = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.current_force_w = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.current_torque_w = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.current_lever_arm_w = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)

    def _validate_nonnegative_range(self, name: str, values: Tuple[float, float]):
        if len(values) != 2:
            raise ValueError(f"{name} must contain exactly two values, got {values}.")
        low = float(values[0])
        high = float(values[1])
        if low < 0.0:
            raise ValueError(f"{name} low must be >= 0, got {low}.")
        if high < low:
            raise ValueError(f"{name} must satisfy low <= high, got {values}.")
        return (low, high)

    def reset(self, env_ids: torch.Tensor):
        if not self.enabled:
            return
        if env_ids.numel() == 0:
            return
        self.total_time_left_s[env_ids] = 0.0
        self.active_time_left_s[env_ids] = 0.0
        self.current_force_w[env_ids] = 0.0
        self.current_torque_w[env_ids] = 0.0
        self.current_lever_arm_w[env_ids] = 0.0

    def update(self):
        if not self.enabled:
            return
        self.total_time_left_s.sub_(self.env.step_dt).clamp_min_(0.0)
        self.active_time_left_s.sub_(self.env.step_dt).clamp_min_(0.0)

        resample_ids = torch.nonzero(self.total_time_left_s <= 1e-6, as_tuple=False).squeeze(-1)
        if resample_ids.numel() == 0:
            return

        n = resample_ids.numel()
        total_time = sample_uniform((n,), *self.total_duration_range_s, device=self.device)
        active_time = sample_uniform((n,), *self.active_duration_range_s, device=self.device)
        active_time = torch.minimum(active_time, total_time)

        self.total_time_left_s[resample_ids] = total_time
        self.active_time_left_s[resample_ids] = active_time

        force_mag = sample_uniform((n,), *self.force_magnitude_range_n, device=self.device)
        lever_mag = sample_uniform((n,), *self.lever_arm_length_range_m, device=self.device)
        force_hat = _rand_unit_vectors((n, 3), device=self.device, dtype=torch.float32)
        lever_hat = _rand_unit_vectors((n, 3), device=self.device, dtype=torch.float32)

        force_w = force_hat * force_mag.unsqueeze(-1)
        lever_arm_w = lever_hat * lever_mag.unsqueeze(-1)
        torque_w = torch.cross(lever_arm_w, force_w, dim=-1)

        self.current_force_w[resample_ids] = force_w
        self.current_lever_arm_w[resample_ids] = lever_arm_w
        self.current_torque_w[resample_ids] = torque_w

    def step(self, substep):
        if not self.enabled:
            return
        active_mask = self.active_time_left_s > 0
        force_w = torch.zeros_like(self.current_force_w)
        torque_w = torch.zeros_like(self.current_torque_w)
        force_w[active_mask] = self.current_force_w[active_mask]
        torque_w[active_mask] = self.current_torque_w[active_mask]

        self.asset.write_external_wrench_to_sim(
            forces=force_w.unsqueeze(1),
            torques=torque_w.unsqueeze(1),
            body_ids=self.body_ids,
        )

    def debug_draw(self):
        if not self.enabled:
            return
        if not self.env._has_gui():
            return
        active_ids = torch.nonzero(self.active_time_left_s > 1e-6, as_tuple=False).squeeze(-1)
        if active_ids.numel() == 0:
            return

        body_pos_w = self.asset.data.body_link_pos_w[active_ids, self.body_id]
        application_pos_w = body_pos_w + self.current_lever_arm_w[active_ids]
        lever_vec_w = self.current_lever_arm_w[active_ids]
        force_vec_w = self.current_force_w[active_ids] * 0.02

        self.env.debug_draw.vector(
            body_pos_w,
            lever_vec_w,
            color=(0.8, 0.8, 0.2, 1.0),
        )
        self.env.debug_draw.vector(
            application_pos_w,
            force_vec_w,
            color=(0.2, 0.8, 1.0, 1.0),
        )
        self.env.debug_draw.point(
            application_pos_w,
            color=(1.0, 0.4, 0.1, 1.0),
            size=10.0,
        )


class window_cap_hand_load(Randomization):
    MODE_NO_FORCE = 0
    MODE_RAMP = 1
    MODE_HOLD = 2
    MODE_PREDROP = 3

    def __init__(
        self,
        env,
        label_path: str = "",
        constant_cap_kg: float | None = None,
        body_names: Tuple[str, str] = ("left_wrist_roll_link", "right_wrist_roll_link"),
        force_application_body_names: Tuple[str, str] | None = None,
        split_ratio_range: Tuple[float, float] = (0.35, 0.65),
        force_cone_half_angle_deg: float = 12.0,
        transition_duration_s: float = 1.0,
        predrop_duration_s: float = 1.0,
        max_load_kg: float = 30.0,
        cap_curriculum_progress_range: Tuple[float, float] = (0.0, 0.8),
        cap_curriculum_scale_range: Tuple[float, float] = (0.0, 1.0),
        cap_safety_scale: float = 1.0,
        inertial_force_scale_range: Tuple[float, float] = (0.0, 0.0),
        inertial_accel_tau_s: float = 0.08,
        inertial_accel_clip_mps2: float = 15.0,
        force_application_offset_radius_range_m: Tuple[float, float] = (0.0, 0.0),
        hand_force_fraction_range: Tuple[float, float] = (1.0, 1.0),
        window_no_load_ratio: float = 0.0,
        single_side_load_ratio: float = 0.0,
        missing_motion_policy: str = "error",
        enabled: bool = False,
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.enabled = bool(enabled)
        if not self.enabled:
            self.body_ids = torch.empty(0, dtype=torch.long, device=self.device)
            return
        if not label_path and constant_cap_kg is None:
            raise ValueError("window_cap_hand_load requires label_path or constant_cap_kg when enabled=True.")

        self.body_ids, self.body_names = self._resolve_ordered_body_ids(body_names)
        if force_application_body_names is None:
            self.force_application_body_ids = self.body_ids
            self.force_application_body_names = self.body_names
        else:
            self.force_application_body_ids, self.force_application_body_names = self._resolve_ordered_body_ids(
                force_application_body_names
            )
        self.transition_duration_s = max(float(transition_duration_s), 1.0e-6)
        self.predrop_duration_s = max(float(predrop_duration_s), 1.0e-6)
        self.max_load_kg = max(float(max_load_kg), 0.0)
        self.constant_cap_kg = None if constant_cap_kg is None else max(float(constant_cap_kg), 0.0)
        self.split_ratio_low = min(max(float(split_ratio_range[0]), 0.0), 1.0)
        self.split_ratio_high = min(max(float(split_ratio_range[1]), 0.0), 1.0)
        if self.split_ratio_high < self.split_ratio_low:
            raise ValueError("split_ratio_range must satisfy low <= high.")
        self.force_cone_half_angle_rad = float(np.deg2rad(max(float(force_cone_half_angle_deg), 0.0)))
        self.inertial_force_scale_range = self._validate_nonnegative_range(
            "inertial_force_scale_range", inertial_force_scale_range
        )
        self.inertial_accel_tau_s = max(float(inertial_accel_tau_s), 0.0)
        self.inertial_accel_clip_mps2 = max(float(inertial_accel_clip_mps2), 0.0)
        if self.inertial_accel_tau_s > 0.0:
            inertial_dt = max(float(self.env.physics_dt), 1.0e-6)
            self.inertial_accel_alpha = 1.0 - float(np.exp(-inertial_dt / self.inertial_accel_tau_s))
        else:
            self.inertial_accel_alpha = 1.0
        self.force_application_offset_radius_range_m = self._validate_nonnegative_range(
            "force_application_offset_radius_range_m", force_application_offset_radius_range_m
        )
        self.force_application_uses_body_com = force_application_body_names is None
        self.hand_force_fraction_range = self._validate_fraction_range(
            "hand_force_fraction_range", hand_force_fraction_range
        )
        self.window_no_load_ratio = self._validate_probability("window_no_load_ratio", window_no_load_ratio)
        self.single_side_load_ratio = self._validate_probability(
            "single_side_load_ratio", single_side_load_ratio
        )
        self.cap_curriculum_progress_range = tuple(float(x) for x in cap_curriculum_progress_range)
        self.cap_curriculum_scale_range = tuple(float(x) for x in cap_curriculum_scale_range)
        if len(self.cap_curriculum_progress_range) != 2 or len(self.cap_curriculum_scale_range) != 2:
            raise ValueError("cap curriculum ranges must contain exactly two values.")
        if self.cap_curriculum_progress_range[1] < self.cap_curriculum_progress_range[0]:
            raise ValueError("cap_curriculum_progress_range must satisfy start <= end.")
        self.cap_curriculum_scale = float(self.cap_curriculum_scale_range[0])

        command_manager = getattr(self.env, "command_manager", None)
        if command_manager is None or not hasattr(command_manager, "dataset"):
            raise RuntimeError("window_cap_hand_load requires MotionTrackingCommand with a dataset.")
        dataset = command_manager.dataset
        self.lookup = None
        if label_path:
            self.lookup = WindowLoadCapacityLookup.from_label_file(
                label_path,
                motion_source_paths=dataset.motion_source_paths,
                motion_labels=getattr(dataset, "motion_labels", None),
                motion_lengths=dataset.global_lengths.detach().cpu(),
                motion_fps=float(dataset.motion_fps),
                device=self.device,
                cap_safety_scale=float(cap_safety_scale),
                missing_motion_policy=str(missing_motion_policy),
            )

        self.current_total_load_kg = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.target_total_load_kg = torch.zeros_like(self.current_total_load_kg)
        self.controlled_load = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.controlled_current_load_kg = torch.zeros_like(self.current_total_load_kg)
        self.controlled_target_load_kg = torch.zeros_like(self.current_total_load_kg)
        self.controlled_ramp_sec = torch.ones_like(self.current_total_load_kg)
        self.transition_start_load_kg = torch.zeros_like(self.current_total_load_kg)
        self.transition_elapsed_s = torch.zeros_like(self.current_total_load_kg)
        self.transition_total_s = torch.full_like(self.current_total_load_kg, self.transition_duration_s)
        self.mode = torch.full((self.num_envs,), self.MODE_NO_FORCE, dtype=torch.long, device=self.device)
        self.reset_flat_bin_idx = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.active_flat_bin_idx = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.body_load_weights = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        self.body_load_weights[:, 0] = 0.5
        self.body_load_weights[:, 1] = 0.5
        self.force_dirs_w = torch.zeros((self.num_envs, 2, 3), dtype=torch.float32, device=self.device)
        self.force_dirs_w[:, :, 2] = -1.0
        self.forces_w = torch.zeros((self.num_envs, 2, 3), dtype=torch.float32, device=self.device)
        self.torques_w = torch.zeros_like(self.forces_w)
        self.root_forces_w = torch.zeros((self.num_envs, 1, 3), dtype=torch.float32, device=self.device)
        self.root_torques_w = torch.zeros_like(self.root_forces_w)
        self.current_inertial_force_scale = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.current_hand_force_fraction = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)
        self.force_application_offset_w = torch.zeros_like(self.forces_w)
        self.prev_body_linvel_w = torch.zeros_like(self.forces_w)
        self.body_accel_w = torch.zeros_like(self.forces_w)
        self.body_accel_initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.root_body_ids = torch.zeros(1, dtype=torch.long, device=self.device)
        self.body_ids_with_root = torch.cat([self.body_ids, self.root_body_ids], dim=0)
        all_env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self._sample_inertial_force_scale(all_env_ids)
        self._sample_hand_force_fraction(all_env_ids)
        self._sample_force_application_offsets(all_env_ids)
        self.step_schedule(0.0, None)

        if self.lookup is not None:
            logging.info(
                "window_cap_hand_load enabled: label=%s, matched_rows=%d, labeled_motions=%d/%d",
                self.lookup.label_source,
                self.lookup.matched_rows,
                self.lookup.labeled_motions,
                self.lookup.num_motions,
            )

    def _validate_nonnegative_range(self, name: str, values: Tuple[float, float]):
        if len(values) != 2:
            raise ValueError(f"{name} must contain exactly two values, got {values}.")
        low = float(values[0])
        high = float(values[1])
        if low < 0.0:
            raise ValueError(f"{name} low must be >= 0, got {low}.")
        if high < low:
            raise ValueError(f"{name} must satisfy low <= high, got {values}.")
        return (low, high)

    def _validate_fraction_range(self, name: str, values: Tuple[float, float]):
        low, high = self._validate_nonnegative_range(name, values)
        if high > 1.0:
            raise ValueError(f"{name} high must be <= 1, got {high}.")
        return (low, high)

    def _validate_probability(self, name: str, value: float):
        value = float(value)
        if value < 0.0 or value > 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}.")
        return value

    def _resolve_ordered_body_ids(self, body_names) -> tuple[torch.Tensor, list[str]]:
        if isinstance(body_names, str):
            body_names = [body_names]
        if len(body_names) != 2:
            raise ValueError("window_cap_hand_load.body_names must resolve left and right body names.")
        ids = []
        names = []
        for pattern in body_names:
            matched_ids, matched_names = self.asset.find_bodies([pattern])
            if len(matched_ids) != 1:
                raise ValueError(
                    f"window_cap_hand_load body pattern {pattern!r} must match exactly one body, "
                    f"got {matched_names}."
                )
            ids.append(int(matched_ids[0]))
            names.append(str(matched_names[0]))
        return torch.as_tensor(ids, dtype=torch.long, device=self.device), names

    def _command_motion_ids_frames(self) -> tuple[torch.Tensor, torch.Tensor]:
        dataset = self.env.command_manager.dataset
        return dataset._motion_ids_A.long(), self.env.command_manager.t.long()

    def _lookup_current(self, env_ids: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        motion_ids, frames = self._command_motion_ids_frames()
        if env_ids is not None:
            motion_ids = motion_ids[env_ids]
            frames = frames[env_ids]
        if self.lookup is None:
            lengths = self.env.command_manager.lengths.long()
            if env_ids is not None:
                lengths = lengths[env_ids]
            valid = torch.ones_like(frames, dtype=torch.bool)
            zeros = torch.zeros_like(frames)
            return {
                "valid": valid,
                "flat_idx": motion_ids,
                "bin_idx": zeros,
                "start": zeros,
                "end": lengths,
                "cap_kg": torch.full_like(frames, self.constant_cap_kg, dtype=torch.float32),
                "next_valid": torch.zeros_like(valid),
                "next_start": zeros,
                "next_cap_kg": torch.zeros_like(frames, dtype=torch.float32),
            }
        return self.lookup.lookup(motion_ids, frames)

    def set_controlled_load(
        self,
        env_ids: torch.Tensor,
        current_load_kg: torch.Tensor,
        target_load_kg: torch.Tensor,
        ramp_sec: float,
    ) -> None:
        self.controlled_load[env_ids] = True
        self.controlled_current_load_kg[env_ids] = current_load_kg
        self.controlled_target_load_kg[env_ids] = target_load_kg
        self.controlled_ramp_sec[env_ids] = ramp_sec
        self._apply_controlled_load(env_ids)

    def _apply_controlled_load(self, env_ids: torch.Tensor) -> None:
        self.body_load_weights[env_ids] = 0.5
        self.force_dirs_w[env_ids] = torch.tensor((0.0, 0.0, -1.0), device=self.device)
        self.current_total_load_kg[env_ids] = self.controlled_current_load_kg[env_ids]
        self.active_flat_bin_idx[env_ids] = self._lookup_current(env_ids)["flat_idx"]
        self._start_transition(
            env_ids,
            self.controlled_target_load_kg[env_ids],
            self.controlled_ramp_sec[env_ids],
            self.MODE_RAMP,
        )

    def step_schedule(self, progress: float, iters: int | None = None):
        if not self.enabled:
            return
        if self.env.student_train:
            self.cap_curriculum_scale = float(self.cap_curriculum_scale_range[1])
            return
        start, end = self.cap_curriculum_progress_range
        start_scale, end_scale = self.cap_curriculum_scale_range
        p = float(min(max(progress, 0.0), 1.0))
        if p <= start:
            scale = start_scale
        elif p >= end or abs(end - start) <= 1.0e-9:
            scale = end_scale
        else:
            alpha = (p - start) / (end - start)
            scale = start_scale + alpha * (end_scale - start_scale)
        self.cap_curriculum_scale = max(float(scale), 0.0)

    def _effective_cap_kg(self, cap_kg: torch.Tensor) -> torch.Tensor:
        return cap_kg.clamp(0.0, self.max_load_kg) * float(self.cap_curriculum_scale)

    def _sample_body_load_weights(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return
        left_ratio = sample_uniform(
            (env_ids.numel(),),
            self.split_ratio_low,
            self.split_ratio_high,
            device=self.device,
        )
        weights = torch.empty((env_ids.numel(), 2), dtype=torch.float32, device=self.device)
        weights[:, 0] = left_ratio
        weights[:, 1] = 1.0 - left_ratio

        if self.single_side_load_ratio > 0.0:
            make_single_side = torch.rand((env_ids.numel(),), device=self.device) < self.single_side_load_ratio
            single_side_local_ids = torch.nonzero(make_single_side, as_tuple=False).squeeze(-1)
            if single_side_local_ids.numel() > 0:
                left_side = torch.rand(single_side_local_ids.shape, device=self.device) < 0.5
                weights[single_side_local_ids] = 0.0
                weights[single_side_local_ids, 0] = left_side.float()
                weights[single_side_local_ids, 1] = (~left_side).float()

        self.body_load_weights[env_ids] = weights

    def _sample_window_target_loads(self, env_ids: torch.Tensor, caps_kg: torch.Tensor) -> torch.Tensor:
        target = torch.rand_like(caps_kg) * caps_kg
        if self.window_no_load_ratio > 0.0:
            no_load = torch.rand(target.shape, device=self.device) < self.window_no_load_ratio
            target = torch.where(no_load, torch.zeros_like(target), target)
        return target

    def reset(self, env_ids: torch.Tensor):
        if not self.enabled:
            return
        if env_ids.numel() == 0:
            return
        env_ids = env_ids.long()
        lookup = self._lookup_current(env_ids)
        self.reset_flat_bin_idx[env_ids] = lookup["flat_idx"]
        self.active_flat_bin_idx[env_ids] = lookup["flat_idx"]

        self._sample_body_load_weights(env_ids)
        self.force_dirs_w[env_ids] = self._sample_force_dirs(env_ids.numel())
        self._sample_inertial_force_scale(env_ids)
        self._sample_hand_force_fraction(env_ids)
        self._sample_force_application_offsets(env_ids)
        self.current_total_load_kg[env_ids] = 0.0
        self.target_total_load_kg[env_ids] = 0.0
        self.transition_start_load_kg[env_ids] = 0.0
        self.transition_elapsed_s[env_ids] = 0.0
        self.transition_total_s[env_ids] = self.transition_duration_s
        self.mode[env_ids] = self.MODE_NO_FORCE
        self.forces_w[env_ids] = 0.0
        self.torques_w[env_ids] = 0.0
        self.root_forces_w[env_ids] = 0.0
        self.root_torques_w[env_ids] = 0.0
        self.body_accel_w[env_ids] = 0.0
        self.prev_body_linvel_w[env_ids] = self._current_force_application_linvel_w(env_ids)
        self.body_accel_initialized[env_ids] = True
        valid = lookup["valid"]
        if valid.any():
            valid_ids = env_ids[valid]
            caps = self._effective_cap_kg(lookup["cap_kg"][valid])
            target = self._sample_window_target_loads(valid_ids, caps)
            self._start_transition(valid_ids, target, self.transition_duration_s, self.MODE_RAMP)
        controlled_ids = env_ids[self.controlled_load[env_ids]]
        if controlled_ids.numel() > 0:
            self._apply_controlled_load(controlled_ids)

    def update(self):
        if not self.enabled:
            return
        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        lookup = self._lookup_current()
        valid = lookup["valid"]
        current_flat_idx = lookup["flat_idx"]

        left_labeled_bin = (~valid) & (self.active_flat_bin_idx >= 0)
        if left_labeled_bin.any():
            leave_ids = env_ids[left_labeled_bin]
            self._start_transition(leave_ids, torch.zeros_like(self.target_total_load_kg[leave_ids]), self.transition_duration_s, self.MODE_RAMP)
            self.active_flat_bin_idx[leave_ids] = -1

        new_bin = valid & (current_flat_idx != self.active_flat_bin_idx)
        if new_bin.any():
            new_ids = env_ids[new_bin]
            caps = self._effective_cap_kg(lookup["cap_kg"][new_bin])
            target = self._sample_window_target_loads(new_ids, caps)
            self.active_flat_bin_idx[new_ids] = current_flat_idx[new_bin]
            self._start_transition(new_ids, target, self.transition_duration_s, self.MODE_RAMP)

        next_cap_kg = self._effective_cap_kg(lookup["next_cap_kg"])
        predrop = valid & lookup["next_valid"] & (self.target_total_load_kg > next_cap_kg)
        remaining_s = (lookup["end"] - self.env.command_manager.t.long()).float().clamp_min(0.0) * float(self.env.step_dt)
        predrop &= remaining_s <= self.predrop_duration_s
        predrop &= remaining_s > 1.0e-6
        predrop &= self.mode != self.MODE_PREDROP
        if predrop.any():
            drop_ids = env_ids[predrop]
            target = next_cap_kg[predrop]
            duration = remaining_s[predrop].clamp_min(float(self.env.step_dt))
            self._start_transition(drop_ids, target, duration, self.MODE_PREDROP)

        self._advance_transitions()

    def _start_transition(
        self,
        env_ids: torch.Tensor,
        target_load_kg: torch.Tensor,
        duration_s: float | torch.Tensor,
        mode: int,
    ):
        if env_ids.numel() == 0:
            return
        self.transition_start_load_kg[env_ids] = self.current_total_load_kg[env_ids]
        self.target_total_load_kg[env_ids] = target_load_kg.float().clamp(0.0, self.max_load_kg)
        self.transition_elapsed_s[env_ids] = 0.0
        if isinstance(duration_s, torch.Tensor):
            self.transition_total_s[env_ids] = duration_s.float().clamp_min(1.0e-6)
        else:
            self.transition_total_s[env_ids] = max(float(duration_s), 1.0e-6)
        self.mode[env_ids] = int(mode)

    def _advance_transitions(self):
        transition_mask = (self.mode == self.MODE_RAMP) | (self.mode == self.MODE_PREDROP)
        if transition_mask.any():
            ids = torch.nonzero(transition_mask, as_tuple=False).squeeze(-1)
            self.transition_elapsed_s[ids] += float(self.env.step_dt)
            alpha = (self.transition_elapsed_s[ids] / self.transition_total_s[ids].clamp_min(1.0e-6)).clamp(0.0, 1.0)
            start = self.transition_start_load_kg[ids]
            target = self.target_total_load_kg[ids]
            self.current_total_load_kg[ids] = start + (target - start) * alpha
            done = alpha >= 1.0 - 1.0e-6
            if done.any():
                done_ids = ids[done]
                self.mode[done_ids] = torch.where(
                    self.target_total_load_kg[done_ids] <= 1.0e-6,
                    torch.full_like(done_ids, self.MODE_NO_FORCE),
                    torch.full_like(done_ids, self.MODE_HOLD),
                )

        hold_mask = self.mode == self.MODE_HOLD
        if hold_mask.any():
            self.current_total_load_kg[hold_mask] = self.target_total_load_kg[hold_mask]

        no_force_mask = self.mode == self.MODE_NO_FORCE
        if no_force_mask.any():
            self.current_total_load_kg[no_force_mask] = 0.0

    def _sample_force_dirs(self, n: int) -> torch.Tensor:
        dirs = torch.zeros((n, 2, 3), dtype=torch.float32, device=self.device)
        if self.force_cone_half_angle_rad <= 1.0e-6:
            dirs[:, :, 2] = -1.0
            return dirs
        cos_low = float(np.cos(self.force_cone_half_angle_rad))
        cos_theta = sample_uniform((n, 2), cos_low, 1.0, device=self.device)
        sin_theta = torch.sqrt((1.0 - cos_theta.square()).clamp_min(0.0))
        phi = sample_uniform((n, 2), 0.0, float(2.0 * np.pi), device=self.device)
        dirs[:, :, 0] = sin_theta * torch.cos(phi)
        dirs[:, :, 1] = sin_theta * torch.sin(phi)
        dirs[:, :, 2] = -cos_theta
        return dirs

    def _sample_inertial_force_scale(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return
        low, high = self.inertial_force_scale_range
        self.current_inertial_force_scale[env_ids] = sample_uniform(
            (env_ids.numel(),),
            low,
            high,
            device=self.device,
        )

    def _sample_hand_force_fraction(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return
        low, high = self.hand_force_fraction_range
        self.current_hand_force_fraction[env_ids] = sample_uniform(
            (env_ids.numel(),),
            low,
            high,
            device=self.device,
        )

    def _sample_force_application_offsets(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return
        low, high = self.force_application_offset_radius_range_m
        n = env_ids.numel()
        radius = sample_uniform((n, 2, 1), low, high, device=self.device)
        direction_w = _rand_unit_vectors((n, 2, 3), device=self.device, dtype=torch.float32)
        self.force_application_offset_w[env_ids] = direction_w * radius

    def _current_force_application_linvel_w(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        if self.force_application_uses_body_com:
            body_linvel_w = self.asset.data.body_com_lin_vel_w[:, self.body_ids]
        else:
            body_linvel_w = self.asset.data.body_link_lin_vel_w[:, self.force_application_body_ids]
        if env_ids is None:
            return body_linvel_w
        return body_linvel_w[env_ids]

    def _current_force_application_pos_w(self) -> torch.Tensor:
        if self.force_application_uses_body_com:
            body_pos_w = self.asset.data.body_com_pos_w[:, self.body_ids]
        else:
            body_pos_w = self.asset.data.body_link_pos_w[:, self.force_application_body_ids]
        return body_pos_w + self.force_application_offset_w

    def _update_body_accel(self) -> None:
        body_linvel_w = self._current_force_application_linvel_w()
        dt = max(float(self.env.physics_dt), 1.0e-6)
        raw_accel = (body_linvel_w - self.prev_body_linvel_w) / dt

        not_initialized = ~self.body_accel_initialized
        if not_initialized.any():
            raw_accel[not_initialized] = 0.0
            self.body_accel_initialized[not_initialized] = True

        if self.inertial_accel_clip_mps2 > 0.0:
            accel_norm = raw_accel.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
            accel_scale = (self.inertial_accel_clip_mps2 / accel_norm).clamp_max(1.0)
            raw_accel = raw_accel * accel_scale

        if self.inertial_accel_alpha >= 1.0:
            self.body_accel_w.copy_(raw_accel)
        else:
            self.body_accel_w.lerp_(raw_accel, self.inertial_accel_alpha)
        self.prev_body_linvel_w.copy_(body_linvel_w)

    def _current_body_force(self) -> torch.Tensor:
        body_load_kg = self.current_total_load_kg.unsqueeze(-1).unsqueeze(-1) * self.body_load_weights.unsqueeze(-1)
        force_w = (
            body_load_kg
            * 9.81
            * self.force_dirs_w
        )
        force_w = force_w - body_load_kg * self.current_inertial_force_scale.view(-1, 1, 1) * self.body_accel_w
        return force_w

    def _recompute_forces(self):
        sampled_forces_w = self._current_body_force()
        hand_fraction = self.current_hand_force_fraction.view(-1, 1, 1)
        self.forces_w[:] = sampled_forces_w * hand_fraction
        application_pos_w = self._current_force_application_pos_w()
        force_body_com_w = self.asset.data.body_com_pos_w[:, self.body_ids]
        self.torques_w[:] = torch.cross(application_pos_w - force_body_com_w, self.forces_w, dim=-1)
        root_forces_by_body_w = sampled_forces_w - self.forces_w
        r_root_to_application_w = application_pos_w - self.asset.data.root_com_pos_w.unsqueeze(1)
        self.root_forces_w[:] = root_forces_by_body_w.sum(dim=1, keepdim=True)
        self.root_torques_w[:] = torch.cross(
            r_root_to_application_w, root_forces_by_body_w, dim=-1
        ).sum(dim=1, keepdim=True)

    def step(self, substep: int):
        if not self.enabled:
            return
        self._update_body_accel()
        self._recompute_forces()
        forces_w = torch.cat([self.forces_w, self.root_forces_w], dim=1)
        torques_w = torch.cat([self.torques_w, self.root_torques_w], dim=1)
        self.asset.write_external_wrench_to_sim(
            forces=forces_w,
            torques=torques_w,
            body_ids=self.body_ids_with_root,
        )

    def debug_draw(self):
        if not self.enabled:
            return
        if not self.env._has_gui():
            return
        active_ids = torch.nonzero(self.current_total_load_kg > 1.0e-6, as_tuple=False).squeeze(-1)
        if active_ids.numel() == 0:
            return
        body_pos_w = self._current_force_application_pos_w()[active_ids]
        self.env.debug_draw.vector(
            body_pos_w.reshape(-1, 3),
            (self.forces_w[active_ids] * 0.02).reshape(-1, 3),
            color=(0.2, 0.8, 1.0, 1.0),
        )

    def has_observation(self) -> bool:
        return self.enabled

    def observe(self, **kwargs) -> torch.Tensor:
        force_denom = max(self.max_load_kg, 1.0e-6) * 9.81
        return self._current_body_force().reshape(self.num_envs, -1) / force_denom

    def observe_sym(self, **kwargs):
        return sym_utils.cartesian_space_symmetry(self.asset, self.body_names, sign=(1, -1, 1))


class perturb_root_vel(Randomization):
    def __init__(
        self,
        env,
        min_s: float,
        max_s: float,
        x: Tuple[float, float] = (0.0, 0.0),
        y: Tuple[float, float] = (0.0, 0.0),
        z: Tuple[float, float] = (0.0, 0.0),
        roll: Tuple[float, float] = (0.0, 0.0),
        pitch: Tuple[float, float] = (0.0, 0.0),
        yaw: Tuple[float, float] = (0.0, 0.0),
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]

        self.min_s = float(min_s)
        self.max_s = float(max_s)
        if self.min_s < 0.0:
            raise ValueError(f"min_s must be >= 0, got {self.min_s}")
        if self.max_s < self.min_s:
            raise ValueError(f"max_s must be >= min_s, got max_s={self.max_s}, min_s={self.min_s}")

        self._range_names = ("x", "y", "z", "roll", "pitch", "yaw")
        ranges = (x, y, z, roll, pitch, yaw)
        lows = []
        highs = []
        for name, axis_range in zip(self._range_names, ranges, strict=True):
            if len(axis_range) != 2:
                raise ValueError(f"{name} range must have exactly 2 values, got {axis_range}")
            low = float(axis_range[0])
            high = float(axis_range[1])
            if high < low:
                raise ValueError(f"{name} range must satisfy low <= high, got {axis_range}")
            lows.append(low)
            highs.append(high)
        self.low = torch.tensor(lows, dtype=torch.float32, device=self.device)
        self.high = torch.tensor(highs, dtype=torch.float32, device=self.device)

        self.time_left_s = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

    def _sample_interval(self, n: int):
        if n <= 0:
            return torch.empty(0, dtype=torch.float32, device=self.device)
        return sample_uniform((n,), self.min_s, self.max_s, device=self.device)

    def _sample_delta_vel(self, n: int):
        if n <= 0:
            return torch.empty((0, 6), dtype=torch.float32, device=self.device)
        return uniform(
            self.low.unsqueeze(0).expand(n, -1),
            self.high.unsqueeze(0).expand(n, -1),
        )

    def reset(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return
        self.time_left_s[env_ids] = self._sample_interval(len(env_ids))

    def update(self):
        self.time_left_s.sub_(self.env.step_dt)
        trigger_ids = torch.nonzero(self.time_left_s <= 1e-6, as_tuple=False).squeeze(-1)
        if trigger_ids.numel() == 0:
            return

        delta_vel = self._sample_delta_vel(trigger_ids.numel())
        lin_vel = self.asset.data.root_link_lin_vel_w[trigger_ids] + delta_vel[:, :3]
        ang_vel = self.asset.data.root_link_ang_vel_w[trigger_ids] + delta_vel[:, 3:]
        root_vel = torch.cat((lin_vel, ang_vel), dim=-1)

        self.asset.write_root_link_velocity_to_sim(root_vel, env_ids=trigger_ids)

        self.time_left_s[trigger_ids] = self._sample_interval(trigger_ids.numel())

class perturb_gravity(Randomization):
    def __init__(self, env, mean: Tuple[float, float, float] = (0.0, 0.0, -9.81), std: float = 0.0):
        super().__init__(env)
        if len(mean) != 3:
            raise ValueError(f"mean must have 3 elements (x, y, z), got {mean}")
        self.mean = torch.tensor(mean, dtype=torch.float32, device=self.device)
        self.std = float(std)
        if self.std < 0.0:
            raise ValueError(f"std must be >= 0, got {self.std}")
        self.asset: Articulation = self.env.scene["robot"]

    def _ensure_per_env_gravity_storage(self):
        gravity = self.env.sim.model.opt.gravity
        has_per_env_storage = gravity.shape[0] == self.num_envs and gravity.stride(0) != 0
        if has_per_env_storage:
            return

        init_gravity = self.mean.unsqueeze(0).expand(self.num_envs, -1).contiguous()
        with wp.ScopedDevice(self.env.sim.wp_device):
            self.env.sim.wp_model.opt.gravity = wp.from_torch(init_gravity, dtype=wp.vec3)
        self.env.sim.model.clear_cache()
        self.env.sim.create_graph()

    def _sample_gravity(self, n_envs: int):
        gravity = self.mean.unsqueeze(0).expand(n_envs, -1).clone()
        if self.std > 0.0:
            gravity = add_spherical_noise(gravity, self.std)
        return gravity

    def startup(self):
        self._ensure_per_env_gravity_storage()
        gravity = self._sample_gravity(self.num_envs)
        self.env.sim.model.opt.gravity[:] = gravity
        self._gravity = gravity

        assert torch.allclose(self.env.sim.model.opt.gravity, gravity)

    def reset(self, env_ids: torch.Tensor):
        pass

    def debug_draw(self):
        if not self.env._has_gui():
            return

        gravity = self._gravity.clone()
        if gravity.shape[0] == 1:
            gravity = gravity.expand(self.num_envs, -1)
        gravity_dir = gravity / gravity.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        starts = self.asset.data.root_link_pos_w + torch.tensor([0.0, 0.0, 0.35], device=self.device)
        vectors = gravity_dir * 0.25
        self.env.debug_draw.vector(starts, vectors, color=(0.2, 0.9, 0.2, 1.0))

    def has_observation(self) -> bool:
        return True

    def observe(self, **kwargs) -> torch.Tensor:
        gravity = self._gravity
        return gravity

    def observe_sym(self, **kwargs):
        return sym_utils.SymmetryTransform(torch.arange(3), torch.tensor([1.0, -1.0, 1.0]))
