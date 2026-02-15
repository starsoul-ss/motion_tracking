import torch
import numpy as np
import logging
from typing import Union, TYPE_CHECKING, Dict, Tuple

import mjlab.utils.lab_api.string as string_utils


if TYPE_CHECKING:
    from mjlab.entity import Entity as Articulation
    from active_adaptation.envs.base import _Env


class Randomization:
    def __init__(self, env):
        self.env: _Env = env

    @property
    def num_envs(self):
        return self.env.num_envs
    
    @property
    def device(self):
        return self.env.device
    
    def startup(self):
        pass
    
    def reset(self, env_ids: torch.Tensor):
        pass
    
    def step(self, substep):
        pass

    def update(self):
        pass

    def debug_draw(self):
        pass


RangeType = Tuple[float, float]
NestedRangeType = Union[RangeType, Dict[str, RangeType]]

class motor_params_implicit(Randomization):
    def __init__(
        self,
        env,
        stiffness_range,
        damping_range,
        armature_range,
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]

        # 存下区间字典
        self.stiffness_range = dict(stiffness_range)
        self.damping_range   = dict(damping_range)
        self.armature_range  = dict(armature_range)
        self.model = self.env.sim.model
        # ------- stiffness / damping via actuator gains (mjlab randomize_pd_gains style) -------
        kp_ids, _, kp_ranges = string_utils.resolve_matching_names_values(
            self.stiffness_range, self.asset.actuator_names
        )
        kd_ids, _, kd_ranges = string_utils.resolve_matching_names_values(
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

        self.kp_low = kp_low
        self.kp_scale = kp_high - kp_low
        self.kd_low = kd_low
        self.kd_scale = kd_high - kd_low

        # ------- armature (改为相对值) -------
        arm_ids, _, arm_ranges = string_utils.resolve_matching_names_values(
            self.armature_range, self.asset.joint_names
        )
        self.arm_ids = torch.tensor(arm_ids, device=self.device, dtype=torch.long)
        self.arm_dof_ids = self.asset.indexing.joint_v_adr[self.arm_ids]
        default_armature = self.env.sim.get_default_field("dof_armature")
        self.arm_def = default_armature[self.arm_dof_ids]

        arm_low, arm_high = torch.tensor(arm_ranges, device=self.device).unbind(1)
        self.arm_low = arm_low
        self.arm_scale = arm_high - arm_low

    def _rand_u(self, n_env: int, k: int):
        return torch.rand(n_env, k, device=self.device)

    # ----------------------------------------------------------
    def reset(self, env_ids):
        n_env = len(env_ids)

        # stiffness (kp)
        kp_samples = self._rand_u(n_env, len(self.kp_ctrl_ids))
        kp_samples = kp_samples * self.kp_scale + self.kp_low
        self.model.actuator_gainprm[env_ids[:, None], self.kp_ctrl_ids[None, :], 0] = (
            self.kp_gain_def * kp_samples
        )
        self.model.actuator_biasprm[env_ids[:, None], self.kp_ctrl_ids[None, :], 1] = (
            self.kp_bias_def * kp_samples
        )

        # damping (kd)
        kd_samples = self._rand_u(n_env, len(self.kd_ctrl_ids))
        kd_samples = kd_samples * self.kd_scale + self.kd_low
        self.model.actuator_biasprm[env_ids[:, None], self.kd_ctrl_ids[None, :], 2] = (
            self.kd_bias_def * kd_samples
        )

        # armature
        arma = self._rand_u(n_env, len(self.arm_ids))
        arma = arma * self.arm_scale + self.arm_low
        self.model.dof_armature[env_ids[:, None], self.arm_dof_ids[None, :]] = (
            self.arm_def * arma
        )


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
        geom_friction = model.geom_friction
        geom_friction[:, self.geom_global_ids, 0] = sf
        model.geom_solref[:, self.geom_global_ids, 0] = tc
        model.geom_solref[:, self.geom_global_ids, 1] = dr


class perturb_body_mass(Randomization):
    def __init__(
        self, env, **perturb_ranges: Tuple[float, float]
    ):
        super().__init__(env)
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
        low = self.mass_ranges[:, 0]
        high = self.mass_ranges[:, 1]
        rand = torch.rand(
            self.num_envs, self.local_body_ids.numel(), device=self.device
        )
        scale = low + (high - low) * rand

        model = self.env.sim.model
        new_mass = self._default_mass * scale
        new_inertia = self._default_inertia * scale.unsqueeze(-1)
        model.body_mass[:, self.global_body_ids] = new_mass
        model.body_inertia[:, self.global_body_ids] = new_inertia


class perturb_body_com(Randomization):
    def __init__(self, env, body_names = ".*", com_range=(-0.05, 0.05)):
        super().__init__(env)
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
        offsets = torch.rand(self.num_envs, num_bodies, 3, device=self.device)
        offsets = low + (high - low) * offsets

        model = self.env.sim.model
        new_ipos = self._default_body_ipos + offsets
        model.body_ipos[:, self.global_body_ids] = new_ipos


class random_joint_offset(Randomization):
    def __init__(self, env, **offset_range: Tuple[float, float]):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids, _, self.offset_range = string_utils.resolve_matching_names_values(dict(offset_range), self.asset.joint_names)
        
        self.joint_ids = torch.tensor(self.joint_ids, device=self.device)
        self.offset_range = torch.tensor(self.offset_range, device=self.device)

        self.action_manager = self.env.action_manager

    def reset(self, env_ids: torch.Tensor):
        offset = uniform(self.offset_range[:, 0], self.offset_range[:, 1])
        self.action_manager.offset[env_ids.unsqueeze(1), self.joint_ids] = offset

def clamp_norm(x: torch.Tensor, min: float = 0.0, max: float = torch.inf):
    x_norm = x.norm(dim=-1, keepdim=True).clamp(1e-6)
    x = torch.where(x_norm < min, x / x_norm * min, x)
    x = torch.where(x_norm > max, x / x_norm * max, x)
    return x


def random_scale(x: torch.Tensor, low: float, high: float, homogeneous: bool=False):
    if homogeneous:
        u = torch.rand(*x.shape[:1], 1, device=x.device)
    else:
        u = torch.rand_like(x)
    return x * (u * (high - low) + low), u

def random_shift(x: torch.Tensor, low: float, high: float):
    return x + x * (torch.rand_like(x) * (high - low) + low)

def sample_uniform(size, low: float, high: float, device: torch.device = "cpu"):
    return torch.rand(size, device=device) * (high - low) + low

def sample_log_uniform(size, low: float, high: float, device: torch.device = "cpu"):
    low_t = torch.tensor(low, device=device, dtype=torch.float32)
    high_t = torch.tensor(high, device=device, dtype=torch.float32)
    return log_uniform(low_t, high_t).expand(size) if size == () else log_uniform(
        low_t.expand(size), high_t.expand(size)
    )

def uniform(low: torch.Tensor, high: torch.Tensor):
    r = torch.rand_like(low)
    return low + r * (high - low)

def uniform_like(x: torch.Tensor, low: torch.Tensor, high: torch.Tensor):
    r = torch.rand_like(x)
    return low + r * (high - low)

def log_uniform(low: torch.Tensor, high: torch.Tensor):
    return uniform(low.log(), high.log()).exp()

def angle_mix(a: torch.Tensor, b: torch.Tensor, weight: float=0.1):
    d = a - b
    d[d > torch.pi] -= 2 * torch.pi
    d[d < -torch.pi] += 2 * torch.pi
    return a - d * weight
