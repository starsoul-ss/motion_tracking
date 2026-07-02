from math import inf
import torch
from typing import TYPE_CHECKING, List, Tuple

import mjlab.utils.lab_api.string as string_utils
from mjlab.utils.lab_api.string import resolve_matching_names
from active_adaptation.utils.math import quat_apply, quat_apply_inverse, yaw_quat, normalize
from active_adaptation.envs.mdp.contact_utils import resolve_contact_indices
from .base import Reward

if TYPE_CHECKING:
    from mjlab.sensor import ContactSensor
    from mjlab.entity import Entity as Articulation


class _ContactMajorityCache:
    """Shared per-env-step contact state using substep majority voting."""

    def __init__(self, env, contact_sensor):
        self.env = env
        self.contact_sensor = contact_sensor
        found = self.contact_sensor.data.found
        if found is None:
            raise RuntimeError(
                "Contact sensor must include 'found' field to use majority contact cache."
            )
        self.decimation = max(int(self.env.decimation), 1)
        self.num_bodies = int(found.shape[1])
        self.substep_found = torch.zeros(
            (self.env.num_envs, self.num_bodies, self.decimation),
            dtype=torch.bool,
            device=self.env.device,
        )
        self.current_contact = torch.zeros(
            (self.env.num_envs, self.num_bodies), dtype=torch.bool, device=self.env.device
        )
        self.first_contact = torch.zeros_like(self.current_contact)
        self.first_air = torch.zeros_like(self.current_contact)
        self._last_post_stamp = (-1, -1)
        self._last_update_stamp = -1

    def reset(self, env_ids: torch.Tensor):
        self.substep_found[env_ids] = False
        self.current_contact[env_ids] = False
        self.first_contact[env_ids] = False
        self.first_air[env_ids] = False

    def post_step(self, substep: int):
        stamp = (int(self.env.timestamp), int(substep))
        if stamp == self._last_post_stamp:
            return
        found = self.contact_sensor.data.found
        if found is None:
            raise RuntimeError(
                "Contact sensor must include 'found' field to use majority contact cache."
            )
        self.substep_found[:, :, substep] = found > 0
        self._last_post_stamp = stamp

    def update(self):
        stamp = int(self.env.timestamp)
        if stamp == self._last_update_stamp:
            return
        votes = self.substep_found.sum(dim=-1)
        contact_majority = votes >= (self.decimation // 2)
        prev_contact = self.current_contact
        self.first_contact[:] = (~prev_contact) & contact_majority
        self.first_air[:] = prev_contact & (~contact_majority)
        self.current_contact[:] = contact_majority
        self.substep_found.zero_()
        self._last_update_stamp = stamp

    def current_for(self, body_ids: torch.Tensor):
        return self.current_contact[:, body_ids]

    def first_contact_for(self, body_ids: torch.Tensor):
        return self.first_contact[:, body_ids]

    def first_air_for(self, body_ids: torch.Tensor):
        return self.first_air[:, body_ids]


def _get_contact_majority_cache(env, contact_sensor):
    cache = getattr(env, "_contact_majority_cache", None)
    if cache is None:
        cache = _ContactMajorityCache(env, contact_sensor)
        env._contact_majority_cache = cache
    elif cache.contact_sensor is not contact_sensor:
        raise RuntimeError("Multiple contact sensors are not supported by shared contact cache.")
    return cache


def _feet_sides(names: list[str], *, context: str) -> list[str]:
    sides = ["left" if "left" in name.lower() else "right" if "right" in name.lower() else None for name in names]
    if len(names) != 2 or sorted(sides) != ["left", "right"]:
        raise ValueError(f"{context} expects exactly one left foot and one right foot, got {names}")
    return sides

def _validate_feet_order(env, resolved_names: list[str], *, context: str):
    command_manager = getattr(env, "command_manager", None)
    cmd_feet_names = list(getattr(command_manager, "feet_names_asset", ()))
    if not cmd_feet_names:
        raise RuntimeError("feet reward requires command_manager.feet_names_asset for order validation.")
    if _feet_sides(resolved_names, context=context) != _feet_sides(cmd_feet_names, context="command feet"):
        raise ValueError(f"{context} resolved feet order {resolved_names} does not match command feet order {cmd_feet_names}")


class survival(Reward):
    def compute(self) -> torch.Tensor:
        return torch.ones(self.num_envs, 1, device=self.device)


class joint_torques_l2(Reward):
    def __init__(
        self, env, weight: float, enabled: bool = True, joint_names: str = ".*"
    ):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids, self.joint_names = resolve_matching_names(joint_names, self.asset.joint_names)

        actuator_names = list(self.asset.actuator_names)
        name_to_act = {n: i for i, n in enumerate(actuator_names)}
        act_idx = []
        for name in self.joint_names:
            if name not in name_to_act:
                raise RuntimeError(f"Actuator for joint '{name}' not found.")
            act_idx.append(name_to_act[name])
        self.act_idx = torch.tensor(act_idx, device=self.device, dtype=torch.long)

    def compute(self) -> torch.Tensor:
        return (
            -self.asset.data.actuator_force[:, self.act_idx]
            .square()
            .sum(1, keepdim=True)
        )


class impact_force_l2(Reward):
    def __init__(
        self,
        env,
        body_names,
        weight: float = 1.0,
        enabled: bool = True,
    ):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]
        self.body_ids, self.articulation_body_ids, self.body_names = resolve_contact_indices(
            self.contact_sensor, self.asset, body_names
        )
        self.articulation_body_ids = torch.tensor(self.articulation_body_ids, device=self.device, dtype=torch.long)
        self.last_contact = torch.zeros(
            self.num_envs, len(self.body_ids), device=self.device, dtype=bool
        )
        self.prev_down_vel = torch.zeros(
            self.num_envs, len(self.body_ids), device=self.device
        )
        self.down_vel = torch.zeros(
            self.num_envs, len(self.body_ids), device=self.device
        )

        print(f"Penalizing impact forces on {self.body_names}.")

    def reset(self, env_ids: torch.Tensor):
        self.last_contact[env_ids] = False
        self.prev_down_vel[env_ids] = 0.0
        self.down_vel[env_ids] = 0.0

    def update(self):
        vel_z = self.asset.data.body_com_lin_vel_w[:, self.articulation_body_ids, 2]
        down_vel = (-vel_z).clamp_min(0.0)
        self.prev_down_vel.copy_(self.down_vel)
        self.down_vel[:] = torch.where(
            self.contact_sensor.data.current_contact_time[:, self.body_ids] > 0.0,
            self.down_vel,
            down_vel,
        )

    def compute(self) -> torch.Tensor:
        current_contact = self.contact_sensor.data.current_contact_time[:, self.body_ids] > self.env.physics_dt
        first_contact = (~self.last_contact) & current_contact
        self.last_contact[:] = current_contact
        impact = self.prev_down_vel.square() * first_contact
        return -impact.clamp_max(10.0).sum(1, True)

class feet_slip(Reward):
    def __init__(
        self, env: "LocomotionEnv", body_names: str, weight: float, enabled: bool = True
    ):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]

        self.body_ids, self.articulation_body_ids, self.body_names = resolve_contact_indices(
            self.contact_sensor, self.asset, body_names
        )
        self.body_ids = torch.tensor(self.body_ids, device=self.env.device)
        self.articulation_body_ids = torch.tensor(self.articulation_body_ids, device=self.env.device)

    def compute(self) -> torch.Tensor:
        in_contact = self.contact_sensor.data.current_contact_time[:, self.body_ids] > self.env.physics_dt
        feet_vel = self.asset.data.body_com_lin_vel_w[:, self.articulation_body_ids, :2]
        slip = (in_contact * feet_vel.norm(dim=-1).square()).sum(dim=1, keepdim=True)
        return -slip

class feet_upright(Reward):
    def __init__(
        self, env, body_names: str, xy_sigma: float, weight: float, enabled: bool = True
    ):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        
        self.body_ids_asset, _ = self.asset.find_bodies(body_names)

        down = torch.tensor([0.0, 0.0, -1.0], device=self.env.device)
        self.down = down.expand(self.num_envs, len(self.body_ids_asset), -1)
        self.xy_sigma = xy_sigma
        
    def compute(self):
        feet_quat_w = self.asset.data.body_link_quat_w[:, self.body_ids_asset]
        feet_projected_down = quat_apply(feet_quat_w, self.down)
        feet_projected_down_xy = feet_projected_down[:, :, :2].norm(dim=-1)
        rew = (torch.exp(-feet_projected_down_xy / self.xy_sigma) - 1.0)
        return rew.float().mean(dim=1, keepdim=True)

class feet_air_time_ref(Reward):
    def __init__(
        self,
        env: "LocomotionEnv",
        body_names: str,
        thres: float,
        weight: float,
        enabled: bool = True,
    ):
        super().__init__(env, weight, enabled)
        self.thres = thres
        self.asset: Articulation = self.env.scene["robot"]
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]

        self.body_ids, _, self.body_names = resolve_contact_indices(
            self.contact_sensor, self.asset, body_names
        )
        _validate_feet_order(self.env, self.body_names, context="feet_air_time_ref")
        self.body_ids = torch.tensor(self.body_ids, device=self.env.device)
        self.contact_cache = _get_contact_majority_cache(self.env, self.contact_sensor)

        self.reward_time = torch.zeros(self.num_envs, len(self.body_ids), device=self.env.device)

    def reset(self, env_ids):
        self.reward_time[env_ids] = 0.0
        self.contact_cache.reset(env_ids)

    def post_step(self, substep: int):
        self.contact_cache.post_step(substep)

    def update(self):
        self.contact_cache.update()

    def compute(self):
        current_contact = self.contact_cache.current_for(self.body_ids)
        first_contact = self.contact_cache.first_contact_for(self.body_ids)

        contact_diff = self.env.command_manager.feet_standing ^ current_contact
        self.reward_time = self.reward_time + torch.where(
            contact_diff, -self.env.step_dt, self.env.step_dt
        )
        
        self.reward = torch.sum(
            (self.reward_time - self.thres).clamp_max(0.0) * first_contact, dim=1, keepdim=True
        )
        
        self.reward_time = self.reward_time * (~current_contact)
        return self.reward

class feet_air_time_ref_dense(Reward):
    def __init__(
        self,
        env: "LocomotionEnv",
        body_names: str,
        body2_names: str | None = None,
        air_h_low: float = 0.035,
        air_h_high: float = 0.155,
        contact_h_low: float = 0.035,
        contact_h_high: float = 0.125,
        weight: float = 1.0,
        enabled: bool = True,
    ):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]

        articulation_body_ids, articulation_body_names = self.asset.find_bodies(body_names)
        _validate_feet_order(self.env, articulation_body_names, context="feet_air_time_ref_dense body_names")
        if body2_names is None:
            articulation_body2_ids = articulation_body_ids
        else:
            articulation_body2_ids, articulation_body2_names = self.asset.find_bodies(body2_names)
            _validate_feet_order(self.env, articulation_body2_names, context="feet_air_time_ref_dense body2_names")
        self.articulation_body_ids = articulation_body_ids
        self.articulation_body2_ids = articulation_body2_ids
        self.articulation_body_ids = torch.tensor(
            self.articulation_body_ids, device=self.env.device, dtype=torch.long
        )
        self.articulation_body2_ids = torch.tensor(
            self.articulation_body2_ids, device=self.env.device, dtype=torch.long
        )

        self.body_ids, _, self.body_names = resolve_contact_indices(
            self.contact_sensor, self.asset, body_names
        )
        _validate_feet_order(self.env, self.body_names, context="feet_air_time_ref_dense contact bodies")
        self.body_ids = torch.tensor(self.body_ids, device=self.env.device)
        self.contact_cache = _get_contact_majority_cache(self.env, self.contact_sensor)

        self.air_h_low = float(air_h_low)
        self.air_h_high = float(air_h_high)
        self.air_h_span = max(self.air_h_high - self.air_h_low, 1e-6)
        self.contact_h_low = float(contact_h_low)
        self.contact_h_high = float(contact_h_high)
        self.contact_h_span = max(self.contact_h_high - self.contact_h_low, 1e-6)

    def reset(self, env_ids):
        self.contact_cache.reset(env_ids)

    def post_step(self, substep: int):
        self.contact_cache.post_step(substep)

    def update(self):
        self.contact_cache.update()

    def compute(self):
        current_contact = self.contact_cache.current_for(self.body_ids)
        target_contact = self.env.command_manager.feet_standing

        # mismatch -> -1; both contact/both air -> height-based penalty in [-1, 0]
        mismatch = current_contact ^ target_contact
        both_air = (~current_contact) & (~target_contact)
        both_contact = current_contact & target_contact

        penalty = torch.zeros_like(current_contact, dtype=torch.float32)
        penalty[mismatch] = -1.0

        feet_height_air = torch.minimum(
            self.asset.data.body_link_pos_w[:, self.articulation_body_ids, 2],
            self.asset.data.body_link_pos_w[:, self.articulation_body2_ids, 2],
        )
        air_ratio = ((feet_height_air - self.air_h_low) / self.air_h_span).clamp(0.0, 1.0)
        air_penalty = -(1.0 - air_ratio)
        penalty = torch.where(both_air, air_penalty, penalty)

        feet_height_contact = torch.maximum(
            self.asset.data.body_link_pos_w[:, self.articulation_body_ids, 2],
            self.asset.data.body_link_pos_w[:, self.articulation_body2_ids, 2],
        )
        t_contact = (
            (feet_height_contact - self.contact_h_low) / self.contact_h_span
        ).clamp(0.0, 1.0)
        # low -> 0, high -> -1
        contact_penalty = -t_contact
        penalty = torch.where(both_contact, contact_penalty, penalty)

        return penalty.mean(dim=1, keepdim=True)

    def debug_draw(self):
        current_contact = self.contact_cache.current_for(self.body_ids)
        target_contact = self.env.command_manager.feet_standing

        feet_pos = 0.5 * (
            self.asset.data.body_link_pos_w[:, self.articulation_body_ids]
            + self.asset.data.body_link_pos_w[:, self.articulation_body2_ids]
        )

        both_match = ~(current_contact ^ target_contact)
        ref_contact_actual_air = target_contact & (~current_contact)
        ref_air_actual_contact = (~target_contact) & current_contact

        green_points = feet_pos[both_match]
        red_points = feet_pos[ref_contact_actual_air]
        blue_points = feet_pos[ref_air_actual_contact]

        if green_points.numel() > 0:
            self.env.debug_draw.point(green_points, color=(0.0, 1.0, 0.0, 1.0), size=40.0)
        if red_points.numel() > 0:
            self.env.debug_draw.point(red_points, color=(1.0, 0.0, 0.0, 1.0), size=40.0)
        if blue_points.numel() > 0:
            self.env.debug_draw.point(blue_points, color=(0.0, 0.3, 1.0, 1.0), size=40.0)

class feet_contact_count(Reward):
    def __init__(
        self, env: "LocomotionEnv", body_names: str, weight: float, enabled: bool = True
    ):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]

        self.body_ids, self.articulation_body_ids, self.body_names = resolve_contact_indices(
            self.contact_sensor, self.asset, body_names
        )
        self.body_ids = torch.tensor(self.body_ids, device=self.env.device)
        self.articulation_body_ids = torch.tensor(self.articulation_body_ids, device=self.env.device)
        self.contact_cache = _get_contact_majority_cache(self.env, self.contact_sensor)

    def reset(self, env_ids: torch.Tensor):
        self.contact_cache.reset(env_ids)

    def post_step(self, substep: int):
        self.contact_cache.post_step(substep)

    def update(self):
        self.contact_cache.update()

    def compute(self):
        first_contact = self.contact_cache.first_contact_for(self.body_ids)
        return first_contact.sum(1, keepdim=True)

    def debug_draw(self):
        current_contact = self.contact_cache.current_for(self.body_ids)
        if not current_contact.any():
            return
        feet_pos = self.asset.data.body_link_pos_w[:, self.articulation_body_ids].clone()
        feet_pos[..., 2] -= 0.1
        points = feet_pos[current_contact]
        if points.numel() > 0:
            self.env.debug_draw.point(points, color=(1.0, 0.0, 0.0, 1.0), size=20.0)


class joint_vel_l2(Reward):
    def __init__(self, env, joint_names: str, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids, _ = self.asset.find_joints(joint_names)
        self.joint_vel = torch.zeros(
            self.num_envs, 2, len(self.joint_ids), device=self.device
        )

    def post_step(self, substep):
        self.joint_vel[:, substep % 2] = self.asset.data.joint_vel[:, self.joint_ids]

    def compute(self) -> torch.Tensor:
        joint_vel = self.joint_vel.mean(1)
        return -joint_vel.square().sum(1, True)

class joint_acc_l2(Reward):
    def __init__(self, env, joint_names: str, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids, self.joint_names = self.asset.find_joints(joint_names)
        self.joint_acc = torch.zeros(
            self.num_envs, self.env.decimation, len(self.joint_ids), device=self.device
        )

    def post_step(self, substep):
        self.joint_acc[:, substep] = self.asset.data.joint_acc[:, self.joint_ids]
    
    def compute(self) -> torch.Tensor:
        joint_acc = self.joint_acc.mean(1)
        joint_acc = joint_acc.clamp(-5000.0, 5000.0)
        return -joint_acc.square().sum(1, True)

class joint_pos_limits(Reward):
    def __init__(self, env, weight: float, joint_names: str | List[str] =".*", soft_factor: float=0.9, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids, self.joint_names = resolve_matching_names(joint_names, self.asset.joint_names)
        jpos_limits = self.asset.data.joint_pos_limits[:, self.joint_ids]
        jpos_mean = (jpos_limits[..., 0] + jpos_limits[..., 1]) / 2
        jpos_range = jpos_limits[..., 1] - jpos_limits[..., 0]
        self.soft_factor = soft_factor
        self.soft_limits = torch.zeros_like(jpos_limits)
        self.soft_limits[..., 0] = jpos_mean - 0.5 * jpos_range * soft_factor
        self.soft_limits[..., 1] = jpos_mean + 0.5 * jpos_range * soft_factor

    def compute(self) -> torch.Tensor:
        jpos = self.asset.data.joint_pos[:, self.joint_ids]
        violation_min = (self.soft_limits[..., 0] - jpos).clamp_min(0.0)
        violation_max = (jpos - self.soft_limits[..., 1]).clamp_min(0.0)
        return -(violation_min + violation_max).sum(1, keepdim=True) / (1-self.soft_factor)

class joint_torque_limits(Reward):
    def __init__(self, env, weight: float, joint_names: str | List[str] =".*", soft_factor: float=0.9, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids, self.joint_names = resolve_matching_names(joint_names, self.asset.joint_names)
        self.soft_factor = float(soft_factor)

        # MJLab: derive limits from actuator forcerange (if available)
        model = self.env.sim.model
        if not hasattr(model, "actuator_forcerange"):
            raise RuntimeError("Actuator force limits are not available in MJLab model.")
        self.ctrl_ids = self.asset.indexing.ctrl_ids

        actuator_names = list(self.asset.actuator_names)
        name_to_act = {n: i for i, n in enumerate(actuator_names)}
        act_idx = []
        for name in self.joint_names:
            if name not in name_to_act:
                raise RuntimeError(f"Actuator for joint '{name}' not found.")
            else:
                act_idx.append(name_to_act[name])
        self.act_idx = torch.tensor(act_idx, device=self.device, dtype=torch.long)

    def _get_soft_limits(self) -> torch.Tensor:
        force_range = self.env.sim.model.actuator_forcerange[:, self.ctrl_ids]
        limits = torch.maximum(force_range[..., 0].abs(), force_range[..., 1].abs())
        return limits[:, self.act_idx] * self.soft_factor
    
    def compute(self) -> torch.Tensor:
        soft_limits = self._get_soft_limits().clamp_min(1e-6)
        applied_torque = self.asset.data.actuator_force[:, self.act_idx]
        violation_high = (applied_torque / soft_limits - 1.0).clamp_min(0.0)
        violation_low = (-applied_torque / soft_limits - 1.0).clamp_min(0.0)
        return - (violation_high + violation_low).sum(dim=1, keepdim=True)

class action_rate_l2(Reward):
    def compute(self) -> torch.Tensor:
        action_buf = self.action_manager.get_recent_action_rate_actions(2)
        action_diff = action_buf[:, 0, :] - action_buf[:, 1, :]
        return - action_diff.square().sum(dim=-1, keepdim=True)


class action_rate2_l2(Reward):
    def compute(self) -> torch.Tensor:
        action_buf = self.action_manager.get_recent_action_rate_actions(3)
        action_diff = (
            action_buf[:, 0, :] - 2 * action_buf[:, 1, :] + action_buf[:, 2, :]
        )
        return - action_diff.square().sum(dim=-1, keepdim=True)
