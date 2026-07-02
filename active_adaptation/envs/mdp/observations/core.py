import torch
import numpy as np
import einops
from typing import TYPE_CHECKING

import active_adaptation
from active_adaptation.utils.math import quat_apply, quat_apply_inverse, yaw_quat, quat_mul, quat_conjugate
import active_adaptation.utils.symmetry as sym_utils
import active_adaptation.utils.joint_order as joint_order_utils
from active_adaptation.utils.helpers import TensorRingBuffer
from active_adaptation.envs.mdp.utils import add_spherical_noise, perturb_quaternion, random_noise

if TYPE_CHECKING:
    from mjlab.entity import Entity as Articulation
    from mjlab.sensor import ContactSensor, BuiltinSensor
    from active_adaptation.envs.base import _Env

from mjlab.utils.lab_api.string import resolve_matching_names
from active_adaptation.envs.mdp.contact_utils import resolve_contact_indices
from .base import Observation


class root_angvel_b_history(Observation):
    def __init__(self, env, noise_std: float=0., history_steps: list[int]=[1]):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.imu_ang_vel_sensor: BuiltinSensor = self.env.scene["robot/imu_ang_vel"]
        self.noise_std = noise_std
        self.history_steps = history_steps
        buffer_size = max(history_steps) + 1
        self.buffer = TensorRingBuffer(
            self.num_envs,
            buffer_size,
            3,
            device=self.device,
            dtype=self.imu_ang_vel_sensor.data.dtype,
        )
        self._history_idx = torch.as_tensor(history_steps, device=self.device, dtype=torch.long)
        self.update()
    
    def reset(self, env_ids):
        self.buffer.reset(env_ids)

    def update(self):
        root_ang_vel_b = self.imu_ang_vel_sensor.data
        if self.noise_std > 0:
            root_ang_vel_b = add_spherical_noise(root_ang_vel_b, self.noise_std)
        self.buffer.push(root_ang_vel_b)

    def compute(self) -> torch.Tensor:
        return self.buffer.take(self._history_idx).reshape(self.num_envs, -1)
    
    def symmetry_transforms(self):
        transform = sym_utils.SymmetryTransform(perm=torch.arange(3), signs=[-1., 1., -1.])
        return transform.repeat(len(self.history_steps))

class root_linacc_b_history(Observation):
    def __init__(self, env, noise_std: float=0., bias_noise_std: float=0., history_steps: list[int]=[0]):
        super().__init__(env)
        self.imu_lin_acc_sensor: BuiltinSensor = self.env.scene["robot/imu_lin_acc"]
        self.noise_std = max(noise_std, 0.)
        self.bias_noise_std = max(bias_noise_std, 0.)
        self.history_steps = history_steps
        buffer_size = max(history_steps) + 1
        self.buffer = TensorRingBuffer(
            self.num_envs,
            buffer_size,
            3,
            device=self.device,
            dtype=self.imu_lin_acc_sensor.data.dtype,
        )
        self.bias = torch.zeros((self.num_envs, 3), device=self.device, dtype=self.imu_lin_acc_sensor.data.dtype)
        self._history_idx = torch.as_tensor(history_steps, device=self.device, dtype=torch.long)
        self.update()

    def reset(self, env_ids):
        if self.bias_noise_std > 0:
            bias = torch.zeros((len(env_ids), 3), device=self.device, dtype=self.bias.dtype)
            self.bias[env_ids] = add_spherical_noise(bias, self.bias_noise_std)
        else:
            self.bias[env_ids] = 0
        self.buffer.reset(env_ids)

    def update(self):
        lin_acc_b = self.imu_lin_acc_sensor.data + self.bias
        if self.noise_std > 0:
            lin_acc_b = add_spherical_noise(lin_acc_b, self.noise_std)
        self.buffer.push(lin_acc_b)

    def compute(self) -> torch.Tensor:
        return self.buffer.take(self._history_idx).reshape(self.num_envs, -1)

    def symmetry_transforms(self):
        transform = sym_utils.SymmetryTransform(perm=torch.arange(3), signs=[1., -1., 1.])
        return transform.repeat(len(self.history_steps))

class projected_gravity_history(Observation):
    def __init__(
        self,
        env,
        noise_std: float=0.,
        history_steps: list[int]=[1],
        bias_noise_std: float=0.,
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.noise_std = noise_std
        self.bias_noise_std = max(bias_noise_std, 0.)
        self.history_steps = history_steps
        buffer_size = max(history_steps) + 1
        self.buffer = TensorRingBuffer(
            self.num_envs,
            buffer_size,
            3,
            device=self.device,
            dtype=torch.float32,
        )
        self.bias_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.float32)
        self.bias_quat[:, 0] = 1.0
        self._gravity_vec_w = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=torch.float32).unsqueeze(0)
        self._history_idx = torch.as_tensor(history_steps, device=self.device, dtype=torch.long)
        self.update()
    
    def reset(self, env_ids):
        base_quat = torch.zeros((len(env_ids), 4), device=self.device, dtype=self.asset.data.root_link_quat_w.dtype)
        base_quat[:, 0] = 1.0
        if self.bias_noise_std > 0:
            base_quat = perturb_quaternion(base_quat, self.bias_noise_std)
        self.bias_quat[env_ids] = base_quat

        self.buffer.reset(env_ids)
    
    def update(self):
        root_quat = quat_mul(self.bias_quat, self.asset.data.root_link_quat_w)
        if self.noise_std > 0:
            root_quat = perturb_quaternion(root_quat, self.noise_std)
        projected_gravity_b = quat_apply_inverse(root_quat, self._gravity_vec_w.expand(self.num_envs, -1))
        projected_gravity_b = projected_gravity_b / projected_gravity_b.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        self.buffer.push(projected_gravity_b)
    
    def compute(self):
        return self.buffer.take(self._history_idx).reshape(self.num_envs, -1)

    def symmetry_transforms(self):
        transform = sym_utils.SymmetryTransform(perm=torch.arange(3), signs=[1, -1, 1])
        return transform.repeat(len(self.history_steps))

class root_linvel_b_history(Observation):
    def __init__(self, env, noise_std: float=0., history_steps: list[int]=[0]):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.noise_std = noise_std
        self.history_steps = history_steps
        buffer_size = max(history_steps) + 1
        self.buffer = TensorRingBuffer(
            self.num_envs,
            buffer_size,
            3,
            device=self.device,
            dtype=self.asset.data.root_link_lin_vel_b.dtype,
        )
        self._history_idx = torch.as_tensor(history_steps, device=self.device, dtype=torch.long)
        self.update()

    def reset(self, env_ids):
        self.buffer.reset(env_ids)

    def update(self):
        root_linvel_b = self.asset.data.root_link_lin_vel_b
        if self.noise_std > 0:
            root_linvel_b = random_noise(root_linvel_b, self.noise_std)
        self.buffer.push(root_linvel_b)
    
    def compute(self) -> torch.Tensor:
        return self.buffer.take(self._history_idx).reshape(self.num_envs, -1)

    def symmetry_transforms(self):
        transform = sym_utils.SymmetryTransform(perm=torch.arange(3), signs=[1, -1, 1])
        return transform.repeat(len(self.history_steps))

    def debug_draw(self):
        if self.env._has_gui():
            linvel = self.asset.data.root_link_lin_vel_w
            self.env.debug_draw.vector(
                self.asset.data.root_link_pos_w + torch.tensor([0.0, 0.0, 0.2], device=self.device),
                linvel,
                color=(0.8, 0.1, 0.1, 1.)
            )
    
class JointObs(Observation):
    def __init__(
        self, 
        env,
        joint_names: str=".*",
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        joint_ids, self.joint_names = joint_order_utils.resolve_joint_order(
            self.asset, joint_names
        )
        self.joint_ids = torch.tensor(joint_ids, device=self.device)
        self.num_joints = len(joint_ids)

class joint_pos_history(JointObs):
    def __init__(
        self,
        env,
        joint_names: str=".*",
        history_steps: list[int]=[1], 
        noise_std: float=0.,
    ):
        super().__init__(env, joint_names)
        self.history_steps = history_steps
        self.buffer_size = max(history_steps) + 1
        self.noise_std = max(float(noise_std), 0.0)
        self.noise_enabled = self.noise_std > 0.0
        from active_adaptation.envs.mdp.action import JointPosition
        action_manager: JointPosition = self.env.action_manager
        self.joint_pos_offset = action_manager.offset

        self.joint_pos = torch.zeros(self.num_envs, 2, self.num_joints, device=self.device)
        self.buffer = TensorRingBuffer(
            self.num_envs,
            self.buffer_size,
            self.num_joints,
            device=self.device,
            dtype=self.joint_pos.dtype,
        )
        self._history_idx = torch.as_tensor(history_steps, device=self.device, dtype=torch.long)
    
    def post_step(self, substep):
        self.joint_pos[:, substep % 2] = self.asset.data.joint_pos[:, self.joint_ids]
    
    def reset(self, env_ids):
        self.buffer.reset(env_ids)
    
    def update(self):
        command_manager = getattr(self.env, "command_manager", None)
        get_shared_joint_pos = getattr(command_manager, "get_shared_noisy_joint_pos", None)
        if (
            self.noise_enabled
            and get_shared_joint_pos is not None
            and getattr(command_manager, "shared_joint_pos_noise_enabled", False)
        ):
            joint_pos = get_shared_joint_pos(self.joint_ids)
        else:
            joint_pos = self.joint_pos.mean(1)
            if self.noise_enabled:
                joint_pos = random_noise(joint_pos, self.noise_std)
        self.buffer.push(joint_pos)
    
    def compute(self):
        joint_pos_selected = self.buffer.take(self._history_idx)
        joint_pos_selected = joint_pos_selected - self.joint_pos_offset[:, None, self.joint_ids]
        return joint_pos_selected.reshape(self.num_envs, -1)

    def symmetry_transforms(self):
        transform = sym_utils.joint_space_symmetry(self.asset, self.joint_names)
        return transform.repeat(len(self.history_steps))

class joint_vel_history(JointObs):
    def __init__(
        self,
        env,
        joint_names: str=".*",
        history_steps: list[int]=[1],
        noise_std: float=0.,
    ):
        super().__init__(env, joint_names)
        self.history_steps = history_steps
        self.buffer_size = max(history_steps) + 1
        self.noise_std = max(float(noise_std), 0.0)
        self.noise_enabled = self.noise_std > 0.0

        self.joint_vel = torch.zeros(self.num_envs, 2, self.num_joints, device=self.device)
        self.buffer = TensorRingBuffer(
            self.num_envs,
            self.buffer_size,
            self.num_joints,
            device=self.device,
            dtype=self.joint_vel.dtype,
        )
        self._history_idx = torch.as_tensor(history_steps, device=self.device, dtype=torch.long)

    def post_step(self, substep):
        self.joint_vel[:, substep % 2] = self.asset.data.joint_vel[:, self.joint_ids]

    def reset(self, env_ids):
        self.buffer.reset(env_ids)

    def update(self):
        command_manager = getattr(self.env, "command_manager", None)
        get_shared_joint_vel = getattr(command_manager, "get_shared_noisy_joint_vel", None)
        if (
            self.noise_enabled
            and get_shared_joint_vel is not None
            and getattr(command_manager, "shared_joint_vel_noise_enabled", False)
        ):
            joint_vel = get_shared_joint_vel(self.joint_ids)
        else:
            joint_vel = self.joint_vel.mean(1)
            if self.noise_enabled:
                joint_vel = random_noise(joint_vel, self.noise_std)
        self.buffer.push(joint_vel)

    def compute(self):
        joint_vel_selected = self.buffer.take(self._history_idx)
        return joint_vel_selected.reshape(self.num_envs, -1)

    def symmetry_transforms(self):
        transform = sym_utils.joint_space_symmetry(self.asset, self.joint_names)
        return transform.repeat(len(self.history_steps))

class applied_torque(JointObs):
    def __init__(self, env, joint_names: str=".*"):
        super().__init__(env, joint_names)

        actuator_names = list(self.asset.actuator_names)
        name_to_act = {n: i for i, n in enumerate(actuator_names)}
        act_idx = []
        for name in self.joint_names:
            if name not in name_to_act:
                raise RuntimeError(f"Actuator for joint '{name}' not found.")
            else:
                act_idx.append(name_to_act[name])
        self.act_idx = torch.tensor(act_idx, device=self.device, dtype=torch.long)
    
    def compute(self) -> torch.Tensor:
        return self.asset.data.actuator_force[:, self.act_idx]

    def symmetry_transforms(self):
        transform = sym_utils.joint_space_symmetry(self.asset, self.joint_names)
        return transform

class feet_contact_state(Observation):
    def __init__(self, env, body_names, divide_by_mass: bool=True):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        robot_cfg = getattr(self.env.cfg, "robot", None)
        mass_total = getattr(robot_cfg, "mass", None) if robot_cfg is not None else None
        if mass_total is None:
            breakpoint()
        self.default_mass_total = mass_total * 9.81
        self.denom = self.default_mass_total if divide_by_mass else 1.
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]
        self.body_ids, _, self.body_names = resolve_contact_indices(
            self.contact_sensor, self.asset, body_names
        )

    def compute(self) -> torch.Tensor:
        contact_forces = self.contact_sensor.data.force_history[:, self.body_ids, :, :].mean(2)
        force = (contact_forces / self.denom).clamp(-10.0, 10.0)
        contact_time = self.contact_sensor.data.current_contact_time[:, self.body_ids]
        air_time = self.contact_sensor.data.current_air_time[:, self.body_ids]
        in_contact = (contact_time > self.env.physics_dt).float()
        return torch.cat(
            [
                force.view(self.num_envs, -1),
                in_contact,
                contact_time,
                air_time,
            ],
            dim=-1,
        )
    
    def symmetry_transforms(self):
        force_transform = sym_utils.cartesian_space_symmetry(
            self.asset, self.body_names, sign=[1, -1, 1]
        )
        scalar_transform = sym_utils.cartesian_space_symmetry(
            self.asset, self.body_names, sign=(1,)
        )
        return sym_utils.SymmetryTransform.cat(
            [force_transform, scalar_transform, scalar_transform, scalar_transform]
        )

class body_height(Observation):
    def __init__(self, env, body_names=".*_foot"):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.body_ids, self.body_names = self.asset.find_bodies(body_names)
    
    def compute(self) -> torch.Tensor:
        return self.asset.data.body_link_pos_w[:, self.body_ids, 2].reshape(
            self.num_envs, -1
        )

    def symmetry_transforms(self):
        return sym_utils.cartesian_space_symmetry(self.asset, self.body_names, sign=(1,))

class prev_actions(Observation):
    def __init__(self, env, steps: int=1):
        super().__init__(env)
        self.steps = steps
        self.action_manager = self.env.action_manager
    
    def compute(self):
        action_buf = self.action_manager.get_recent_action_obs(self.steps)
        return action_buf.reshape(self.num_envs, -1)

    def symmetry_transforms(self):
        transform = self.action_manager.symmetry_transforms().repeat(self.steps)
        return transform

class applied_action(JointObs):
    def __init__(self, env):
        super().__init__(env)
        self.action_manager = self.env.action_manager

    def compute(self) -> torch.Tensor:
        return self.asset.data.joint_pos_target[:, self.joint_ids]

    def symmetry_transforms(self):
        transform = sym_utils.joint_space_symmetry(self.asset, self.joint_names)
        return transform
