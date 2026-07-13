from pathlib import Path

import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from scripts.utils.helpers import make_env_policy


def neutralize_eval_randomization(cfg):
    cfg.task.command.reinit_prob = 0.0
    for key in cfg.task.command.init_noise:
        cfg.task.command.init_noise[key] = 0.0
    for values in (cfg.task.command.shared_joint_pos.std, cfg.task.command.shared_joint_pos.bias, cfg.task.command.shared_joint_vel.std):
        for key in values:
            values[key] = 0.0
    for term in cfg.task.observation.policy.values():
        if "noise_std" in term:
            term.noise_std = 0.0
        if "bias_noise_std" in term:
            term.bias_noise_std = 0.0

    randomization = cfg.task.randomization
    randomization.perturb_body_com.com_range = [0.0, 0.0]
    randomization.perturb_body_materials.static_friction_range = [1.0, 1.0]
    randomization.perturb_body_materials.solref_time_constant_range = [0.02, 0.02]
    randomization.perturb_body_materials.solref_dampratio_range = [1.0, 1.0]
    for name in ("stiffness_range", "damping_range", "armature_range"):
        for key in randomization.motor_params_implicit[name]:
            randomization.motor_params_implicit[name][key] = [1.0, 1.0]
    for key in randomization.motor_params_implicit.frictionloss_range:
        randomization.motor_params_implicit.frictionloss_range[key] = [0.0, 0.0]
    for key in randomization.random_joint_offset:
        randomization.random_joint_offset[key] = [0.0, 0.0]
    for key in ("x", "y", "z", "roll", "pitch", "yaw"):
        randomization.perturb_root_vel[key] = [0.0, 0.0]
    randomization.perturb_body_wrench.enabled = False
    randomization.perturb_gravity.std = 0.0


def configure_controlled_load(cfg, max_load_kg: float):
    load_cfg = cfg.task.randomization.window_cap_hand_load
    load_cfg.enabled = True
    load_cfg.label_path = ""
    load_cfg.constant_cap_kg = max_load_kg
    load_cfg.max_load_kg = max_load_kg
    load_cfg.body_names = ["left_wrist_roll_link", "right_wrist_roll_link"]
    load_cfg.force_application_body_names = ["left_hand_mimic", "right_hand_mimic"]
    load_cfg.inertial_force_scale_range = [0.0, 0.0]
    load_cfg.force_application_offset_radius_range_m = [0.0, 0.0]
    load_cfg.hand_force_fraction_range = [1.0, 1.0]
    load_cfg.window_no_load_ratio = 0.0
    load_cfg.single_side_load_ratio = 0.0


def make_eval_env(cfg_path: str, checkpoint_path: str, num_envs: int, max_load_kg: float):
    cfg = OmegaConf.load(Path(cfg_path).expanduser())
    OmegaConf.set_struct(cfg, False)
    cfg.checkpoint_path = str(Path(checkpoint_path).expanduser())
    cfg.vecnorm = "eval"
    cfg.eval_render = False
    cfg.app.headless = True
    cfg.app.enable_cameras = False
    cfg.wandb.mode = "disabled"
    cfg.task.num_envs = int(num_envs)
    cfg.task.max_episode_length = 20000
    neutralize_eval_randomization(cfg)

    configure_controlled_load(cfg, max_load_kg)

    env, agent, _, _ = make_env_policy(cfg)
    if hasattr(agent, "step_schedule"):
        agent.step_schedule(1.0, 0)
    if hasattr(env, "step_schedule"):
        env.step_schedule(1.0, 0)
    base_env = unwrap_base_env(env)
    scheduler = MotionScheduler(base_env)
    scheduler.install()
    return env, agent, base_env, scheduler


def unwrap_base_env(env):
    base_env = env
    while hasattr(base_env, "base_env"):
        base_env = base_env.base_env
    return base_env


class MotionScheduler:
    def __init__(self, base_env):
        self.command = base_env.command_manager
        self.motion_ids = torch.zeros(base_env.num_envs, dtype=torch.long, device=base_env.device)
        self.starts = torch.zeros_like(self.motion_ids)

    def install(self):
        command = self.command

        def sample_init(env_ids: torch.Tensor):
            env_ids = env_ids.long()
            motion_ids = self.motion_ids[env_ids]
            lengths = command.dataset.set_motion_ids(env_ids, motion_ids)
            starts = torch.minimum(self.starts[env_ids], (lengths.long() - 1).clamp_min(0))
            command.lengths[env_ids] = lengths
            command.t[env_ids] = starts.to(command.t.dtype)
            command._reinit_requested[env_ids] = False
            teacher, student = command.dataset.get_teacher_student_slice(
                env_ids,
                command.t[env_ids],
                teacher_steps=1,
                student_steps=1,
            )
            command._set_motion_origin_offset(env_ids, teacher, student)
            motion, _ = command._apply_motion_origin_offset(teacher, student, env_ids)
            command.sample_init_robot(env_ids, motion)

        command.sample_init = sample_init

    def assign(self, env_id: int, motion_id: int, start: int):
        self.motion_ids[env_id] = motion_id
        self.starts[env_id] = max(start, 0)


def set_load(base_env, env_ids: list[int], current_kg: list[float], target_kg: list[float], ramp_sec: float):
    ids = torch.as_tensor(env_ids, dtype=torch.long, device=base_env.device)
    load = base_env.randomizations["window_cap_hand_load"]
    load.set_controlled_load(
        ids,
        torch.as_tensor(current_kg, dtype=torch.float32, device=base_env.device),
        torch.as_tensor(target_kg, dtype=torch.float32, device=base_env.device),
        ramp_sec,
    )


def reset_envs(env, base_env, env_ids: list[int]):
    mask = torch.zeros((base_env.num_envs, 1), dtype=torch.bool, device=base_env.device)
    mask[env_ids, 0] = True
    return env.reset(TensorDict({"_reset": mask}, batch_size=[base_env.num_envs], device=base_env.device))


def merge_reset(td, reset_td, env_ids: list[int]):
    ids = torch.as_tensor(env_ids, dtype=torch.long, device=td.device)
    for key in reset_td.keys(True, True):
        dst = td.get(key)
        src = reset_td.get(key)
        if isinstance(dst, torch.Tensor) and isinstance(src, torch.Tensor) and dst.shape[:1] == src.shape[:1]:
            dst[ids] = src[ids]
            td.set(key, dst)
    return td


def window_bounds(length: int, window_steps: int):
    end = max(int(length), 0)
    return [(start, min(start + window_steps, end)) for start in range(0, end, window_steps)]
