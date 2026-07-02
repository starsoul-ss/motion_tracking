import torch
import einops
from typing import Dict, Literal, Tuple, Union
from tensordict import TensorDictBase
import mjlab.utils.lab_api.string as string_utils
import active_adaptation.utils.symmetry as symmetry_utils
import active_adaptation.utils.joint_order as joint_order_utils
from active_adaptation.utils.helpers import TensorRingBuffer

from .base import ActionManager


class JointPosition(ActionManager):
    def __init__(
        self,
        env,
        action_scaling: Dict[str, float] | float = 0.5,
        max_delay: int | None = None,
        delay_full_progress: float = 1.0,
        boot_delay_steps: int = 0,
        alpha: Tuple[float, float] = (0.9, 0.9),
        prev_action_obs: Literal["sampled", "mean"] = "sampled",
        action_rate_source: Literal["sampled", "mean"] = "sampled",
        torque_limit_scale_range: Tuple[float, float] = (1.0, 1.0),
        torque_limit_progress_range: Tuple[float, float] = (0.0, 1.0),
        boot_protect: bool = False,
        **kwargs,
    ):
        super().__init__(env)

        # ------------------------------------------------------------------ cfg
        self.joint_ids, self.joint_names, self.action_scaling = (
            joint_order_utils.resolve_joint_order_with_values(
                self.asset, dict(action_scaling)
            )
        )
        self.action_scaling = torch.tensor(self.action_scaling, device=self.device)
        self.action_dim = len(self.joint_ids)

        self.max_delay = max_delay or 0  # physics steps
        self.delay_full_progress = float(delay_full_progress)
        if not (0.0 < self.delay_full_progress <= 1.0):
            raise ValueError(
                f"delay_full_progress must be in (0, 1], got {self.delay_full_progress}."
            )
        self.boot_delay_steps = int(max(boot_delay_steps, 0))
        self.prev_action_obs = str(prev_action_obs)
        if self.prev_action_obs not in {"sampled", "mean"}:
            raise ValueError(
                f"prev_action_obs must be 'sampled' or 'mean', got {self.prev_action_obs!r}."
            )
        self.action_rate_source = str(action_rate_source)
        if self.action_rate_source not in {"sampled", "mean"}:
            raise ValueError(
                f"action_rate_source must be 'sampled' or 'mean', got {self.action_rate_source!r}."
            )

        self.alpha_range = alpha
        self.torque_limit_scale_range = tuple(float(x) for x in torque_limit_scale_range)
        self.torque_limit_progress_range = tuple(float(x) for x in torque_limit_progress_range)
        self._torque_limit_scale = None
        self._torque_limit_curriculum_enabled = self.torque_limit_scale_range != (1.0, 1.0)

        # Boot‑protection ----------------------------------------------------
        self.boot_protect_enabled = boot_protect
        if self.boot_protect_enabled:
            self.boot_delay = torch.zeros(self.num_envs, 1, dtype=int, device=self.device)

        # Persistent tensors -------------------------------------------------
        self.default_joint_pos = self.asset.data.default_joint_pos.clone()
        self.offset = torch.zeros_like(self.default_joint_pos)

        with torch.device(self.device):
            # Max delay index used below is ceil(max_delay / decimation), so
            # history length must be at least that index + 1.
            max_hist_idx = (self.max_delay + self.env.decimation - 1) // self.env.decimation
            hist = max(max_hist_idx + 1, 8)
            self._action_buf = TensorRingBuffer(
                self.num_envs,
                hist,
                self.action_dim,
                device=self.device,
                dtype=torch.float32,
            )
            self._action_mean_buf = TensorRingBuffer(
                self.num_envs,
                hist,
                self.action_dim,
                device=self.device,
                dtype=torch.float32,
            )
            self.applied_action = torch.zeros(self.num_envs, self.action_dim)
            self.alpha = torch.ones(self.num_envs, 1)
            self.delay = torch.zeros(self.num_envs, 1, dtype=int)
            self.delay_probs = torch.zeros(int(self.max_delay) + 1, dtype=torch.float32)

        # Torque limit curriculum -------------------------------------------
        if "actuator_forcerange" not in self.env.sim.expanded_fields:
            self.env.sim.expand_model_fields(("actuator_forcerange",))
        self.model = self.env.sim.model
        self._setup_torque_limit_curriculum()

        # Initialize scheduled delay cap at progress=0.
        self.step_schedule(0.0, None)

    # --------------------------------------------------------------------- util
    def resolve(self, spec, names=None):
        """Convenience helper for user APIs."""
        target_names = (
            joint_order_utils.get_joint_name_order(self.asset)
            if names is None
            else names
        )
        return string_utils.resolve_matching_names_values(dict(spec), target_names)
    
    def symmetry_transforms(self):
        transform = symmetry_utils.joint_space_symmetry(self.asset, self.joint_names)
        return transform

    def _setup_torque_limit_curriculum(self):
        self.torque_limit_ctrl_ids = self.asset.indexing.ctrl_ids
        if len(self.torque_limit_ctrl_ids) != self.action_dim:
            raise RuntimeError(
                "Torque limit curriculum currently assumes one controllable actuator per action dimension, "
                f"got {len(self.torque_limit_ctrl_ids)} actuators and {self.action_dim} action dims."
            )
        default_force_range = self.env.sim.get_default_field("actuator_forcerange")
        self.default_actuator_forcerange = default_force_range[self.torque_limit_ctrl_ids].clone()

    def _set_torque_limit_scale(self, scale: float):
        scale = float(scale)
        if self._torque_limit_scale is not None and abs(scale - self._torque_limit_scale) < 1e-6:
            return
        force_range = self.default_actuator_forcerange.unsqueeze(0) * scale
        self.model.actuator_forcerange[:, self.torque_limit_ctrl_ids] = force_range
        self._torque_limit_scale = scale

    def _schedule_torque_limit(self, progress: float):
        if not self._torque_limit_curriculum_enabled:
            self._set_torque_limit_scale(1.0)
            return
        start, end = self.torque_limit_progress_range
        start_scale, end_scale = self.torque_limit_scale_range
        p = float(min(max(progress, 0.0), 1.0))
        if p <= start:
            scale = start_scale
        elif p >= end:
            scale = end_scale
        else:
            alpha = (p - start) / (end - start)
            scale = start_scale + alpha * (end_scale - start_scale)
        self._set_torque_limit_scale(scale)

    def step_schedule(self, progress: float, iters: int | None = None):
        D = int(self.max_delay)
        if self.env.student_train:
            self.delay_probs.fill_(1.0 / float(D + 1))
            self._schedule_torque_limit(1.0)
            return

        if D <= 0:
            self.delay_probs.fill_(1.0)
            self._schedule_torque_limit(progress)
            return

        p = float(min(max(progress, 0.0), 1.0))
        p = min(p / self.delay_full_progress, 1.0)
        # Smooth transition:
        # [0, 1/(D+1)] -> all delay=0,
        # then linearly blend uniform(0..k) -> uniform(0..k+1) per segment.
        q = max(0.0, min(float(D), p * float(D + 1) - 1.0))
        k = int(q)
        a = q - float(k)
        kp1 = min(k + 1, D)

        self.delay_probs.zero_()

        self.delay_probs[: k + 1] += (1.0 - a) / float(k + 1)
        if kp1 > k:
            self.delay_probs[: kp1 + 1] += a / float(kp1 + 1)

        # Keep probabilities normalized against numerical drift.
        self.delay_probs /= self.delay_probs.sum().clamp_min(1e-8)
        self._schedule_torque_limit(progress)

    # ------------------------------------------------------------------- reset
    def reset(self, env_ids: torch.Tensor):
        self._action_buf.reset(env_ids)
        self._action_mean_buf.reset(env_ids)
        self.applied_action[env_ids] = 0

        # Delay selection ---------------------------------------------------
        sampled_delay = torch.multinomial(self.delay_probs, len(env_ids), replacement=True)
        self.delay[env_ids] = sampled_delay.unsqueeze(-1).to(self.delay.dtype)
        if self.boot_protect_enabled:
            self.boot_delay[env_ids] = self.boot_delay_steps

        # α per environment --------------------------------------------------
        alpha = torch.empty(len(env_ids), 1, device=self.device).uniform_(*self.alpha_range)
        self.alpha[env_ids] = alpha

    # ---------------------------------------------------------------- forward
    def __call__(self, tensordict: TensorDictBase, substep: int):
        if substep == 0:
            raw_action = tensordict["action"].clamp(-10, 10)

            ### debug symmetry
            # raw_action = self.symmetry_transforms().to(raw_action.device).forward(raw_action)

            self._action_buf.push(raw_action)
            if self.prev_action_obs == "mean" or self.action_rate_source == "mean":
                mean_action = tensordict.get("loc", None)
                if mean_action is None:
                    raise KeyError(
                        "JointPosition mean-action history requires policy output key 'loc' "
                        "in the action tensordict."
                    )
                self._action_mean_buf.push(mean_action.clamp(-10, 10))

        # Communication delay ----------------------------------------------
        idx = (self.delay - substep + self.env.decimation - 1) // self.env.decimation
        idx = idx.clamp(0, self._action_buf.capacity - 1)
        delayed_action = self._action_buf.take_per_env(idx.squeeze(-1))
        self.applied_action.lerp_(delayed_action, self.alpha)
        # self.applied_action[:] = delayed_action  # no smoothing for now

        # Joint targets -----------------------------------------------------
        pos_tgt = self.default_joint_pos + self.offset
        pos_tgt[:, self.joint_ids] += self.applied_action * self.action_scaling

        # Optional boot‑protection -----------------------------------------
        if self.boot_protect_enabled:
            pos_tgt = torch.where(
                self.boot_delay > 0,
                self.env.command_manager.joint_pos_boot_protect,
                pos_tgt,
            )
            self.boot_delay.sub_(1).clamp_min_(0)

        # Write to simulator -----------------------------------------------
        self.asset.set_joint_position_target(pos_tgt)

    def get_recent_actions(self, steps: int) -> torch.Tensor:
        return self._action_buf.recent(steps)

    def _recent_actions_from_source(self, source: str, steps: int) -> torch.Tensor:
        if source == "mean":
            return self._action_mean_buf.recent(steps)
        if source == "sampled":
            return self._action_buf.recent(steps)
        raise ValueError(f"Unknown action history source: {source!r}.")

    def get_recent_action_obs(self, steps: int) -> torch.Tensor:
        return self._recent_actions_from_source(self.prev_action_obs, steps)

    def get_recent_action_rate_actions(self, steps: int) -> torch.Tensor:
        return self._recent_actions_from_source(self.action_rate_source, steps)
