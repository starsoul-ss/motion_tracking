import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from torchrl.data import Composite, TensorSpec
from torchrl.modules import ProbabilisticActor
from tensordict import TensorDict
from tensordict.nn import (
    TensorDictModuleBase,
    TensorDictModule as Mod,
    TensorDictSequential as Seq,
)

from hydra.core.config_store import ConfigStore

# ---- utils ------------------------------------------------------------------------------------ #
from ..modules.distributions import IndependentNormal
from ..modules.valuenorm import ValueNorm1, ValueNormFake
from .common import *
from ..modules.opt import build_optimizer
import active_adaptation as aa

__all__ = ["PPOPolicy", "PPOConfig"]


def _schedule_value(schedule: Any, progress: float) -> float:
    if isinstance(schedule, (int, float)):
        return float(schedule)

    points = [(float(x), float(y)) for x, y in schedule]
    if not points:
        raise ValueError("schedule cannot be empty")

    progress = float(progress)
    if progress <= points[0][0]:
        return points[0][1]

    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x1 <= x0:
            raise ValueError("schedule progress points must be strictly increasing")
        if progress <= x1:
            return y0 + (y1 - y0) * (progress - x0) / (x1 - x0)

    return points[-1][1]


# ------------------------------------------------------------------------------------------------ #
# 1. Config
# ------------------------------------------------------------------------------------------------ #


@dataclass
class PPOConfig:
    _target_: str = "active_adaptation.learning.ppo.ppo.PPOPolicy"
    name: str = "ppo"

    # PPO hyper‑params
    train_every: int = 32
    ppo_epochs: int = 3
    num_minibatches: int = 8

    actor_start_lr: float = 1e-4
    # actor_start_lr: float = 5e-4
    critic_lr: float = 5e-4
    optimizer: str = "muon"  # adam | muon
    optimizer_weight_decay: float = 0.0

    # actor lr scheduling based on kl divergence
    desired_kl_upper: Any = field(default_factory=lambda: [
        [0.0, 0.015],
        [0.15, 0.015],
        [0.2, 0.01],
        [0.8, 0.0075],
        [1.0, 0.0075],
    ])
    desired_kl_lower: Any = 0.0
    lr_schedule_scale_factor: float = 1.05
    lr_schedule_min: float = 1e-7
    lr_schedule_max: float = 1e-3

    clip_param: float = 0.2

    entropy_coef_start: float = 0.005
    entropy_coef_end: float = 0.002

    init_noise_scale: float = 1.0  # initial std for actor
    init_noise_scale_overrides: Dict[str, float] = field(default_factory=dict)  # regex map overrides
    load_noise_scale: float | None = None  # multiplier on std loaded from checkpoint

    latent_dim: int = 256

    # distillation
    reg_lambda: float = 0.2  # weight of priv-feature alignment
    # misc
    layer_norm: Union[str, None] = "before"
    value_norm: bool = False

    # phase switch
    phase: str = "train"  # train | finetune | adapt
    vecnorm: Union[str, None] = None
    symmetry_enabled: bool = True

    # I/O keys
    in_keys: List[str] = field(
        default_factory=lambda: [
            OBS_KEY,
            OBS_PRIV_KEY,
            CRITIC_PRIV_KEY,
        ]
    )

    command_modes: Union[List[int], None] = None
    checkpoint_path: Union[str, None] = None


cs = ConfigStore.instance()
cs.store("ppo_train", node=PPOConfig(phase="train", vecnorm="train", entropy_coef_start=0.01, entropy_coef_end=0.005), group="algo")
cs.store("ppo_adapt", node=PPOConfig(phase="adapt", vecnorm="eval", train_every=16), group="algo")
cs.store("ppo_finetune", node=PPOConfig(phase="finetune", vecnorm="eval", entropy_coef_start=0.005, entropy_coef_end=0.0025), group="algo")


class PPOPolicy(TensorDictModuleBase):
    # ------------------------------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------------------------------ #
    def __init__(
        self,
        cfg: PPOConfig,
        observation_spec: Composite,
        action_spec: Composite,
        reward_spec: TensorSpec,
        device: str = "cuda:0",
        env = None
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.observation_spec = observation_spec
        assert cfg.phase in {"train", "finetune", "adapt"}

        self.entropy_coef = cfg.entropy_coef_start
        self.clip_param = cfg.clip_param
        self.action_dim = action_spec.shape[-1]
        self.action_manager = env.action_manager
        self.joint_names = env.action_manager.joint_names
        init_noise_scale = self._resolve_init_noise_scale()
        self._init_noise_scale_max = torch.tensor(init_noise_scale, device=device, dtype=torch.float32)
        self.gae = GAE(0.99, 0.95)
        self.reg_lambda = 0.0  # will be annealed
        self.num_minibatches = cfg.num_minibatches
        self.progress = 0.0

        self.reward_groups = list(env.cfg.reward.keys())

        if cfg.value_norm:
            value_norm_cls = ValueNorm1
        else:
            value_norm_cls = ValueNormFake
        self.value_norm = value_norm_cls(input_shape=1).to(self.device)

        fake_td = observation_spec.zero().to(device)

        # ---------------------------------------------------------------------------- private encoder
        self.encoder_priv = Seq(
            Mod(nn.Sequential(make_mlp([512]), nn.LazyLinear(self.cfg.latent_dim)), [OBS_PRIV_KEY], ["priv_feature"]),
        ).to(device)

        # ---------------------------------------------------------------------------- state estimator (student)
        self.adapt_module = Mod(
            nn.Sequential(
                make_mlp([1024, 512]),
                nn.LazyLinear(self.cfg.latent_dim),
            ),
            [OBS_KEY],
            ["priv_pred"],
        ).to(device)
        # ---------------------------------------------------------------------------- actor(s)
        actor_in_keys_train = [OBS_KEY, "priv_feature"]
        actor_in_keys_adapt = [OBS_KEY, "priv_pred"]

        def build_actor(in_keys):
            return ProbabilisticActor(
                module=Seq(
                    CatTensors(in_keys, "_actor_inp", del_keys=False, sort=False),
                    Mod(make_mlp([1024, 1024, 512]), ["_actor_inp"], ["_actor_feature"]),
                    Mod(Actor(self.action_dim, init_noise_scale=init_noise_scale, load_noise_scale=self.cfg.load_noise_scale), ["_actor_feature"], ["loc", "scale"]),
                ),
                in_keys=["loc", "scale"],
                out_keys=[ACTION_KEY],
                distribution_class=IndependentNormal,
                return_log_prob=True,
            ).to(device)

        self.actor_teacher = build_actor(actor_in_keys_train)
        self.actor_student = build_actor(actor_in_keys_adapt)

        # ---------------------------------------------------------------------------- critic (shared)
        self.critic = Seq(
            CatTensors([OBS_KEY, OBS_PRIV_KEY, CRITIC_PRIV_KEY], "_critic_inp", del_keys=False),
            Mod(CriticNet([1024, 512, 512]), ["_critic_inp"], ["state_value"]),
        ).to(device)

        # ---------------------------------------------------------------------------- lazy init pass
        with torch.device(device):
            fake_td["is_init"] = torch.ones(fake_td.shape[0], 1, dtype=torch.bool)
        self.encoder_priv(fake_td)
        self.adapt_module(fake_td)
        self.actor_teacher(fake_td)
        self.actor_student(fake_td)
        self.critic(fake_td)

        # init weights (orthogonal for MLPS/linear)
        def ortho_(m):
            if isinstance(m, nn.Linear):
                weight = torch.empty_like(m.weight, device="cpu")
                nn.init.orthogonal_(weight, gain=0.01)
                with torch.no_grad():
                    m.weight.copy_(weight.to(device=m.weight.device, dtype=m.weight.dtype))
                    nn.init.zeros_(m.bias)

        self.apply(ortho_)

        self.world_size = 1
        self.num_updates = 0
        if aa.is_distributed():
            self.world_size = aa.get_world_size()
            self._wrap_ddp(local_rank=aa.get_local_rank())

        # ---------------------------------------------------------------------------- optimisers
        self.opt_teacher = build_optimizer(
            params=list(self.actor_teacher.parameters()) + list(self.encoder_priv.parameters()),
            optimizer=self.cfg.optimizer,
            lr=self.cfg.actor_start_lr,
            weight_decay=self.cfg.optimizer_weight_decay,
            adamw_only_params=collect_adamw_only_params(
                self._unwrap_module(self.actor_teacher), Actor, expected_modules=1
            ),
        )
        self.opt_student = build_optimizer(
            params=self.actor_student.parameters(),
            optimizer=self.cfg.optimizer,
            lr=self.cfg.actor_start_lr,
            weight_decay=self.cfg.optimizer_weight_decay,
            adamw_only_params=collect_adamw_only_params(
                self._unwrap_module(self.actor_student), Actor, expected_modules=1
            ),
        )
        self.opt_critic = build_optimizer(
            params=self.critic.parameters(),
            optimizer=self.cfg.optimizer,
            lr=self.cfg.critic_lr,
            weight_decay=0.0,
            adamw_only_params=collect_adamw_only_params(
                self._unwrap_module(self.critic), CriticNet, expected_modules=1
            ),
        )
        self.opt_estimator = build_optimizer(
            params=self.adapt_module.parameters(),
            optimizer=self.cfg.optimizer,
            lr=self.cfg.critic_lr,
            weight_decay=0.0,
        )

        self.use_symmetry_ppo = bool(getattr(self.cfg, "symmetry_enabled", True))
        aa.print(f"use_symmetry_ppo={self.use_symmetry_ppo}")
        if self.use_symmetry_ppo:
            self.obs_transform = env.observation_funcs[OBS_KEY].symmetry_transforms().to(self.device)
            self.obs_priv_transform = env.observation_funcs[OBS_PRIV_KEY].symmetry_transforms().to(self.device)
            self.critic_priv_transform = env.observation_funcs[CRITIC_PRIV_KEY].symmetry_transforms().to(self.device)
            self.act_transform = env.action_manager.symmetry_transforms().to(self.device)
        else:
            self.obs_transform = None
            self.obs_priv_transform = None
            self.critic_priv_transform = None
            self.act_transform = None

    # ------------------------------------------------------------------------------------------ #
    # Setup Helpers
    # ------------------------------------------------------------------------------------------ #
    def _wrap_ddp(self, local_rank: int):
        ddp_kwargs = dict(device_ids=[local_rank], output_device=local_rank,
                        broadcast_buffers=True, find_unused_parameters=False)

        self.actor_teacher = DDP(self.actor_teacher, **ddp_kwargs)
        self.actor_student = DDP(self.actor_student, **ddp_kwargs)
        self.encoder_priv  = DDP(self.encoder_priv,  **ddp_kwargs)
        self.critic        = DDP(self.critic,        **ddp_kwargs)
        self.adapt_module  = DDP(self.adapt_module,  **ddp_kwargs)

    @staticmethod
    def _unwrap_module(module):
        return module.module if isinstance(module, DDP) else module

    def broadcast_parameters(self, extra_modules=[]):
        info = {}
        if self.num_updates % 32 == 0:
            update_list = [self.value_norm] + extra_modules
            if aa.is_distributed():
                info.update(self._ddp_param_consistency_info())
                for m in update_list:
                    for p in m.parameters():
                        dist.broadcast(p, src=0)
                    for p in m.buffers():
                        dist.broadcast(p, src=0)
        return info

    @staticmethod
    def _param_signature(module: nn.Module, device: str) -> torch.Tensor:
        signature = torch.zeros(4, device=device, dtype=torch.float64)
        with torch.no_grad():
            for param in module.parameters():
                data = param.detach().reshape(-1).to(torch.float64)
                if not data.numel():
                    continue
                weights = torch.linspace(0.5, 1.5, data.numel(), device=device, dtype=torch.float64)
                signature[0] += data.sum()
                signature[1] += data.square().sum()
                signature[2] = torch.maximum(signature[2], data.abs().max())
                signature[3] += (data * weights).sum()
        return signature

    def _ddp_param_consistency_info(self) -> dict[str, float]:
        if not aa.is_distributed():
            return {}

        modules = {
            "actor_teacher": self.actor_teacher,
            "actor_student": self.actor_student,
            "encoder_priv": self.encoder_priv,
            "critic": self.critic,
            "adapt_module": self.adapt_module,
        }
        info = {}
        max_gap = 0.0
        for name, module in modules.items():
            signature = self._param_signature(self._unwrap_module(module), self.device)
            sig_max = signature.clone()
            sig_min = signature.clone()
            dist.all_reduce(sig_max, op=dist.ReduceOp.MAX)
            dist.all_reduce(sig_min, op=dist.ReduceOp.MIN)
            gap = (sig_max - sig_min).abs()
            module_gap = float(gap.max().item())
            max_gap = max(max_gap, module_gap)
            info[f"ddp_param/{name}_sum_gap"] = float(gap[0].item())
            info[f"ddp_param/{name}_sqsum_gap"] = float(gap[1].item())
            info[f"ddp_param/{name}_absmax_gap"] = float(gap[2].item())
            info[f"ddp_param/{name}_weighted_sum_gap"] = float(gap[3].item())
            info[f"ddp_param/{name}_max_gap"] = module_gap
        info["ddp_param/max_gap"] = max_gap
        return info

    def _resolve_init_noise_scale(self):
        base_scale = float(self.cfg.init_noise_scale)
        overrides = getattr(self.cfg, "init_noise_scale_overrides", None) or {}
        overrides = dict(overrides)
        if not overrides:
            return base_scale

        scales = [base_scale] * self.action_dim
        joint_ids, _, joint_scales = self.action_manager.resolve(
            overrides, names=self.joint_names
        )
        for idx, scale in zip(joint_ids, joint_scales):
            scales[idx] = float(scale)
        return scales

    # ------------------------------------------------------------------------------------------ #
    # Runtime Interface
    # ------------------------------------------------------------------------------------------ #
    @staticmethod
    def _get_optimizer_lr(opt) -> float:
        lrs = [float(group["lr"]) for group in opt.param_groups]
        if len(lrs) == 0:
            raise RuntimeError("optimizer has no param_groups")
        return sum(lrs) / len(lrs)

    @staticmethod
    def _set_optimizer_lr(opt, lr: float):
        lr = float(lr)
        for param_group in opt.param_groups:
            param_group["lr"] = lr

    def get_lr(self, target: str | None = None):
        lrs = {
            "actor": 0.5 * (
                self._get_optimizer_lr(self.opt_teacher) + self._get_optimizer_lr(self.opt_student)
            ),
            "critic": self._get_optimizer_lr(self.opt_critic),
            "estimator": self._get_optimizer_lr(self.opt_estimator),
        }
        if target is None:
            return lrs
        if target not in lrs:
            raise ValueError(f"unsupported lr target: {target}")
        return lrs[target]

    def set_lr(self, target: str, lr: float):
        if target == "actor":
            self._set_optimizer_lr(self.opt_teacher, lr)
            self._set_optimizer_lr(self.opt_student, lr)
            return
        if target == "critic":
            self._set_optimizer_lr(self.opt_critic, lr)
            return
        if target == "estimator":
            self._set_optimizer_lr(self.opt_estimator, lr)
            return
        raise ValueError(f"unsupported lr target: {target}")

    def do_lr_schedule(self, kl):
        schedule_progress = float(self.progress if self.cfg.phase == "train" else 1.0)
        kl_upper = _schedule_value(self.cfg.desired_kl_upper, schedule_progress)
        kl_lower = _schedule_value(self.cfg.desired_kl_lower, schedule_progress)

        new_lr = self.get_lr("actor")
        if kl > kl_upper:
            new_lr = max(self.cfg.lr_schedule_min, new_lr / self.cfg.lr_schedule_scale_factor)
        elif 0.0 < kl < kl_lower:
            new_lr = min(self.cfg.lr_schedule_max, new_lr * self.cfg.lr_schedule_scale_factor)

        if aa.is_distributed():
            lr_tensor = torch.tensor(new_lr, device=self.device)
            dist.all_reduce(lr_tensor, op=dist.ReduceOp.SUM)
            new_lr = (lr_tensor / self.world_size).item()

        self.set_lr("actor", new_lr)
        return kl_upper, kl_lower, schedule_progress

    def make_tensordict_primer(self):
        return None

    def get_rollout_policy(self, mode: str = ""):
        modules = []
        if self.cfg.phase == "train":
            modules += [self.encoder_priv, self.actor_teacher]
        elif self.cfg.phase == "finetune":
            modules += [self.adapt_module]
            modules += [self.actor_student]
        elif self.cfg.phase == "adapt":
            modules += [self.adapt_module]
            modules += [self.actor_student]

        policy = Seq(*modules)
        return policy

    def step_schedule(self, progress: float, iter: int):
        self.reg_lambda = progress * self.cfg.reg_lambda
        start = self.cfg.entropy_coef_start
        end = self.cfg.entropy_coef_end
        # exponential decay from start to end based on progress in [0,1]
        self.entropy_coef = start * (end / start) ** progress
        self.progress = progress

    # ------------------------------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------------------------------ #
    def train_op(self, td: TensorDict, vecnorm):
        """One optimisation step on a batched rollout tensor-dict."""
        info = {}
        if self.cfg.phase == "train":
            info.update(self._train_ppo(td, mode="teacher"))
            info.update(self._train_estimator(td))
        elif self.cfg.phase == "finetune":
            info.update(self._train_ppo(td, mode="student"))
        else:  # adapt
            info.update(self._train_estimator(td))
        self.num_updates += 1
        info.update(self.broadcast_parameters(extra_modules=[vecnorm]))
        return info

    # ------------------------------------------------------------------------------------------ #
    # Minibatch Preparation
    # ------------------------------------------------------------------------------------------ #
    def _prepare_mb(self, mb: TensorDict, include_adv_ret: bool, apply_symmetry: bool = False) -> tuple[TensorDict, torch.Tensor]:
        keys = [OBS_KEY, OBS_PRIV_KEY, CRITIC_PRIV_KEY, "is_init"]
        if include_adv_ret:
            keys.extend(["adv", "ret"])
        mb = mb.select(*keys)
        if apply_symmetry and self.use_symmetry_ppo:
            mb_sym = mb.clone()
            mb_sym[OBS_KEY] = self.obs_transform(mb_sym[OBS_KEY])
            mb_sym[OBS_PRIV_KEY] = self.obs_priv_transform(mb_sym[OBS_PRIV_KEY])
            mb_sym[CRITIC_PRIV_KEY] = self.critic_priv_transform(mb_sym[CRITIC_PRIV_KEY])
            mb = torch.cat([mb, mb_sym], dim=0)
        valid = ~mb["is_init"]
        return mb, valid

    # ------------------------------------------------------------------------------------------ #
    # PPO Update
    # ------------------------------------------------------------------------------------------ #
    def _train_ppo(self, td, mode: str):
        infos = []
        self._compute_advantage(td, self.critic, self.gae, self.value_norm, REWARD_KEY=REWARD_KEY, TERM_KEY=TERM_KEY, DONE_KEY=DONE_KEY)
        adv_normalize(td["adv"], ~td["is_init"])

        for _ in range(self.cfg.ppo_epochs):
            for mb in make_batch(td, self.num_minibatches):
                infos.append(TensorDict(self._update_ppo_batch(mb, mode=mode), []))
        info = {k: v.mean().item() for k, v in torch.stack(infos).items()}

        with torch.no_grad():
            actor = self.actor_teacher if mode == "teacher" else self.actor_student
            action_std = self._get_actor_std(actor)
            if action_std is None:
                raise RuntimeError("failed to locate actor_std for logging")
            for joint_name, std in zip(self.joint_names, action_std):
                info[f"actor_std/{joint_name}"] = std
            info["actor_std/mean"] = action_std.mean()

        kl = info["actor/kl"]
        kl_upper, kl_lower, kl_schedule_progress = self.do_lr_schedule(kl)
        lrs = self.get_lr()
        info["lr"] = lrs["actor"]
        info["lr/actor"] = lrs["actor"]
        info["lr/critic"] = lrs["critic"]
        info["lr/estimator"] = lrs["estimator"]
        info["lr/kl_upper"] = torch.tensor(kl_upper, device=self.device)
        info["lr/kl_lower"] = torch.tensor(kl_lower, device=self.device)
        info["lr/kl_schedule_progress"] = torch.tensor(kl_schedule_progress, device=self.device)

        neg_reward_ratio = (td[REWARD_KEY] <= 0.0).float().mean().item()
        info["critic/neg_reward_ratio"] = neg_reward_ratio

        return info

    def _update_ppo_batch(self, mb, mode: str):
        if mode == "teacher":
            actor = self.actor_teacher
            encoder = self.encoder_priv
            update_encoder = True
            opt_actor = self.opt_teacher
        elif mode == "student":
            actor = self.actor_student
            encoder = self.adapt_module
            update_encoder = False
            opt_actor = self.opt_student
        else:
            raise ValueError(f"unsupported ppo mode: {mode}")

        bsize = mb.shape[0]
        loc_old, scale_old = mb["loc"], mb["scale"]
        action_old = mb["action"]
        logp_old = mb["action_log_prob"]

        mb, valid = self._prepare_mb(mb, include_adv_ret=True, apply_symmetry=True)

        if encoder is not None:
            if update_encoder:
                encoder(mb)
            else:
                with torch.no_grad():
                    encoder(mb)

        actor(mb)

        dist = IndependentNormal(mb["loc"][:bsize], mb["scale"][:bsize])
        logp = dist.log_prob(action_old)
        entropy = dist.entropy().mean()

        ratio = torch.exp(logp - logp_old).unsqueeze(-1)
        surr1 = mb["adv"][:bsize] * ratio
        surr2 = mb["adv"][:bsize] * ratio.clamp(1 - self.clip_param, 1 + self.clip_param)
        policy_loss = - torch.mean(torch.min(surr1, surr2) * valid[:bsize])
        entropy_loss = - self.entropy_coef * entropy

        values = self.critic(mb)["state_value"]
        value_loss = F.mse_loss(mb["ret"], values, reduction="none")
        value_loss = (value_loss * valid).mean(dim=0)

        if self.cfg.phase == "train":
            if "priv_pred" not in mb.keys():
                with torch.no_grad():
                    self.adapt_module(mb)
            reg_loss = F.mse_loss(mb["priv_pred"], mb["priv_feature"], reduction="none")
            reg_loss = self.reg_lambda * torch.mean(reg_loss * valid)
        else:
            reg_loss = 0.0

        if self.use_symmetry_ppo:
            symmetry_loss_loc = F.mse_loss(mb["loc"][:bsize], self.act_transform(mb["loc"][bsize:])) * 0.2
            symmetry_loss_std = F.mse_loss(
                mb["scale"][:bsize],
                self.act_transform(mb["scale"][bsize:], sign=False),
            ) * 10
        else:
            symmetry_loss_loc = torch.zeros((), device=self.device)
            symmetry_loss_std = torch.zeros((), device=self.device)

        loss = policy_loss + entropy_loss + value_loss.mean() + reg_loss + symmetry_loss_loc + symmetry_loss_std

        # do optimisation step
        opt_actor.zero_grad()
        self.opt_critic.zero_grad()

        loss.backward()

        if update_encoder and encoder is not None:
            encoder_grad_norm = nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        else:
            encoder_grad_norm = torch.tensor(0.0, device=self.device)

        actor_grad_norm = nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        opt_actor.step()
        self._clamp_actor_std(actor)

        critic_grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)

        self.opt_critic.step()

        with torch.no_grad():
            explained_var = 1 - value_loss / (mb["ret"] * valid).var(dim=0)
            clipfrac = ((ratio - 1.0).abs() > self.clip_param).float().mean()
            loc, scale = mb["loc"][:bsize], mb["scale"][:bsize]
            kl = torch.sum(
                torch.log(scale) - torch.log(scale_old)
                + (torch.square(scale_old) + torch.square(loc_old - loc)) / (2.0 * torch.square(scale))
                - 0.5,
                axis=-1,
            ).mean()

        info = {
            "actor/policy_loss": policy_loss.detach(),
            "actor/entropy": entropy.detach(),
            "adapt/reg_loss": reg_loss if isinstance(reg_loss, torch.Tensor) else torch.tensor(0.0),
            "actor/actor_grad_norm": actor_grad_norm,
            "action/encoder_grad_norm": encoder_grad_norm,
            "actor/clamp_ratio": clipfrac,
            "critic/critic_grad_norm": critic_grad_norm,
            "actor/kl": kl.detach(),
            "actor/symmetry_loss_loc": symmetry_loss_loc.detach(),
            "actor/symmetry_loss_std": symmetry_loss_std.detach(),
        }

        info["critic/explained_var"] = explained_var.mean().detach()
        info["critic/value_loss"] = value_loss.mean().detach()

        return info

    # ------------------------------------------------------------------------------------------ #
    # Actor Utils
    # ------------------------------------------------------------------------------------------ #
    def _get_actor_std(self, actor):
        base = self._unwrap_module(actor)
        for module in base.modules():
            if isinstance(module, Actor) and hasattr(module, "actor_std"):
                return module.actor_std
        return None

    def _clamp_actor_std(self, actor):
        actor_std = self._get_actor_std(actor)
        if actor_std is None:
            return
        actor_std.data = torch.minimum(actor_std.data, self._init_noise_scale_max)

    # ------------------------------------------------------------------------------------------ #
    # Estimator Update
    # ------------------------------------------------------------------------------------------ #
    def _train_estimator(self, td):
        infos = []
        
        for _ in range(2):
            for mb in make_batch(td, self.num_minibatches, self.cfg.train_every):
                mb, valid = self._prepare_mb(mb, include_adv_ret=False, apply_symmetry=True)

                with torch.no_grad():
                    self.encoder_priv(mb)
                self.adapt_module(mb)

                loss = torch.mean(F.mse_loss(mb["priv_pred"], mb["priv_feature"], reduction="none") * (valid))

                self.opt_estimator.zero_grad()
                loss.backward()
                self.opt_estimator.step()

                infos.append(TensorDict({"adapt/estimator_loss": loss.detach()}, []))

        return {k: v.mean().item() for k, v in torch.stack(infos).items()}

    # ------------------------------------------------------------------------------------------ #
    # Statistics
    # ------------------------------------------------------------------------------------------ #
    @staticmethod
    @torch.compile
    @torch.no_grad()
    def _compute_advantage(td, critic, gae, value_norm, REWARD_KEY="reward", TERM_KEY="term", DONE_KEY="done"):
        keys = td.keys(True, True)
        if not ("state_value" in keys and ("next", "state_value") in keys):
            with td.view(-1) as flat:
                critic(flat)
                critic(flat["next"])

        v = td["state_value"]
        v_next = td["next", "state_value"]

        rewards = td[REWARD_KEY].sum(dim=-1, keepdim=True).clamp_min(0.)

        adv, ret = gae(
            rewards,
            td[TERM_KEY],
            td[DONE_KEY],
            value_norm.denormalize(v),
            value_norm.denormalize(v_next),
        )

        value_norm.update(ret)
        td["adv"], td["ret"] = adv, value_norm.normalize(ret)

    # ------------------------------------------------------------------------------------------ #
    # Checkpoint IO
    # ------------------------------------------------------------------------------------------ #
    def state_dict(self):
        state = OrderedDict()
        for n, m in self.named_children():
            state[n] = self._unwrap_module(m).state_dict()

        state["last_phase"] = self.cfg.phase

        state["_meta"] = {
            "lrs": self.get_lr(),
            "entropy_coef": getattr(self, "entropy_coef", self.cfg.entropy_coef_start),
            "reg_lambda": getattr(self, "reg_lambda", 0.0),
            "progress": getattr(self, "progress", 0.0),
            "num_updates": getattr(self, "num_updates", 0),
            "world_size": getattr(self, "world_size", 1),
        }

        return state

    def load_state_dict(self, state_dict, strict=True):
        for n, m in self.named_children():
            try:
                self._unwrap_module(m).load_state_dict(state_dict.get(n, {}), strict=strict)
            except Exception as e:
                warnings.warn(f"Failed to load {n}: {e}")

        last_phase = state_dict.get("last_phase", "train")

        # Initialize student actor from teacher if starting from a 'train' phase checkpoint
        if last_phase == "train":
            warnings.warn("Last phase was 'train'. Performing a hard copy from `actor_teacher` to `actor_student`.")
            src = self._unwrap_module(self.actor_teacher)
            dst = self._unwrap_module(self.actor_student)
            hard_copy_(src, dst)

        meta = state_dict.get("_meta", {})
        saved_lrs = meta.get("lrs", None)
        if saved_lrs is not None:
            for target, lr in saved_lrs.items():
                self.set_lr(target, lr)
        if state_dict["last_phase"] == self.cfg.phase:
            self.entropy_coef = meta.get("entropy_coef", self.entropy_coef)
            self.reg_lambda   = meta.get("reg_lambda", self.reg_lambda)
            self.progress     = meta.get("progress", self.progress)
            self.num_updates  = meta.get("num_updates", self.num_updates)
