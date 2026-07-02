# MIT License
# 
# Copyright (c) 2023 Botian Xu, Tsinghua University
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import torch
import torch.nn as nn
import torch.distributed as dist
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModuleBase as ModBase
import active_adaptation as aa


OBS_KEY = "policy" # ("agents", "observation")
OBS_PRIV_KEY = "priv"
CRITIC_PRIV_KEY = "priv_critic"
OBS_HIST_KEY = "policy_h"
ACTION_KEY = "action" # ("agents", "action")
REWARD_KEY = ("next", "reward") # ("agents", "reward")
# DONE_KEY = ("next", "done")
TERM_KEY = ("next", "terminated")
DONE_KEY = ("next", "done")
CMD_KEY = "command"


def make_mlp(num_units, activation=nn.Mish, norm="before", dropout=0.):
    assert norm in ("before", "after", None)
    layers = []
    for n in num_units:
        layers.append(nn.LazyLinear(n))
        if norm == "before":
            layers.append(nn.LayerNorm(n))
            layers.append(activation())
        elif norm == "after":
            layers.append(activation())
            layers.append(nn.LayerNorm(n))
        else:
            layers.append(activation())
        if dropout > 0. :
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


def make_batch(tensordict: TensorDict, num_minibatches: int, seq_len: int = -1):
    if seq_len > 1:
        N, T = tensordict.shape
        T = (T // seq_len) * seq_len
        tensordict = tensordict[:, :T].reshape(-1, seq_len)
        perm = torch.randperm(
            (tensordict.shape[0] // num_minibatches) * num_minibatches,
            device=tensordict.device,
        ).reshape(num_minibatches, -1)
        for indices in perm:
            yield tensordict[indices].clone()
    else:
        tensordict = tensordict.reshape(-1)
        perm = torch.randperm(
            (tensordict.shape[0] // num_minibatches) * num_minibatches,
            device=tensordict.device,
        ).reshape(num_minibatches, -1)
        for indices in perm:
            yield tensordict[indices].clone()


def unique_trainable_params(params):
    result = []
    seen = set()
    for p in params:
        if not p.requires_grad:
            continue
        pid = id(p)
        if pid in seen:
            continue
        seen.add(pid)
        result.append(p)
    return result

def collect_adamw_only_params(
    module: nn.Module,
    owner_type: type[nn.Module],
    *,
    expected_modules: int | None = None,
):
    params = []
    tagged_modules = 0
    for submodule in module.modules():
        if not isinstance(submodule, owner_type):
            continue
        getter = getattr(submodule, "adamw_only_parameters", None)
        if not callable(getter):
            raise RuntimeError(f"{owner_type.__name__} must implement adamw_only_parameters()")
        tagged_modules += 1
        params.extend(getter())
    if expected_modules is not None and tagged_modules != expected_modules:
        raise RuntimeError(
            f"expected {expected_modules} {owner_type.__name__} module(s), found {tagged_modules}"
        )
    return unique_trainable_params(params)


class Actor(nn.Module):
    def __init__(
        self,
        action_dim: int,
        init_noise_scale: float | torch.Tensor | list = 1.0,
        predict_std: bool = False,
        load_noise_scale: float | None = None
    ) -> None:
        super().__init__()
        self.predict_std = predict_std
        if predict_std:
            self.actor_mean = nn.LazyLinear(action_dim * 2)
        else:
            self.actor_mean = nn.LazyLinear(action_dim)
            if isinstance(init_noise_scale, torch.Tensor):
                init_std = init_noise_scale.to(dtype=torch.float32)
            elif isinstance(init_noise_scale, (list, tuple)):
                init_std = torch.tensor(init_noise_scale, dtype=torch.float32)
            else:
                init_std = torch.full((action_dim,), float(init_noise_scale), dtype=torch.float32)
            if init_std.numel() != action_dim:
                raise ValueError(
                    f"init_noise_scale length {init_std.numel()} does not match action_dim {action_dim}."
                )
            self.actor_std = nn.Parameter(init_std)
        self.scale_mapping = nn.Identity()
        self.load_noise_scale = load_noise_scale
    
    def forward(self, features: torch.Tensor):
        if self.predict_std:
            loc, scale = self.actor_mean(features).chunk(2, dim=-1)
        else:
            loc = self.actor_mean(features)
            scale = torch.ones_like(loc) * self.actor_std
        scale = self.scale_mapping(scale)
        return loc, scale
    
    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        if self.load_noise_scale is not None and hasattr(self, "actor_std"):
            print("scale actor noise std by config factor")
            self.actor_std.data.mul_(self.load_noise_scale)

    def adamw_only_parameters(self):
        return self.actor_mean.parameters()


class CriticNet(nn.Sequential):
    def __init__(self, hidden_units) -> None:
        super().__init__(
            make_mlp(hidden_units),
            nn.LazyLinear(1),
        )

    @property
    def backbone(self):
        return self[0]

    @property
    def head(self):
        return self[1]

    def adamw_only_parameters(self):
        return self.head.parameters()


class GAE(nn.Module):
    def __init__(self, gamma, lmbda):
        super().__init__()
        self.register_buffer("gamma", torch.tensor(gamma))
        self.register_buffer("lmbda", torch.tensor(lmbda))
        self.gamma: torch.Tensor
        self.lmbda: torch.Tensor
    
    def forward(
        self, 
        reward: torch.Tensor, 
        terminated: torch.Tensor,
        done: torch.Tensor, 
        value: torch.Tensor, 
        next_value: torch.Tensor,
        discount: torch.Tensor=None
    ):
        num_steps = terminated.shape[1]
        advantages = torch.zeros_like(reward)
        nonterm = 1 - terminated.float() # whether to backup value
        nondone = 1 - done.float()       # whether to backup reward
        if discount is None:
            discount = torch.ones_like(nonterm)
        gae = 0
        for step in reversed(range(num_steps)):
            next_value_t = next_value[:, step] * nonterm[:, step]
            gamma_t = discount[:, step] * self.gamma
            delta = reward[:, step] + gamma_t * next_value_t - value[:, step]
            advantages[:, step] = gae = delta + (gamma_t * self.lmbda * nondone[:, step] * gae)
        returns = advantages + value
        return advantages, returns

def hard_copy_(source_module: nn.Module, target_module: nn.Module):
    for params_source, params_target in zip(source_module.parameters(), target_module.parameters()):
        params_target.data.copy_(params_source.data)

def soft_copy_(source_module: nn.Module, target_module: nn.Module, tau: float = 0.01):
    for params_source, params_target in zip(source_module.parameters(), target_module.parameters()):
        params_target.data.lerp_(params_source.data, tau)


@torch.compile
def adv_normalize(v: torch.Tensor, mask: torch.Tensor):
    if aa.is_distributed():
        local_count = mask.sum()
        local_sum = (v * mask).sum()
        local_sum_sq = (v * v * mask).sum()

        stats = torch.stack([local_sum, local_sum_sq, local_count.float()])
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

        global_sum, global_sum_sq, global_count = stats
        global_count.clamp_min_(1)

        mean = global_sum / global_count
        var = (global_sum_sq / global_count) - (mean * mean)
        std = torch.sqrt(var.clamp(min=0.0)).clamp(min=1e-5)
    else:
        count = mask.sum().clamp_min_(1)
        sum_ = (v * mask).sum()
        sum_sq = (v * v * mask).sum()

        mean = sum_ / count
        var = (sum_sq / count) - (mean * mean)
        std = torch.sqrt(var.clamp(min=0.0)).clamp(min=1e-5)

    v[mask] = (v[mask] - mean) / std
    return v


class CatTensors(ModBase):
    def __init__(self, in_keys, out_key, del_keys=False, sort=True):
        super().__init__()
        self.in_keys = in_keys
        self.out_keys = [out_key]

        self.del_keys = del_keys
        self.sort = sort
        if self.sort:
            self.in_keys = sorted(self.in_keys)

    def forward(self, tensordict: TensorDictBase):
        out = torch.cat([tensordict.get(k) for k in self.in_keys], dim=-1)
        tensordict.set(self.out_keys[0], out)
        if self.del_keys:
            tensordict.exclude(*self.in_keys, inplace=True)
        return tensordict
