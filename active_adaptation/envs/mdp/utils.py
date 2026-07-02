import re
from collections.abc import Mapping
from typing import Dict, Union

import torch
from active_adaptation.utils.math import quat_from_angle_axis, quat_mul


def random_noise(x: torch.Tensor, std: float | torch.Tensor | None):
    if std is None:
        return x
    if not torch.is_tensor(std):
        if std <= 0.0:
            return x
        std = float(std)
    else:
        std = std.to(device=x.device, dtype=x.dtype)
        if not bool((std > 0.0).any()):
            return x
    return x + (torch.rand_like(x) * 2.0 - 1.0) * std


def clamp_norm(
    x: torch.Tensor,
    max: float | torch.Tensor = torch.inf,
    *,
    min: float | torch.Tensor = 0.0,
    dim: int = -1,
    eps: float = 1e-6,
):
    x_norm = x.norm(dim=dim, keepdim=True).clamp_min(eps)
    min_t = torch.as_tensor(min, device=x.device, dtype=x.dtype)
    max_t = torch.as_tensor(max, device=x.device, dtype=x.dtype)
    x = torch.where(x_norm < min_t, x / x_norm * min_t, x)
    x = torch.where(x_norm > max_t, x / x_norm * max_t, x)
    return x


def rand_points_disk(
    N: int,
    M: int,
    r_max: float = 1.0,
    *,
    device=None,
    dtype=torch.float32,
    generator: torch.Generator | None = None,
):
    u = torch.rand((N, M, 1), device=device, dtype=dtype, generator=generator) * r_max
    direction = torch.randn((N, M, 2), device=device, dtype=dtype, generator=generator)
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return direction * u


def _rand_unit_vectors(shape: tuple[int, ...], *, device=None, dtype=torch.float32):
    vec = torch.randn(shape, device=device, dtype=dtype)
    return vec / vec.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def add_spherical_noise(x: torch.Tensor, noise_std: float) -> torch.Tensor:
    if noise_std <= 0.0:
        return x
    if x.shape[-1] != 3:
        raise ValueError(f"add_spherical_noise expects last dim 3, got shape {tuple(x.shape)}")
    direction = _rand_unit_vectors(tuple(x.shape), device=x.device, dtype=x.dtype)
    radius = torch.rand(tuple(x.shape[:-1]) + (1,), device=x.device, dtype=x.dtype) * float(noise_std)
    return x + direction * radius


def perturb_quaternion(quat: torch.Tensor, angle_std: float) -> torch.Tensor:
    if angle_std <= 0.0:
        return quat
    if quat.shape[-1] != 4:
        raise ValueError(f"perturb_quaternion expects last dim 4, got shape {tuple(quat.shape)}")
    axis = _rand_unit_vectors(tuple(quat.shape[:-1]) + (3,), device=quat.device, dtype=quat.dtype)
    angle = (torch.rand(quat.shape[:-1], device=quat.device, dtype=quat.dtype) * 2.0 - 1.0) * float(angle_std)
    delta = quat_from_angle_axis(angle, axis)
    return quat_mul(delta, quat)


def resolve_named_std(
    spec: Union[float, Dict[str, float], Mapping[str, float], None],
    names: list[str],
    device,
    dtype,
    context: str,
):
    if spec is None:
        return None
    if isinstance(spec, Mapping):
        out = torch.zeros(len(names), device=device, dtype=dtype)
        for pattern, value in spec.items():
            matched = False
            for i, name in enumerate(names):
                if re.match(pattern, name):
                    out[i] = value
                    matched = True
            if not matched:
                raise ValueError(f"{context}: pattern '{pattern}' matched no names.")
        return out
    return torch.full((len(names),), float(spec), device=device, dtype=dtype)


def sample_uniform(size, low: float, high: float, device: torch.device = "cpu"):
    return torch.rand(size, device=device) * (high - low) + low


def uniform(low: torch.Tensor, high: torch.Tensor):
    r = torch.rand_like(low)
    return low + r * (high - low)


def uniform_like(x: torch.Tensor, low: torch.Tensor, high: torch.Tensor):
    r = torch.rand_like(x)
    return low + r * (high - low)


def log_uniform(low: torch.Tensor, high: torch.Tensor):
    return uniform(low.log(), high.log()).exp()


def sample_log_uniform(size, low: float, high: float, device: torch.device = "cpu"):
    low_t = torch.tensor(low, device=device, dtype=torch.float32)
    high_t = torch.tensor(high, device=device, dtype=torch.float32)
    return log_uniform(low_t, high_t).expand(size) if size == () else log_uniform(
        low_t.expand(size), high_t.expand(size)
    )
