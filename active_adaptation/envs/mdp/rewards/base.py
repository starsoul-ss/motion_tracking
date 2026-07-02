import abc
from typing import Callable

import torch

from ..component import EnvBoundComponent, RegisteredComponent, method_marker, resolve_wrapper_kwargs


class Reward(RegisteredComponent, EnvBoundComponent):
    registry: dict[str, type["Reward"]] = {}
    component_kind = "Reward"

    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env)
        self.weight = weight
        self.enabled = enabled

    def __call__(self) -> torch.Tensor:
        result = self.compute()
        if isinstance(result, torch.Tensor):
            rew, count = result, result.numel()
        elif isinstance(result, tuple):
            rew, is_active = result
            rew = rew * is_active.float()
            count = is_active.sum().item()
        return self.weight * rew, count

    @abc.abstractmethod
    def compute(self) -> torch.Tensor:
        raise NotImplementedError

    def debug_draw(self):
        pass

def reward_wrapper(func: Callable[[], torch.Tensor], *, error_name: str | None = None):
    """Adapt a callable into a Reward instance at env construction time."""
    error_name = error_name or getattr(func, "__name__", func.__class__.__name__)

    class RewardWrapper(Reward, register=False):
        def __init__(self, env, weight: float, enabled: bool = True, **params):
            super().__init__(env, weight=weight, enabled=enabled)
            self.params = params
            self._func_kwargs = resolve_wrapper_kwargs(
                func,
                params,
                kind="reward",
                error_name=error_name,
            )

        def compute(self):
            return func(**self._func_kwargs)

    return RewardWrapper


def reward_method(func):
    """Mark a Command method so _Env wraps it as a reward."""
    return method_marker("_mdp_reward_method")(func)
