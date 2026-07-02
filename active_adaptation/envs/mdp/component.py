import inspect
from typing import TYPE_CHECKING, Callable

import torch

if TYPE_CHECKING:
    from active_adaptation.envs.base import _Env


class RegisteredComponent:
    registry: dict[str, type] = {}
    component_kind = "Component"

    def __init_subclass__(cls, *, register: bool = True, name: str | None = None, **kwargs):
        super().__init_subclass__(**kwargs)
        if register:
            key = name or cls.__name__
            if key in cls.registry:
                raise ValueError(f"{cls.component_kind} '{key}' already registered.")
            cls.registry[key] = cls


class EnvBoundComponent:
    def __init__(self, env):
        self.env: _Env = env
        self.scene = env.scene
        self.sim = env.sim
        self.command_manager = getattr(env, "command_manager", None)
        self.action_manager = getattr(env, "action_manager", None)

    @property
    def num_envs(self):
        return self.env.num_envs

    @property
    def device(self):
        return self.env.device

    def startup(self):
        pass

    def post_step(self, substep: int):
        pass

    def step(self, substep: int):
        pass

    def update(self):
        pass

    def reset(self, env_ids: torch.Tensor):
        pass

    def debug_draw(self):
        pass


def select_kwargs(fn: Callable, params: dict, runtime_keys: set[str] | None = None):
    runtime_keys = runtime_keys or set()
    sig = inspect.signature(fn)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return dict(params), True
    valid_keys = {
        name
        for name, p in sig.parameters.items()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    yaml_keys = valid_keys - runtime_keys
    return {k: v for k, v in params.items() if k in yaml_keys}, False


def resolve_wrapper_kwargs(
    fn: Callable,
    params: dict,
    *,
    kind: str,
    error_name: str,
    runtime_keys: set[str] | None = None,
):
    func_kwargs, accepts_all = select_kwargs(fn, params, runtime_keys=runtime_keys)
    if not accepts_all:
        unknown = set(params.keys()) - set(func_kwargs.keys())
        if unknown:
            raise ValueError(
                f"Unknown YAML params for wrapped {kind} '{error_name}': {sorted(unknown)}"
            )
    return func_kwargs


def method_marker(attr_name: str, value=True):
    def decorator(func):
        setattr(func, attr_name, value)
        return func

    return decorator
