import torch

from mjlab.managers.event_manager import RecomputeLevel
from ..component import EnvBoundComponent, RegisteredComponent, method_marker, resolve_wrapper_kwargs


class Randomization(RegisteredComponent, EnvBoundComponent):
    registry: dict[str, type["Randomization"]] = {}
    component_kind = "Randomization"
    _RECOMPUTE_DERIVED_FIELDS = {
        RecomputeLevel.none: (),
        RecomputeLevel.set_const_fixed: ("body_subtreemass",),
        RecomputeLevel.set_const_0: (
            "dof_invweight0",
            "body_invweight0",
            "tendon_length0",
            "tendon_invweight0",
        ),
        RecomputeLevel.set_const: (
            "body_subtreemass",
            "dof_invweight0",
            "body_invweight0",
            "tendon_length0",
            "tendon_invweight0",
        ),
    }

    def __init__(self, env):
        super().__init__(env)

    def has_observation(self) -> bool:
        return False

    def observe(self, **kwargs) -> torch.Tensor:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not expose a domain observation."
        )

    def observe_sym(self, **kwargs):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not expose a symmetry transform for domain observation."
        )

    def ensure_model_fields_expanded(self, *fields: str):
        missing = tuple(f for f in fields if f not in self.env.sim.expanded_fields)
        if missing:
            self.env.sim.expand_model_fields(missing)

    def ensure_recompute_fields_expanded(self, level: RecomputeLevel):
        self.ensure_model_fields_expanded(*self._RECOMPUTE_DERIVED_FIELDS[level])


_RANDOMIZATION_HOOKS = ("startup", "reset", "step", "update", "debug_draw")
_RANDOMIZATION_RUNTIME_KEYS = {
    "startup": {"env"},
    "reset": {"env", "env_ids"},
    "step": {"env", "substep"},
    "update": {"env"},
    "debug_draw": {"env"},
}


def _validate_randomization_hook(hook: str):
    if hook not in _RANDOMIZATION_HOOKS:
        raise ValueError(
            f"Invalid randomization hook '{hook}', expected one of {_RANDOMIZATION_HOOKS}."
        )


def randomization_wrapper(
    func,
    *,
    hook: str = "reset",
    error_name: str | None = None,
    bind_env: bool = False,
    register: bool = False,
    name: str | None = None,
):
    """Adapt a callable into a Randomization instance bound to one lifecycle hook."""
    _validate_randomization_hook(hook)
    error_name = error_name or getattr(func, "__name__", func.__class__.__name__)
    runtime_keys = _RANDOMIZATION_RUNTIME_KEYS[hook]

    class RandomizationWrapper(Randomization, register=register, name=name):
        def __init__(self, env, **params):
            super().__init__(env)
            self.params = params
            self._func_kwargs = resolve_wrapper_kwargs(
                func,
                params,
                kind="randomization",
                error_name=error_name,
                runtime_keys=runtime_keys,
            )

        def _call(self, **runtime_kwargs):
            kwargs = dict(self._func_kwargs)
            kwargs.update(runtime_kwargs)
            if bind_env:
                kwargs["env"] = self.env
            return func(**kwargs)

        def startup(self):
            if hook == "startup":
                self._call()

        def reset(self, env_ids: torch.Tensor):
            if hook == "reset":
                self._call(env_ids=env_ids)

        def step(self, substep):
            if hook == "step":
                self._call(substep=substep)

        def update(self):
            if hook == "update":
                self._call()

        def debug_draw(self):
            if hook == "debug_draw":
                self._call()

    return RandomizationWrapper

def randomization_method(*, hook: str = "reset"):
    """Mark a Command method so _Env wraps it as a randomization hook."""
    _validate_randomization_hook(hook)
    return method_marker("_mdp_randomization_method", hook)
