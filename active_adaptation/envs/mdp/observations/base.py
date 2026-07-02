import abc
from typing import Callable

import torch

from ..component import (
    EnvBoundComponent,
    RegisteredComponent,
    method_marker,
    resolve_wrapper_kwargs,
    select_kwargs,
)

class Observation(RegisteredComponent, EnvBoundComponent):
    registry: dict[str, type["Observation"]] = {}
    component_kind = "Observation"

    def __init__(self, env):
        super().__init__(env)
        self.command_manager = env.command_manager

    @abc.abstractmethod
    def compute(self) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self) -> torch.Tensor:
        return self.compute()

    def symmetry_transforms(self):
        raise NotImplementedError(
            "This observation does not support symmetry transforms. "
            "Please implement the symmetry_transforms method if needed."
        )


def observation_wrapper(
    func: Callable[[], torch.Tensor],
    func_sym: Callable | None = None,
    *,
    error_name: str | None = None,
):
    """Adapt a callable into an Observation instance at env construction time."""
    error_name = error_name or getattr(func, "__name__", func.__class__.__name__)

    class ObservationWrapper(Observation, register=False):
        def __init__(self, env, **params):
            super().__init__(env)
            self.params = params
            self._func_kwargs = resolve_wrapper_kwargs(
                func,
                params,
                kind="observation",
                error_name=error_name,
            )
            if func_sym is not None:
                self._func_sym_kwargs, _ = select_kwargs(func_sym, params)
            else:
                self._func_sym_kwargs = None

        def compute(self):
            return func(**self._func_kwargs)

        def symmetry_transforms(self):
            if func_sym is None:
                raise NotImplementedError(
                    f"Wrapped observation '{error_name}' does not provide symmetry transforms."
                )
            return func_sym(**self._func_sym_kwargs)

    return ObservationWrapper


def observation_method(func):
    """Mark a Command method so _Env wraps it as an observation."""
    return method_marker("_mdp_observation_method")(func)
