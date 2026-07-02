import abc

import torch

from ..component import EnvBoundComponent, RegisteredComponent, method_marker, resolve_wrapper_kwargs


class Termination(RegisteredComponent, EnvBoundComponent):
    registry: dict[str, type["Termination"]] = {}
    component_kind = "Termination"
    VALID_TERMINATION_TYPES = ("terminated", "truncated")

    def __init__(self, env, termination_type: str = "terminated"):
        super().__init__(env)
        self.set_termination_type(termination_type)

    @abc.abstractmethod
    def __call__(self) -> torch.Tensor:
        raise NotImplementedError

    def set_termination_type(self, termination_type: str):
        if termination_type not in self.VALID_TERMINATION_TYPES:
            raise ValueError(
                f"Invalid termination_type='{termination_type}', "
                f"expected one of {self.VALID_TERMINATION_TYPES}"
            )
        self.termination_type = termination_type

def termination_wrapper(func, *, error_name: str | None = None):
    """Adapt a callable into a Termination instance at env construction time."""
    error_name = error_name or getattr(func, "__name__", func.__class__.__name__)

    class TerminationWrapper(Termination, register=False):
        def __init__(self, env, **params):
            super().__init__(env)
            self.params = params
            self._func_kwargs = resolve_wrapper_kwargs(
                func,
                params,
                kind="termination",
                error_name=error_name,
            )

        def __call__(self):
            return func(**self._func_kwargs)

    return TerminationWrapper


def termination_method(func):
    """Mark a Command method so _Env wraps it as a termination."""
    return method_marker("_mdp_termination_method")(func)
