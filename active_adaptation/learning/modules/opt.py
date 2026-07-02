import torch
import warnings
from ..ppo.common import unique_trainable_params


class OptimizerGroup(torch.optim.Optimizer):
    """
    Wrapper around multiple optimizers so they can be used through a single
    optimizer-like interface (step/zero_grad/param_groups).
    """

    def __init__(self, optimizers: list[torch.optim.Optimizer]):
        if len(optimizers) == 0:
            raise ValueError("OptimizerGroup requires at least one optimizer.")

        # Collect all parameters from the wrapped optimizers so that the base
        # Optimizer constructor is satisfied (it disallows an empty parameter
        # list). We won't use the base step/zero_grad implementations, only
        # some of its bookkeeping.
        all_params = []
        for opt in optimizers:
            for group in opt.param_groups:
                all_params.extend(group["params"])
        if len(all_params) == 0:
            raise ValueError(
                "OptimizerGroup underlying optimizers have no parameters."
            )

        super().__init__(params=all_params, defaults={})
        self.optimizers = optimizers

        # Flatten the underlying param_groups so external code can keep using
        # `opt.param_groups[0]['lr']` etc. These dict objects come from the
        # wrapped optimizers, so mutating them here updates those optimizers.
        self.param_groups = []
        for opt in self.optimizers:
            self.param_groups.extend(opt.param_groups)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            # Use the closure with the first optimizer to preserve semantics,
            # then step the remaining optimizers without a closure.
            loss = self.optimizers[0].step(closure)
            for opt in self.optimizers[1:]:
                opt.step()
            return loss

        for opt in self.optimizers:
            _loss = opt.step()
            if loss is None:
                loss = _loss
        return loss

    def zero_grad(self, set_to_none: bool | None = None):
        for opt in self.optimizers:
            if set_to_none is None:
                opt.zero_grad()
            else:
                opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        # Simple, explicit format: a list of state dicts for the wrapped
        # optimizers.
        return {
            "optimizers": [opt.state_dict() for opt in self.optimizers],
            "class": self.__class__.__name__,
        }

    def load_state_dict(self, state_dict):
        opt_states = state_dict.get("optimizers", None)
        if opt_states is None:
            return
        if len(opt_states) != len(self.optimizers):
            warnings.warn(
                f"OptimizerGroup state has {len(opt_states)} optimizers, "
                f"but current instance has {len(self.optimizers)}. "
                "Loading states for the matching prefix only."
            )
        for opt, opt_state in zip(self.optimizers, opt_states):
            opt.load_state_dict(opt_state)


def _get_muon_optimizer():
    muon = getattr(torch.optim, "Muon", None)
    if muon is None:
        raise ImportError(
            "optimizer='muon' requires torch.optim.Muon, but this PyTorch build does not provide it."
        )
    return muon

def build_optimizer(
    params,
    *,
    optimizer: str,
    lr: float,
    weight_decay: float,
    adamw_only_params=None,
):
    params = unique_trainable_params([] if params is None else params)
    forced_adamw_params = unique_trainable_params([] if adamw_only_params is None else adamw_only_params)
    if not params:
        raise ValueError("optimizer received no trainable parameters")

    param_ids = {id(p) for p in params}
    if any(id(p) not in param_ids for p in forced_adamw_params):
        raise ValueError("build_optimizer received `adamw_only_params` that are not part of `params`.")

    forced_adamw_ids = {id(p) for p in forced_adamw_params}
    muon_params = []
    adamw_matrix_params = []
    adamw_other_params = []
    for p in params:
        if id(p) in forced_adamw_ids:
            target = adamw_matrix_params if p.ndim == 2 else adamw_other_params
        elif p.ndim == 2:
            target = muon_params
        else:
            target = adamw_other_params
        target.append(p)

    if optimizer == "adam":
        adamw_param_groups = []
        matrix_params = muon_params + adamw_matrix_params
        if matrix_params:
            adamw_param_groups.append({"params": matrix_params, "weight_decay": weight_decay})
        if adamw_other_params:
            adamw_param_groups.append({"params": adamw_other_params, "weight_decay": 0.0})
        if not adamw_param_groups:
            raise ValueError("optimizer received no trainable parameters")
        return torch.optim.AdamW(adamw_param_groups, lr=lr, weight_decay=weight_decay)
    if optimizer != "muon":
        raise ValueError(f"unsupported optimizer: {optimizer}")

    muon_cls = _get_muon_optimizer()
    optimizers = []
    adamw_param_groups = []
    if adamw_matrix_params:
        adamw_param_groups.append({"params": adamw_matrix_params, "weight_decay": weight_decay})
    if adamw_other_params:
        adamw_param_groups.append({"params": adamw_other_params, "weight_decay": 0.0})
    if adamw_param_groups:
        optimizers.append(
            torch.optim.AdamW(
                adamw_param_groups,
                lr=lr,
                weight_decay=weight_decay,
            )
        )
    if muon_params:
        optimizers.append(
            muon_cls(
                [{"params": muon_params, "weight_decay": weight_decay}],
                lr=lr,
                adjust_lr_fn="match_rms_adamw",
                weight_decay=weight_decay,
            )
        )
    if len(optimizers) == 1:
        return optimizers[0]
    return OptimizerGroup(optimizers)
