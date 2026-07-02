import torch
import torch.distributions as D
from torch.distributions import constraints

D.Distribution.set_default_validate_args(False)


class IndependentNormal(D.Independent):
    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}

    def __init__(self, loc, scale, validate_args=None):
        scale = torch.clamp_min(scale, 1e-6)
        base_dist = D.Normal(loc, scale)
        super().__init__(base_dist, 1, validate_args=validate_args)

    @property
    def scale(self):
        return self.base_dist.scale

    @property
    def deterministic_sample(self):
        return self.base_dist.mean
