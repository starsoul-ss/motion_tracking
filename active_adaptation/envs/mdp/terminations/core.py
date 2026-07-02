import torch

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mjlab.entity import Entity as Articulation
    from mjlab.sensor import ContactSensor

from active_adaptation.envs.mdp.contact_utils import resolve_contact_indices
from .base import Termination

class fall_over(Termination):
    def __init__(
        self, 
        env, 
        xy_thres: float=0.8,
        z_thres: float=0.5
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.xy_thres = xy_thres
        self.z_thres = z_thres
    
    def __call__(self):
        fall_over = (self.asset.data.projected_gravity_b[:, :2].norm(dim=1, keepdim=True) >= self.xy_thres) | (-self.asset.data.projected_gravity_b[:, 2:] < self.z_thres)
        return fall_over

class episode_timeout(Termination):
    def __call__(self) -> torch.Tensor:
        return (self.env.episode_length_buf >= self.env.max_episode_length).unsqueeze(1)
