from .base import Reward, reward_method, reward_wrapper
from . import locomotion as _locomotion

REW_REGISTRY = Reward.registry

__all__ = [
    "Reward",
    "REW_REGISTRY",
    "reward_method",
    "reward_wrapper",
]
