import torch
from typing import Sequence


class TensorRingBuffer:
    """Small helper for fixed-size per-env history buffers.

    Storage is kept in `[num_envs, capacity, *value_shape]` layout. New values
    are written by moving a head pointer instead of shifting the whole buffer.
    """

    def __init__(
        self,
        num_envs: int,
        capacity: int,
        value_shape: int | Sequence[int],
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ):
        self.num_envs = int(num_envs)
        self.capacity = int(capacity)
        if isinstance(value_shape, int):
            value_shape = (value_shape,)
        self.value_shape = tuple(int(v) for v in value_shape)
        self.device = torch.device(device)
        self.dtype = dtype

        self.buffer = torch.zeros(
            (self.num_envs, self.capacity, *self.value_shape),
            device=self.device,
            dtype=self.dtype,
        )
        self.head = 0
        self._offsets = torch.arange(self.capacity, device=self.device, dtype=torch.long)

    def reset(self, env_ids: torch.Tensor):
        self.buffer[env_ids] = 0

    def push(self, value: torch.Tensor):
        self.head = (self.head - 1) % self.capacity
        self.buffer[:, self.head].copy_(value)

    def take(self, offsets: torch.Tensor) -> torch.Tensor:
        offsets = offsets.to(device=self.device, dtype=torch.long)
        idx = (offsets + self.head) % self.capacity
        return self.buffer.index_select(1, idx)

    def recent(self, steps: int) -> torch.Tensor:
        steps = min(int(steps), self.capacity)
        return self.take(self._offsets[:steps])

    def take_per_env(self, offsets: torch.Tensor) -> torch.Tensor:
        offsets = offsets.to(device=self.device, dtype=torch.long)
        squeeze_dim = offsets.ndim == 1
        if squeeze_dim:
            offsets = offsets.unsqueeze(1)

        idx = (offsets + self.head) % self.capacity
        index = idx
        for _ in self.value_shape:
            index = index.unsqueeze(-1)
        index = index.expand(idx.shape[0], idx.shape[1], *self.value_shape)
        values = self.buffer.take_along_dim(index, dim=1)
        return values.squeeze(1) if squeeze_dim else values
