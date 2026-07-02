import re
from typing import Sequence

import torch


def _match_indices(motion_names, asset_names, patterns, device=None, debug=False):
    asset_idx, motion_idx = [], []
    for i, a in enumerate(asset_names):
        if any(re.match(p, a) for p in patterns):
            if a in motion_names:
                asset_idx.append(i)
                motion_idx.append(motion_names.index(a))
                if debug:
                    print(f"Matched asset '{a}' (idx {i}) to motion '{a}' (idx {motion_names.index(a)})")
    return torch.tensor(motion_idx, device=device), torch.tensor(asset_idx, device=device)

def _calc_exp_sigma(error : torch.Tensor, sigma_list : list[float], reduce_last_dim : bool = False):
    if sigma_list is None or len(sigma_list) == 0:
        raise ValueError("sigma must be provided and non-empty.")
    count = len(sigma_list)
    if reduce_last_dim:
        rewards = [torch.exp(- error / sigma).mean(dim=-1, keepdim=True) for sigma in sigma_list]
    else:
        rewards = [torch.exp(- error / sigma) for sigma in sigma_list]
    return sum(rewards) / count

def get_items_by_index(values, indexes):
    if isinstance(indexes, torch.Tensor):
        indexes = indexes.tolist()
    return [values[i] for i in indexes]

def convert_dtype(dtype_str):
    dtype_map = {
        'float32': torch.float32,
        'float64': torch.float64,
        'int32': torch.int32,
        'int64': torch.int64,
        'bool': bool,
        'long': torch.long
    }
    if isinstance(dtype_str, str):
        if dtype_str not in dtype_map:
            raise ValueError(f"Unsupported dtype string: {dtype_str}")
        return dtype_map[dtype_str]
    return dtype_str


def _resolve_joint_indices(
    motion_joint_names: Sequence[str],
    asset_joint_names: Sequence[str],
    ordered_joint_names: Sequence[str],
    *,
    device=None,
    context: str = "joint mapping",
):
    motion_name_to_idx = {n: i for i, n in enumerate(motion_joint_names)}
    asset_name_to_idx = {n: i for i, n in enumerate(asset_joint_names)}

    selected_joint_names = []
    joint_idx_motion = []
    joint_idx_asset = []

    for name in ordered_joint_names:
        in_motion = name in motion_name_to_idx
        in_asset = name in asset_name_to_idx
        if not (in_motion and in_asset):
            raise ValueError(f"Joint '{name}' in {context} is not found in motion dataset or asset.")

        selected_joint_names.append(name)
        joint_idx_motion.append(motion_name_to_idx[name])
        joint_idx_asset.append(asset_name_to_idx[name])

    if len(selected_joint_names) == 0:
        raise RuntimeError(f"No joints resolved for {context}.")

    return (
        selected_joint_names,
        torch.tensor(joint_idx_motion, device=device, dtype=torch.long),
        torch.tensor(joint_idx_asset, device=device, dtype=torch.long),
    )
