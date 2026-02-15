from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

import torch

from active_adaptation.utils.math import (
    normalize,
    quat_apply,
    quat_apply_inverse,
    quat_conjugate,
    quat_from_angle_axis,
    quat_mul,
)

if TYPE_CHECKING:
    from active_adaptation.utils.motion import MotionData


def _basename(name: str) -> str:
    return name.split("/")[-1]


def _as_torch(value: Any, *, device: torch.device | None = None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        out = value
    elif hasattr(value, "_tensor"):
        # mjlab TorchArray wrapper
        out = value._tensor
    else:
        try:
            import warp as wp  # type: ignore

            if isinstance(value, wp.array):  # type: ignore[arg-type]
                out = wp.to_torch(value)
            else:
                out = torch.as_tensor(value)
        except Exception:
            out = torch.as_tensor(value)
    if device is not None and out.device != device:
        out = out.to(device=device)
    return out


def _as_scalar_1d(value: Any, *, device: torch.device) -> torch.Tensor:
    out = _as_torch(value, device=device)
    if out.ndim == 2 and out.shape[0] == 1:
        out = out[0]
    if out.ndim != 1:
        raise ValueError(f"Expected scalar field [N] or [1,N], got {tuple(out.shape)}")
    return out


def _as_vec_field(value: Any, *, dim: int, device: torch.device) -> torch.Tensor:
    out = _as_torch(value, device=device)
    if out.ndim == 3:
        out = out[0]
    if out.ndim != 2 or out.shape[-1] != dim:
        raise ValueError(f"Expected vector field [N,{dim}] or [1,N,{dim}], got {tuple(out.shape)}")
    return out


@dataclass
class FKChainInfo:
    tracked_body_names: list[str]
    tracked_joint_names: list[str | None]
    tracked_body_model_ids: torch.Tensor
    tracked_body_dataset_idx: torch.Tensor
    tracked_joint_dataset_idx: torch.Tensor


class UpperBodyFKHelper:
    """FK helper on the union tree from base to multiple EE links.

    Input/Output interface is MotionData-compatible: it rewrites body-related fields
    in place for tracked bodies only.
    """

    def __init__(
        self,
        *,
        device: torch.device,
        base_body_id: int,
        tracked_body_ids: torch.Tensor,
        parent_local_idx: torch.Tensor,
        body_pos0: torch.Tensor,
        body_quat0: torch.Tensor,
        joint_types: torch.Tensor,
        joint_pos_local: torch.Tensor,
        joint_axis_local: torch.Tensor,
        joint_dataset_idx: torch.Tensor,
        body_dataset_idx: torch.Tensor,
        tracked_body_names: list[str],
        tracked_joint_names: list[str | None],
        children_local: list[list[int]],
    ):
        self.device = device
        self.base_body_id = int(base_body_id)

        self.tracked_body_ids = tracked_body_ids.to(device=device, dtype=torch.long)
        self.parent_local_idx = parent_local_idx.to(device=device, dtype=torch.long)
        self.body_pos0 = body_pos0.to(device=device)
        self.body_quat0 = body_quat0.to(device=device)
        self.joint_types = joint_types.to(device=device, dtype=torch.long)
        self.joint_pos_local = joint_pos_local.to(device=device)
        self.joint_axis_local = joint_axis_local.to(device=device)
        self.joint_dataset_idx = joint_dataset_idx.to(device=device, dtype=torch.long)
        self.body_dataset_idx = body_dataset_idx.to(device=device, dtype=torch.long)

        self.tracked_body_names = tracked_body_names
        self.tracked_joint_names = tracked_joint_names
        self.children_local = children_local
        self.base_local_idx = int((self.tracked_body_ids == self.base_body_id).nonzero(as_tuple=False)[0].item())
        self.valid_local_idx = (self.body_dataset_idx >= 0).nonzero(as_tuple=False).squeeze(-1)
        self.valid_dataset_body_idx = self.body_dataset_idx[self.valid_local_idx]

        self._dtype_cache: dict[torch.dtype, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    @property
    def chain_info(self) -> FKChainInfo:
        return FKChainInfo(
            tracked_body_names=self.tracked_body_names,
            tracked_joint_names=self.tracked_joint_names,
            tracked_body_model_ids=self.tracked_body_ids,
            tracked_body_dataset_idx=self.body_dataset_idx,
            tracked_joint_dataset_idx=self.joint_dataset_idx,
        )

    @classmethod
    def from_mjlab_asset(
        cls,
        *,
        asset: Any,
        dataset_joint_names: Sequence[str],
        dataset_body_names: Sequence[str],
        ee_link_names: Sequence[str],
        base_body_name: str | None = None,
    ) -> "UpperBodyFKHelper":
        dev = torch.device(asset.data.device)
        model = asset.data.model

        body_name_to_id = {"world": 0}
        for i, name in enumerate(asset.body_names):
            gid = int(asset.indexing.body_ids[i].item())
            body_name_to_id[_basename(name)] = gid

        joint_id_to_name: dict[int, str] = {}
        for i, name in enumerate(asset.joint_names):
            gid = int(asset.indexing.joint_ids[i].item())
            joint_id_to_name[gid] = _basename(name)

        if base_body_name is None:
            base_body_name = _basename(asset.body_names[0])

        return cls._build(
            model=model,
            body_name_to_id=body_name_to_id,
            joint_id_to_name=joint_id_to_name,
            dataset_joint_names=dataset_joint_names,
            dataset_body_names=dataset_body_names,
            ee_link_names=ee_link_names,
            base_body_name=base_body_name,
            device=dev,
        )

    @classmethod
    def _build(
        cls,
        *,
        model: Any,
        body_name_to_id: dict[str, int],
        joint_id_to_name: dict[int, str],
        dataset_joint_names: Sequence[str],
        dataset_body_names: Sequence[str],
        ee_link_names: Sequence[str],
        base_body_name: str,
        device: torch.device,
    ) -> "UpperBodyFKHelper":
        body_parentid = _as_scalar_1d(getattr(model, "body_parentid"), device=device).to(torch.long)
        body_jntnum = _as_scalar_1d(getattr(model, "body_jntnum"), device=device).to(torch.long)
        body_jntadr = _as_scalar_1d(getattr(model, "body_jntadr"), device=device).to(torch.long)
        body_pos_all = _as_vec_field(getattr(model, "body_pos"), dim=3, device=device)
        body_quat_all = _as_vec_field(getattr(model, "body_quat"), dim=4, device=device)
        jnt_type = _as_scalar_1d(getattr(model, "jnt_type"), device=device).to(torch.long)
        jnt_pos = _as_vec_field(getattr(model, "jnt_pos"), dim=3, device=device)
        jnt_axis = _as_vec_field(getattr(model, "jnt_axis"), dim=3, device=device)

        if base_body_name not in body_name_to_id:
            raise ValueError(f"Base body '{base_body_name}' not found in model")
        base_body_id = int(body_name_to_id[base_body_name])

        ee_body_ids: list[int] = []
        for ee_name in ee_link_names:
            if ee_name not in body_name_to_id:
                raise ValueError(f"EE body '{ee_name}' not found in model")
            ee_body_ids.append(int(body_name_to_id[ee_name]))

        # Union of paths: ee -> ... -> base
        selected: set[int] = set()
        for ee_id in ee_body_ids:
            cur = ee_id
            while True:
                selected.add(cur)
                if cur == base_body_id:
                    break
                parent = int(body_parentid[cur].item())
                if parent < 0:
                    raise RuntimeError(f"Cannot trace ee body id={ee_id} to base '{base_body_name}'")
                cur = parent

        # Build subtree adjacency then DFS from base. Shared torso-like branches are naturally deduplicated.
        children_by_gid: dict[int, list[int]] = {bid: [] for bid in selected}
        for bid in selected:
            if bid == base_body_id:
                continue
            pid = int(body_parentid[bid].item())
            if pid in selected:
                children_by_gid[pid].append(bid)
        for v in children_by_gid.values():
            v.sort()

        order_gid: list[int] = []

        def _dfs(bid: int):
            order_gid.append(bid)
            for cid in children_by_gid.get(bid, []):
                _dfs(cid)

        _dfs(base_body_id)
        order = torch.tensor(order_gid, device=device, dtype=torch.long)
        gid_to_local = {gid: i for i, gid in enumerate(order_gid)}

        ds_joint_idx = {n: i for i, n in enumerate(dataset_joint_names)}
        ds_body_idx = {n: i for i, n in enumerate(dataset_body_names)}

        K = len(order_gid)
        parent_local_idx = torch.full((K,), -1, device=device, dtype=torch.long)
        body_pos0 = torch.empty((K, 3), device=device, dtype=body_pos_all.dtype)
        body_quat0 = torch.empty((K, 4), device=device, dtype=body_quat_all.dtype)
        joint_types = torch.full((K,), -1, device=device, dtype=torch.long)
        joint_pos_local = torch.zeros((K, 3), device=device, dtype=jnt_pos.dtype)
        joint_axis_local = torch.zeros((K, 3), device=device, dtype=jnt_axis.dtype)
        joint_dataset_idx = torch.full((K,), -1, device=device, dtype=torch.long)
        body_dataset_idx = torch.full((K,), -1, device=device, dtype=torch.long)
        tracked_body_names: list[str] = []
        tracked_joint_names: list[str | None] = []

        id_to_body_name = {v: k for k, v in body_name_to_id.items()}

        for i, gid in enumerate(order_gid):
            pid = int(body_parentid[gid].item())
            parent_local_idx[i] = gid_to_local[pid] if pid in gid_to_local else -1

            body_pos0[i] = body_pos_all[gid]
            body_quat0[i] = body_quat_all[gid]

            body_name = id_to_body_name.get(gid, f"body_{gid}")
            tracked_body_names.append(body_name)
            body_dataset_idx[i] = int(ds_body_idx.get(body_name, -1))

            jn = int(body_jntnum[gid].item())
            if jn > 1:
                raise NotImplementedError(
                    f"Body '{body_name}' has {jn} joints; helper currently supports <=1 joint/body."
                )

            joint_name: str | None = None
            if jn == 1:
                jid = int(body_jntadr[gid].item())
                jtype = int(jnt_type[jid].item())
                # Ignore base freejoint, root state comes from MotionData input.
                if not (gid == base_body_id and jtype == 0):
                    joint_types[i] = jtype
                    joint_pos_local[i] = jnt_pos[jid]
                    joint_axis_local[i] = jnt_axis[jid]

                    joint_name = joint_id_to_name.get(jid, None)
                    if joint_name is None:
                        raise ValueError(f"Joint id={jid} has no name mapping.")
                    jidx = int(ds_joint_idx.get(joint_name, -1))
                    if jidx < 0:
                        raise ValueError(f"Joint '{joint_name}' on tracked chain missing in dataset_joint_names.")
                    joint_dataset_idx[i] = jidx
                    if jtype not in (2, 3):  # slide/hinge
                        raise NotImplementedError(f"Joint '{joint_name}' type={jtype} unsupported.")
            tracked_joint_names.append(joint_name)

        # local-idx adjacency
        children_local: list[list[int]] = [[] for _ in range(K)]
        for i in range(K):
            p = int(parent_local_idx[i].item())
            if p >= 0:
                children_local[p].append(i)

        return cls(
            device=device,
            base_body_id=base_body_id,
            tracked_body_ids=order,
            parent_local_idx=parent_local_idx,
            body_pos0=body_pos0,
            body_quat0=body_quat0,
            joint_types=joint_types,
            joint_pos_local=joint_pos_local,
            joint_axis_local=joint_axis_local,
            joint_dataset_idx=joint_dataset_idx,
            body_dataset_idx=body_dataset_idx,
            tracked_body_names=tracked_body_names,
            tracked_joint_names=tracked_joint_names,
            children_local=children_local,
        )

    def _typed_constants(self, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cached = self._dtype_cache.get(dtype, None)
        if cached is None:
            cached = (
                self.body_pos0.to(dtype=dtype),
                normalize(self.body_quat0.to(dtype=dtype)),
                self.joint_pos_local.to(dtype=dtype),
                normalize(self.joint_axis_local.to(dtype=dtype)),
            )
            self._dtype_cache[dtype] = cached
        return cached

    def forward(self, motion: "MotionData") -> "MotionData":
        """Rewrite tracked body fields in MotionData and return the same object."""
        return self.rewrite_motion_data_(motion)

    def rewrite_motion_data_(self, motion: "MotionData") -> "MotionData":
        root_pos = motion.root_pos_w
        root_quat = motion.root_quat_w
        root_lin_vel = motion.root_lin_vel_w
        root_ang_vel = motion.root_ang_vel_w
        joint_pos = motion.joint_pos
        joint_vel = motion.joint_vel

        body_pos_w = motion.body_pos_w
        body_pos_b = motion.body_pos_b
        body_vel_w = motion.body_vel_w
        body_vel_b = motion.body_vel_b
        body_quat_w = motion.body_quat_w
        body_quat_b = motion.body_quat_b
        body_angvel_w = motion.body_angvel_w
        body_angvel_b = motion.body_angvel_b

        dtype = root_pos.dtype
        dev = root_pos.device

        if dev != self.device:
            raise RuntimeError(
                f"FK helper device mismatch: helper={self.device}, motion={dev}. "
                "Please build helper on the same device."
            )

        prefix = root_pos.shape[:-1]
        M = int(torch.tensor(prefix).prod().item()) if len(prefix) > 0 else 1
        K = len(self.tracked_body_names)

        # Flatten arbitrary leading batch dims (e.g. [N], [N,S], ...).
        root_pos_f = root_pos.reshape(M, 3)
        root_quat_f = root_quat.reshape(M, 4)
        root_lin_vel_f = root_lin_vel.reshape(M, 3)
        root_ang_vel_f = root_ang_vel.reshape(M, 3)
        joint_pos_f = joint_pos.reshape(M, joint_pos.shape[-1])
        joint_vel_f = joint_vel.reshape(M, joint_vel.shape[-1])

        body_pos_w_f = body_pos_w.reshape(M, body_pos_w.shape[-2], 3)
        body_pos_b_f = body_pos_b.reshape(M, body_pos_b.shape[-2], 3)
        body_vel_w_f = body_vel_w.reshape(M, body_vel_w.shape[-2], 3)
        body_vel_b_f = body_vel_b.reshape(M, body_vel_b.shape[-2], 3)
        body_quat_w_f = body_quat_w.reshape(M, body_quat_w.shape[-2], 4)
        body_quat_b_f = body_quat_b.reshape(M, body_quat_b.shape[-2], 4)
        body_angvel_w_f = body_angvel_w.reshape(M, body_angvel_w.shape[-2], 3)
        body_angvel_b_f = body_angvel_b.reshape(M, body_angvel_b.shape[-2], 3)

        bpos0, bquat0, jpos_local, jaxis_local = self._typed_constants(dtype)
        jtypes = self.joint_types
        jidx = self.joint_dataset_idx

        out_pos_w = torch.zeros((M, K, 3), device=dev, dtype=dtype)
        out_quat_w = torch.zeros((M, K, 4), device=dev, dtype=dtype)
        out_vel_w = torch.zeros((M, K, 3), device=dev, dtype=dtype)
        out_angvel_w = torch.zeros((M, K, 3), device=dev, dtype=dtype)

        b0 = self.base_local_idx
        out_pos_w[:, b0] = root_pos_f
        out_quat_w[:, b0] = root_quat_f
        out_vel_w[:, b0] = root_lin_vel_f
        out_angvel_w[:, b0] = root_ang_vel_f

        # Topological local order from DFS; parent always computed before child.
        for li in range(K):
            if li == b0:
                continue
            p = int(self.parent_local_idx[li].item())
            if p < 0:
                continue

            p_pos = out_pos_w[:, p]
            p_quat = out_quat_w[:, p]
            p_vel = out_vel_w[:, p]
            p_ang = out_angvel_w[:, p]

            p0 = bpos0[li].unsqueeze(0).expand(M, -1)
            q0 = bquat0[li].unsqueeze(0).expand(M, -1)

            jt = int(jtypes[li].item())
            if jt < 0:
                q_rel = q0
                pos_rel = p0
                c_quat = normalize(quat_mul(p_quat, q_rel))
                c_pos = p_pos + quat_apply(p_quat, pos_rel)
                c_ang = p_ang
                c_vel = p_vel + torch.cross(p_ang, c_pos - p_pos, dim=-1)
            else:
                ji = int(jidx[li].item())
                if ji < 0:
                    raise RuntimeError(f"Tracked joint '{self.tracked_joint_names[li]}' has invalid dataset index")
                q = joint_pos_f[:, ji]
                qd = joint_vel_f[:, ji]
                axis_l = jaxis_local[li].unsqueeze(0).expand(M, -1)
                anchor_l = jpos_local[li].unsqueeze(0).expand(M, -1)

                if jt == 3:  # hinge
                    qj = quat_from_angle_axis(q, axis_l)
                    q_rel = quat_mul(q0, qj)
                    pos_rel = p0 + quat_apply(q0, anchor_l - quat_apply(qj, anchor_l))
                    c_quat = normalize(quat_mul(p_quat, q_rel))
                    c_pos = p_pos + quat_apply(p_quat, pos_rel)

                    axis_parent = quat_apply(q0, axis_l)
                    axis_w = quat_apply(p_quat, axis_parent)
                    w_rel = axis_w * qd.unsqueeze(-1)
                    c_ang = p_ang + w_rel

                    anchor_rel = p0 + quat_apply(q0, anchor_l)
                    anchor_w = p_pos + quat_apply(p_quat, anchor_rel)
                    c_vel = p_vel + torch.cross(p_ang, c_pos - p_pos, dim=-1) + torch.cross(
                        w_rel, c_pos - anchor_w, dim=-1
                    )
                elif jt == 2:  # slide
                    axis_parent = quat_apply(q0, axis_l)
                    pos_rel = p0 + axis_parent * q.unsqueeze(-1)
                    q_rel = q0
                    c_quat = normalize(quat_mul(p_quat, q_rel))
                    c_pos = p_pos + quat_apply(p_quat, pos_rel)

                    axis_w = quat_apply(p_quat, axis_parent)
                    c_ang = p_ang
                    c_vel = p_vel + torch.cross(p_ang, c_pos - p_pos, dim=-1) + axis_w * qd.unsqueeze(-1)
                else:
                    raise NotImplementedError(
                        f"Unsupported joint type {jt} for joint '{self.tracked_joint_names[li]}'."
                    )

            out_pos_w[:, li] = c_pos
            out_quat_w[:, li] = c_quat
            out_vel_w[:, li] = c_vel
            out_angvel_w[:, li] = c_ang

        if self.valid_local_idx.numel() == 0:
            return motion

        ds_ids = self.valid_dataset_body_idx
        sel_pos_w = out_pos_w[:, self.valid_local_idx]
        sel_quat_w = out_quat_w[:, self.valid_local_idx]
        sel_vel_w = out_vel_w[:, self.valid_local_idx]
        sel_ang_w = out_angvel_w[:, self.valid_local_idx]

        body_pos_w_f[:, ds_ids] = sel_pos_w
        body_quat_w_f[:, ds_ids] = sel_quat_w
        body_vel_w_f[:, ds_ids] = sel_vel_w
        body_angvel_w_f[:, ds_ids] = sel_ang_w

        root_q = root_quat_f.unsqueeze(1)
        root_p = root_pos_f.unsqueeze(1)
        root_lv = root_lin_vel_f.unsqueeze(1)
        root_av = root_ang_vel_f.unsqueeze(1)

        # Match MotionDataset formulas: b-frame values are computed from world values and root state.
        body_pos_b_f[:, ds_ids] = quat_apply_inverse(root_q, sel_pos_w - root_p)
        body_vel_b_f[:, ds_ids] = quat_apply_inverse(root_q, sel_vel_w - root_lv)
        body_quat_b_f[:, ds_ids] = quat_mul(quat_conjugate(root_q), sel_quat_w)
        body_angvel_b_f[:, ds_ids] = quat_apply_inverse(root_q, sel_ang_w - root_av)

        return motion
