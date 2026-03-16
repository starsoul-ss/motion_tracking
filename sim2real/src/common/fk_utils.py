from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from paths import REAL_G1_ROOT


def _require_cfg(policy, key: str):
    cfg = getattr(policy, "config", None)
    if cfg is None or not hasattr(cfg, key):
        raise KeyError(f"Missing required config key '{key}' for observation setup.")
    return getattr(cfg, key)


def _resolve_path_from_repo(path_like: str) -> Path:
    p = Path(path_like)
    if p.is_absolute():
        return p

    repo_root = REAL_G1_ROOT.parent
    candidate_repo = (repo_root / p).resolve()
    if candidate_repo.exists():
        return candidate_repo

    candidate_sim2real = (REAL_G1_ROOT / p).resolve()
    if candidate_sim2real.exists():
        return candidate_sim2real

    return candidate_repo


class PolicyKeypointPinFK:
    @staticmethod
    def _try_load_motion_body_names(policy) -> Optional[List[str]]:
        cfg = getattr(policy, "config", None)
        motions = list(getattr(cfg, "motions", [])) if cfg is not None else []
        for m in motions:
            path_raw = m.get("path") if isinstance(m, dict) else getattr(m, "path", None)
            if path_raw is None:
                continue
            path = _resolve_path_from_repo(str(path_raw))
            if not path.exists():
                continue
            try:
                with np.load(path, allow_pickle=True) as data:
                    if "body_names" not in data.files:
                        continue
                    names = []
                    for n in data["body_names"].tolist():
                        if isinstance(n, (bytes, bytearray, np.bytes_)):
                            names.append(n.decode("utf-8"))
                        else:
                            names.append(str(n))
                    if len(names) > 0:
                        return names
            except Exception:
                continue
        return None

    def __init__(self, policy):
        self.policy = policy

        try:
            import pinocchio as pin
        except Exception as e:
            raise ImportError("pinocchio is required for target_policy_keypoints_pos_b_obs") from e
        self.pin = pin

        urdf_path_raw = str(_require_cfg(policy, "policy_keypoints_fk_urdf"))
        patterns = [str(p) for p in list(_require_cfg(policy, "policy_keypoint_patterns"))]
        if len(patterns) == 0:
            raise ValueError("policy_keypoint_patterns must be non-empty.")

        urdf_path = _resolve_path_from_repo(urdf_path_raw)
        if not urdf_path.exists():
            raise FileNotFoundError(f"URDF not found for policy keypoint FK: {urdf_path}")

        self.model = self.pin.buildModelFromUrdf(str(urdf_path), self.pin.JointModelFreeFlyer())
        self.data = self.model.createData()
        self.q_neutral = self.pin.neutral(self.model)

        self.joint_q_indices = np.zeros(len(policy.obs_joint_names), dtype=np.int32)
        for i, joint_name in enumerate(policy.obs_joint_names):
            jid = int(self.model.getJointId(joint_name))
            if jid <= 0:
                raise ValueError(f"Joint '{joint_name}' not found in FK URDF '{urdf_path}'.")
            jmodel = self.model.joints[jid]
            if int(jmodel.nq) != 1:
                raise ValueError(
                    f"Joint '{joint_name}' has nq={jmodel.nq}, expected 1 for single-DoF tracking joints."
                )
            self.joint_q_indices[i] = int(jmodel.idx_q)

        self.body_frame_ids: Dict[str, int] = {}
        for fid, frame in enumerate(self.model.frames):
            if frame.type == self.pin.FrameType.BODY and frame.name not in self.body_frame_ids:
                self.body_frame_ids[frame.name] = int(fid)

        motion_body_names = self._try_load_motion_body_names(policy)
        motion_body_name_set = set(motion_body_names) if motion_body_names is not None else None

        keypoint_names: List[str] = []
        for name in self.body_frame_ids.keys():
            if any(re.match(p, name) for p in patterns):
                if motion_body_name_set is not None and name not in motion_body_name_set:
                    continue
                keypoint_names.append(name)

        if len(keypoint_names) == 0:
            raise ValueError(
                f"No policy keypoints matched patterns={patterns} in FK URDF frames."
            )

        self.keypoint_names = keypoint_names
        self.keypoint_frame_ids = [self.body_frame_ids[n] for n in self.keypoint_names]
        self.num_keypoints = len(self.keypoint_names)

    def compute_keypoints_world(
        self,
        root_pos_w: np.ndarray,
        root_quat_wxyz: np.ndarray,
        joint_pos: np.ndarray,
    ) -> np.ndarray:
        root_pos_w = np.asarray(root_pos_w, dtype=np.float64)
        root_quat_wxyz = np.asarray(root_quat_wxyz, dtype=np.float64)
        joint_pos = np.asarray(joint_pos, dtype=np.float64)

        if root_pos_w.ndim != 2 or root_pos_w.shape[1] != 3:
            raise ValueError(f"root_pos_w must be [S,3], got {root_pos_w.shape}")
        if root_quat_wxyz.ndim != 2 or root_quat_wxyz.shape[1] != 4:
            raise ValueError(f"root_quat_wxyz must be [S,4], got {root_quat_wxyz.shape}")
        if joint_pos.ndim != 2:
            raise ValueError(f"joint_pos must be [S,J], got {joint_pos.shape}")
        if not (root_pos_w.shape[0] == root_quat_wxyz.shape[0] == joint_pos.shape[0]):
            raise ValueError(
                f"Step size mismatch: pos={root_pos_w.shape}, quat={root_quat_wxyz.shape}, joint={joint_pos.shape}"
            )

        n_steps = int(root_pos_w.shape[0])
        out = np.zeros((n_steps, self.num_keypoints, 3), dtype=np.float32)
        q = self.q_neutral.copy()

        for i in range(n_steps):
            q[:] = self.q_neutral
            q[:3] = root_pos_w[i]

            qwxyz = root_quat_wxyz[i]
            qxyzw = np.array([qwxyz[1], qwxyz[2], qwxyz[3], qwxyz[0]], dtype=np.float64)
            qn = float(np.linalg.norm(qxyzw))
            if qn < 1e-9:
                qxyzw[:] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
            else:
                qxyzw /= qn
            q[3:7] = qxyzw

            q[self.joint_q_indices] = joint_pos[i]
            self.pin.forwardKinematics(self.model, self.data, q)
            self.pin.updateFramePlacements(self.model, self.data)

            for k, fid in enumerate(self.keypoint_frame_ids):
                out[i, k] = np.asarray(self.data.oMf[fid].translation, dtype=np.float32)
        return out
