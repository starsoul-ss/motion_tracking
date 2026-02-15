import torch
import numpy as np
import json
import functools
from pathlib import Path
from typing import Callable
from tensordict import TensorClass, MemoryMappedTensor
from scipy.spatial.transform import Rotation as sRot
import os
from active_adaptation.utils.motion_utils import (
    interpolate,
    rotate_to_body,
    select_in_order,
    angvel_from_rot,
    angvel_from_quat_wxyz_batch,
    finite_diff_vel,
)

class MotionData(TensorClass):
    motion_id: torch.Tensor
    step: torch.Tensor
    root_pos_w: torch.Tensor
    root_quat_w: torch.Tensor
    root_lin_vel_w: torch.Tensor
    root_ang_vel_w: torch.Tensor
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    body_pos_w: torch.Tensor
    body_pos_b: torch.Tensor
    body_vel_w: torch.Tensor
    body_vel_b: torch.Tensor
    body_quat_w: torch.Tensor
    body_quat_b: torch.Tensor
    body_angvel_w: torch.Tensor
    body_angvel_b: torch.Tensor


class MotionOriginalData(TensorClass):
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    body_pos_w: torch.Tensor

from tqdm import tqdm

class MotionDataset:
    def __init__(self, body_names: list, joint_names: list, starts: list, ends: list, data: MotionData, info: list, device: torch.device = torch.device('cpu')):
        self.body_names = body_names
        self.joint_names = joint_names
        self.starts = torch.as_tensor(starts, dtype=torch.int32, device=device)
        self.ends = torch.as_tensor(ends, dtype=torch.int32, device=device)
        self.lengths = self.ends - self.starts
        self.data = data
        self.info = info

    @classmethod
    def create_from_path_lazy(cls, mem_path: str, dataset_extra_keys: list[dict] = [], device: torch.device = torch.device('cpu')):
        # get mempath root
        path_root = os.environ.get("MEMPATH")
        if path_root is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.join(current_dir, "../..")
            path_root = os.path.join(project_root, "dataset")
        mem_path = os.path.join(path_root, mem_path)

        data = MotionData.load(mem_path)
        data = data.to(device)
        with open(mem_path + "/meta_motion.json", "r") as f:
            meta = json.load(f)
        
        infos = {}
        for k in dataset_extra_keys:
            k_name = k['name']
            k_shape = k['shape']
            if k_name not in meta['info'].keys():
                infos[k_name] = torch.zeros((len(meta['starts']), k_shape), dtype=k['dtype'], device=device)
            else:
                infos[k_name] = torch.tensor(meta['info'][k_name], dtype=k['dtype'], device=device)
            if infos[k_name].shape != (len(meta['starts']), k_shape):
                raise ValueError(f"Shape of {k_name} does not match: {infos[k_name].shape} != {len(meta['starts'])}, {k_shape}")

        return cls(body_names=meta['body_names'], joint_names=meta['joint_names'], starts=meta['starts'], ends=meta['ends'], data=data, info=infos, device=device)

    @classmethod
    def create_from_path(
        cls,
        root_path: str,
        target_fps: int = 50,
        mem_path: str | None = None,
        motion_processer: Callable | None = None,
        motion_filter: Callable | None = None,
        callback: Callable | None = None,
        pad_before: int = 0,
        pad_after: int = 0,
        segment_len: int = 1000,
        *,
        build_dataset: bool = True,
        storage_float_dtype: torch.dtype = torch.float16,
        storage_int_dtype: torch.dtype = torch.int32,
    ):
        root = Path(root_path)
        # Support single file or directory
        meta = None
        if root.is_file() and root.suffix == '.npz':
            paths = [root]
        else:
            paths = list(root.rglob('*.npz'))
        if not paths:
            raise RuntimeError(f"No motions found in {root_path}")

        motions = [] if build_dataset else None
        id_labels = [] if build_dataset else None
        total = 0
        
        pb = tqdm(paths)
        # Read and interpolate
        # Calculate velocities and world frame poses based on the following keys before segmentation
        preserved_keys = [
            'fps',
            'qpos',
            'qvel',
            'xpos',
            'xquat',
            'body_vel_w',
            'body_vel_b',
            'body_quat_b',
            'body_angvel_w',
            'body_angvel_b',
        ]
        SEGMENT_LEN = segment_len
        
        foot_names = ['left_ankle_roll_link', 'right_ankle_roll_link']
        foot_idx = None
        
        joint_names_keep = ['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint', 'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint', 'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint', 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint']
        body_names_keep = ["world", "pelvis", "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link", "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link", "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link", "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link", "torso_link", "left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link", "left_elbow_link", "left_wrist_roll_link", "right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link", "right_elbow_link", "right_wrist_roll_link", "head_mimic", "left_hand_mimic", "right_hand_mimic"]
        for p in pb:
            m = dict(np.load(p, allow_pickle=True))
            
            # Initialize metadata from first motion
            if meta is None:
                joint_names = m["joint_names"].tolist()
                body_names = m["body_names"].tolist()

                sel_joint_names, sel_joint_idx = select_in_order(joint_names, joint_names_keep, return_missing=False)
                sel_body_names, sel_body_idx = select_in_order(body_names, body_names_keep, return_missing=False)

                meta = {"joint_names": sel_joint_names,
                        "body_names": sel_body_names}
                meta['info'] = []
            if foot_idx is None:
                foot_idx = [meta['body_names'].index(n) for n in foot_names]
            
            # Extract standard field
            m['fps'] = int(m.get('mocap_framerate', m.get('frequency', m.get('fps', 0))))

            # Pad data with specified frames before and after
            def pad_data(data, pad_before, pad_after):
                slot = np.zeros((pad_before + pad_after + data.shape[0], *data.shape[1:]), dtype=data.dtype)
                slot[pad_before:pad_before + data.shape[0], :] = data
                if pad_before > 0:
                    slot[:pad_before] = slot[pad_before:pad_before+1]
                if pad_after > 0:
                    slot[-pad_after:] = slot[-pad_after-1:-pad_after]
                return slot
            
            m["root_pos"] = pad_data(m["root_pos"], pad_before, pad_after)
            m["root_rot"] = pad_data(m["root_rot"], pad_before, pad_after)
            m["dof_pos"] = pad_data(m["dof_pos"][:, sel_joint_idx], pad_before, pad_after)
            m["local_body_pos"] = pad_data(m["local_body_pos"][:, sel_body_idx, :], pad_before, pad_after)
            m["local_body_rot"] = pad_data(m["local_body_rot"][:, sel_body_idx, :], pad_before, pad_after)

            # Calculate velocities and world frame poses based on new data format
            # (completed before segmentation to avoid boundary artifacts)
            T_full = m['root_pos'].shape[0]
            fps = int(m['fps'])
            # Root position and orientation
            root_pos = m['root_pos'].astype(np.float32)
            root_quat_xyzw = m['root_rot'].astype(np.float32)
            # Convert to SciPy Rotation
            R_root = sRot.from_quat(root_quat_xyzw)  # expects (x,y,z,w)
            R_root_m = R_root.as_matrix()            # (T,3,3)
            # Local body poses (relative to root)
            local_pos = m['local_body_pos'].astype(np.float32)     # (T,B,3)
            local_quat_xyzw = m['local_body_rot'].astype(np.float32)  # (T,B,4)
            # World frame body poses
            body_pos_w = np.einsum('tij,tbj->tbi', R_root_m, local_pos) + root_pos[:, None, :]
            R_local_m = sRot.from_quat(local_quat_xyzw.reshape(-1, 4)).as_matrix().reshape(T_full, -1, 3, 3)
            R_world_m = np.einsum('tij,tbjk->t bik', R_root_m, R_local_m)
            body_quat_w_xyzw = sRot.from_matrix(R_world_m.reshape(-1, 3, 3)).as_quat().reshape(T_full, -1, 4)
            # Root linear velocity (world frame)
            root_lin_vel = finite_diff_vel(root_pos, fps).astype(np.float32)
            # Joint velocities
            dof_pos = m['dof_pos'].astype(np.float32)
            joint_vel = finite_diff_vel(dof_pos, fps).astype(np.float32)
            # Root angular velocity (world frame): using central difference of ΔR = R_{t+1} @ R_{t-1}^T
            root_ang_vel = angvel_from_rot(R_root, fps=fps)

            # Assemble qpos, qvel (quaternions in qpos are arranged as wxyz to match subsequent usage)
            # Convert xyzw -> wxyz
            root_quat_wxyz = np.concatenate([root_quat_xyzw[:, 3:4], root_quat_xyzw[:, :3]], axis=-1)
            body_quat_w_wxyz = np.concatenate([body_quat_w_xyzw[..., 3:4], body_quat_w_xyzw[..., :3]], axis=-1)
            body_quat_b_wxyz = np.concatenate([local_quat_xyzw[..., 3:4], local_quat_xyzw[..., :3]], axis=-1)
            # Body linear velocity in world frame
            body_vel_w = finite_diff_vel(body_pos_w, fps).astype(np.float32)
            # Root frame body velocity / orientation
            body_vel_b = rotate_to_body(root_quat_wxyz, body_vel_w - root_lin_vel[:, None]).astype(np.float32)
            # Body angular velocity (root frame) from body_quat_b
            body_angvel_b = angvel_from_quat_wxyz_batch(body_quat_b_wxyz, fps=fps)
            # Body angular velocity in world frame
            body_angvel_w = np.einsum('tij,tbj->tbi', R_root_m, body_angvel_b) + root_ang_vel[:, None]

            qpos = np.concatenate([root_pos, root_quat_wxyz, dof_pos], axis=-1)
            qvel = np.concatenate([root_lin_vel, root_ang_vel, joint_vel], axis=-1)

            # Store back to motion dictionary for subsequent segmentation and downstream use
            m['xpos'] = body_pos_w.astype(np.float32)
            m['xquat'] = body_quat_w_wxyz.astype(np.float32)
            m['body_vel_w'] = body_vel_w.astype(np.float32)
            m['body_vel_b'] = body_vel_b.astype(np.float32)
            m['body_quat_b'] = body_quat_b_wxyz.astype(np.float32)
            m['body_angvel_w'] = body_angvel_w.astype(np.float32)
            m['body_angvel_b'] = body_angvel_b.astype(np.float32)
            m['qpos'] = qpos.astype(np.float32)
            m['qvel'] = qvel.astype(np.float32)
            
            # Process metadata and save motion
            T = m['root_pos'].shape[0];

            for start_idx in range(0, T, SEGMENT_LEN):
                end_idx   = min(start_idx + SEGMENT_LEN, T)
                m_seg = {}
                for k in preserved_keys:
                    m_seg[k] = m[k] if k == 'fps' else m[k][start_idx:end_idx]
                m_seg["joint_names"] = meta["joint_names"]
                m_seg["body_names"] = meta['body_names']

                if motion_processer is not None:
                    try:
                        m_seg = motion_processer(m_seg, foot_idx, p, start_idx, end_idx)
                    except TypeError:
                        m_seg = motion_processer(m_seg, foot_idx)

                if motion_filter is not None:
                    try:
                        ok = motion_filter(m_seg, foot_idx, p, start_idx, end_idx)
                    except TypeError:
                        ok = motion_filter(m_seg, foot_idx, p)
                    if not ok:
                        continue

                total += m_seg["qpos"].shape[0]  # Accumulate accepted frame count

                if callback is not None:
                    ctx = locals()
                    callback(ctx, m_seg)

                if build_dataset:
                    motions.append(m_seg)  # Append segmented motion
                    meta['info'].append(m_seg.get('metadata') or {})  # Synchronize metadata
                    id_labels.append({
                        "source_path": str(p),
                        "segment_start": int(start_idx),
                        "segment_end": int(end_idx),
                    })
            pb.set_postfix(total=total)

        if not build_dataset:
            return None, meta

        # Stack metadata
        meta_keys = []

        for k in meta['info'][0].keys():
            meta_keys.append(k)
        meta_keys = list(set(meta_keys))
        
        info = {}
        for k in meta_keys:
            info[k] = []
        for m in meta['info']:
            for k in meta_keys:
                info[k].append(m[k])
        for k in meta_keys:
            tmp = np.array(info[k])
            tmp = tmp.reshape(len(motions), -1)
            info[k] = tmp.tolist()
        
        meta['info'] = info

        # Pre-allocate memory-mapped tensors
        mm = {}
        mm['motion_id']      = MemoryMappedTensor.empty(total, dtype=storage_int_dtype)
        mm['step']           = MemoryMappedTensor.empty(total, dtype=storage_int_dtype)
        mm['root_pos_w']     = MemoryMappedTensor.empty(total, 3, dtype=storage_float_dtype)
        mm['root_quat_w']    = MemoryMappedTensor.empty(total, 4, dtype=storage_float_dtype)
        mm['root_lin_vel_w'] = MemoryMappedTensor.empty(total, 3, dtype=storage_float_dtype)
        mm['root_ang_vel_w'] = MemoryMappedTensor.empty(total, 3, dtype=storage_float_dtype)
        mm['joint_pos']      = MemoryMappedTensor.empty(total, len(meta['joint_names']), dtype=storage_float_dtype)
        mm['joint_vel']      = MemoryMappedTensor.empty(total, len(meta['joint_names']), dtype=storage_float_dtype)
        mm['body_pos_w']     = MemoryMappedTensor.empty(total, len(meta['body_names']), 3, dtype=storage_float_dtype)
        mm['body_pos_b']     = MemoryMappedTensor.empty(total, len(meta['body_names']), 3, dtype=storage_float_dtype)
        mm['body_vel_w']     = MemoryMappedTensor.empty(total, len(meta['body_names']), 3, dtype=storage_float_dtype)
        mm['body_vel_b']     = MemoryMappedTensor.empty(total, len(meta['body_names']), 3, dtype=storage_float_dtype)
        mm['body_quat_w']    = MemoryMappedTensor.empty(total, len(meta['body_names']), 4, dtype=storage_float_dtype)
        mm['body_quat_b']    = MemoryMappedTensor.empty(total, len(meta['body_names']), 4, dtype=storage_float_dtype)
        mm['body_angvel_w']  = MemoryMappedTensor.empty(total, len(meta['body_names']), 3, dtype=storage_float_dtype)
        mm['body_angvel_b']  = MemoryMappedTensor.empty(total, len(meta['body_names']), 3, dtype=storage_float_dtype)

        cursor = 0
        starts = []
        ends = []
        for i, m in enumerate(motions):
            T = m['qpos'].shape[0]
            # Root position and velocity
            root_pos_w     = m['qpos'][:, :3]
            root_quat_w    = m['qpos'][:, 3:7]
            root_lin_vel_w = m['qvel'][:, :3]
            root_ang_vel_w = m['qvel'][:, 3:6]
            # Joints
            J = len(meta['joint_names'])
            joint_pos = m['qpos'][:, 7:7+J]
            joint_vel = m['qvel'][:, 6:6+J]
            # Bodies
            B = len(meta['body_names'])
            body_pos_w     = m['xpos']
            # Body frame
            body_pos_b     = rotate_to_body(root_quat_w, body_pos_w - root_pos_w[:, None]).astype(np.float32)
            body_quat_w    = m['xquat']
            body_vel_w    = m['body_vel_w']
            body_vel_b    = m['body_vel_b']
            body_quat_b   = m['body_quat_b']
            body_angvel_w = m['body_angvel_w']
            body_angvel_b = m['body_angvel_b']
            # Fill tensors
            mm['step'          ][cursor:cursor+T] = torch.arange(T, dtype=storage_int_dtype)
            mm['motion_id'     ][cursor:cursor+T] = i
            mm['root_pos_w'    ][cursor:cursor+T] = torch.as_tensor(root_pos_w, dtype=storage_float_dtype)
            mm['root_quat_w'   ][cursor:cursor+T] = torch.as_tensor(root_quat_w, dtype=storage_float_dtype)
            mm['root_lin_vel_w'][cursor:cursor+T] = torch.as_tensor(root_lin_vel_w, dtype=storage_float_dtype)
            mm['root_ang_vel_w'][cursor:cursor+T] = torch.as_tensor(root_ang_vel_w, dtype=storage_float_dtype)
            mm['joint_pos'     ][cursor:cursor+T] = torch.as_tensor(joint_pos, dtype=storage_float_dtype)
            mm['joint_vel'     ][cursor:cursor+T] = torch.as_tensor(joint_vel, dtype=storage_float_dtype)
            mm['body_pos_w'    ][cursor:cursor+T] = torch.as_tensor(body_pos_w, dtype=storage_float_dtype)
            mm['body_pos_b'    ][cursor:cursor+T] = torch.as_tensor(body_pos_b, dtype=storage_float_dtype)
            mm['body_vel_w'    ][cursor:cursor+T] = torch.as_tensor(body_vel_w, dtype=storage_float_dtype)
            mm['body_vel_b'    ][cursor:cursor+T] = torch.as_tensor(body_vel_b, dtype=storage_float_dtype)
            mm['body_quat_w'   ][cursor:cursor+T] = torch.as_tensor(body_quat_w, dtype=storage_float_dtype)
            mm['body_quat_b'   ][cursor:cursor+T] = torch.as_tensor(body_quat_b, dtype=storage_float_dtype)
            mm['body_angvel_w' ][cursor:cursor+T] = torch.as_tensor(body_angvel_w, dtype=storage_float_dtype)
            mm['body_angvel_b' ][cursor:cursor+T] = torch.as_tensor(body_angvel_b, dtype=storage_float_dtype)

            starts.append(cursor)
            cursor += T
            ends.append(cursor)
        data = MotionData(**mm, batch_size=[total])
        
        # Save to mem_path
        if mem_path is not None:
            path = mem_path
            data.memmap(path)
            # Write metadata
            dump_data = {
                "body_names": meta['body_names'],
                "joint_names": meta['joint_names'],
                "starts": starts,
                "ends": ends,
                "info": meta['info']
            }
            with open(path + "/meta_motion.json", "w") as f:
                json.dump(dump_data, f)
            if id_labels is not None:
                with open(path + "/id_label.json", "w") as f:
                    json.dump(id_labels, f, ensure_ascii=True)
        
        return data, meta

    @property
    def num_motions(self):
        return len(self.starts)

    @property
    def num_steps(self):
        return len(self.data)

    def get_slice(self, motion_ids: torch.Tensor, starts: torch.Tensor, steps: int = 1) -> MotionData:
        motion_ids = motion_ids.to(dtype=torch.long, device=self.starts.device)
        starts = starts.to(dtype=torch.long, device=self.starts.device)
        if isinstance(steps, int):
            idx = (self.starts[motion_ids].to(torch.long) + starts).unsqueeze(1) + torch.arange(
                steps, device=self.starts.device, dtype=torch.long
            )
        else:
            idx = (self.starts[motion_ids].to(torch.long) + starts).unsqueeze(1) + steps.to(
                dtype=torch.long, device=self.starts.device
            )
        ends_per_motion = (self.ends[motion_ids].to(torch.long) - 1).unsqueeze(1)
        idx = idx.clamp(max=ends_per_motion)
        return self.data[idx]
