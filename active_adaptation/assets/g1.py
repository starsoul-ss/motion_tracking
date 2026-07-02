from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import FULL_COLLISION
from ._auto_symmetry import generate_auto_symmetry, load_spec_from_xml

ASSET_PATH = os.path.dirname(__file__)
G1_XML = Path(ASSET_PATH) / "G1" / "g1.xml"
G1_XML_5 = Path(ASSET_PATH) / "G1" / "g1_5.xml"
G1_FULL_COLLISION_ALL_PRIORITY = replace(
    FULL_COLLISION,
    condim=3,
    priority=1,
)


def _get_spec_from_xml(xml_path: Path) -> mujoco.MjSpec:
    return load_spec_from_xml(xml_path)


def _get_spec() -> mujoco.MjSpec:
    return _get_spec_from_xml(G1_XML)


def _get_spec_5() -> mujoco.MjSpec:
    return _get_spec_from_xml(G1_XML_5)

G1_INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.74),
    joint_pos={
        ".*_hip_pitch_joint": -0.28,
        ".*_knee_joint": 0.5,
        ".*_ankle_pitch_joint": -0.23,
        ".*_elbow_joint": 0.87,
        "left_shoulder_roll_joint": 0.16,
        "left_shoulder_pitch_joint": 0.35,
        "right_shoulder_roll_joint": -0.16,
        "right_shoulder_pitch_joint": 0.35,
        ".*_wrist_roll_joint": 0.0,
        ".*_wrist_pitch_joint": 0.0,
        ".*_wrist_yaw_joint": 0.0,
        ".*": 0.0,
    },
    joint_vel={".*": 0.0},
)

G1_ACTUATOR_UPPER = BuiltinPositionActuatorCfg(
    target_names_expr=(
        ".*_elbow_joint",
        ".*_shoulder_pitch_joint",
        ".*_shoulder_roll_joint",
        ".*_shoulder_yaw_joint",
        ".*_wrist_roll_joint",
    ),
    armature=0.003609725,
    stiffness=14.25062309787429,
    damping=0.907222843292423,
    effort_limit=25.0,
)
G1_ACTUATOR_HIP_YAW_WAIST_YAW = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_hip_yaw_joint", "waist_yaw_joint"),
    armature=0.010177520,
    stiffness=40.17923847137318,
    damping=2.5578897650279457,
    effort_limit=88.0,
)
G1_ACTUATOR_HIP_PITCH_ROLL_KNEE = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_hip_pitch_joint", ".*_hip_roll_joint", ".*_knee_joint"),
    armature=0.025101925,
    stiffness=99.09842777666113,
    damping=6.3088018534966395,
    effort_limit=139.0,
)
G1_ACTUATOR_WRIST_PITCH_YAW = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_wrist_pitch_joint", ".*_wrist_yaw_joint"),
    armature=0.0021812,
    stiffness=8.611032447370201,
    damping=0.548195351665136,
    effort_limit=13.4,
)
G1_ACTUATOR_WAIST = BuiltinPositionActuatorCfg(
    target_names_expr=("waist_pitch_joint", "waist_roll_joint"),
    armature=0.00721945,
    stiffness=28.50124619574858,
    damping=1.814445686584846,
    effort_limit=35.0,
)
G1_ACTUATOR_ANKLE = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
    armature=0.00721945,
    stiffness=28.50124619574858,
    damping=1.814445686584846,
    effort_limit=35.0,
)

G1_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        G1_ACTUATOR_UPPER,
        G1_ACTUATOR_HIP_PITCH_ROLL_KNEE,
        G1_ACTUATOR_HIP_YAW_WAIST_YAW,
        G1_ACTUATOR_WRIST_PITCH_YAW,
        G1_ACTUATOR_WAIST,
        G1_ACTUATOR_ANKLE,
    ),
    soft_joint_pos_limit_factor=0.9,
)

G1_ACTUATOR_HIP_PITCH_5 = replace(
    G1_ACTUATOR_HIP_PITCH_ROLL_KNEE,
    target_names_expr=(".*_hip_pitch_joint",),
    armature=0.010177520,
    effort_limit=88.0,
)
G1_ACTUATOR_HIP_ROLL_KNEE_5 = replace(
    G1_ACTUATOR_HIP_PITCH_ROLL_KNEE,
    target_names_expr=(".*_hip_roll_joint", ".*_knee_joint"),
)
G1_ACTUATOR_WRIST_PITCH_YAW_5 = replace(
    G1_ACTUATOR_WRIST_PITCH_YAW,
    armature=0.00425,
    effort_limit=5.0,
)

G1_ARTICULATION_5 = EntityArticulationInfoCfg(
    actuators=(
        G1_ACTUATOR_UPPER,
        G1_ACTUATOR_HIP_PITCH_5,
        G1_ACTUATOR_HIP_ROLL_KNEE_5,
        G1_ACTUATOR_HIP_YAW_WAIST_YAW,
        G1_ACTUATOR_WRIST_PITCH_YAW_5,
        G1_ACTUATOR_WAIST,
        G1_ACTUATOR_ANKLE,
    ),
    soft_joint_pos_limit_factor=0.9,
)

G1_JOINT_ORDER = ('left_hip_pitch_joint', 'right_hip_pitch_joint', 'waist_yaw_joint', 'left_hip_roll_joint', 'right_hip_roll_joint', 'waist_roll_joint', 'left_hip_yaw_joint', 'right_hip_yaw_joint', 'waist_pitch_joint', 'left_knee_joint', 'right_knee_joint', 'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint', 'left_ankle_pitch_joint', 'right_ankle_pitch_joint', 'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 'left_ankle_roll_joint', 'right_ankle_roll_joint', 'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint', 'left_elbow_joint', 'right_elbow_joint', 'left_wrist_roll_joint', 'right_wrist_roll_joint', 'left_wrist_pitch_joint', 'right_wrist_pitch_joint', 'left_wrist_yaw_joint', 'right_wrist_yaw_joint')

JOINT_SYMMETRY_MAP, SPATIAL_SYMMETRY_MAP = generate_auto_symmetry(
    xml_path=G1_XML,
    spec_fn=_get_spec,
    body_pos_tol=1.0e-3,
    body_quat_tol=1.0e-6,
    fk_body_pos_tol=1.0e-3,
    fk_body_quat_tol=1.0e-6,
)

G1_CFG = EntityCfg(
    init_state=G1_INIT_STATE,
    collisions=(G1_FULL_COLLISION_ALL_PRIORITY,),
    spec_fn=_get_spec,
    articulation=G1_ARTICULATION,
)
G1_CFG_5 = EntityCfg(
    init_state=G1_INIT_STATE,
    collisions=(G1_FULL_COLLISION_ALL_PRIORITY,),
    spec_fn=_get_spec_5,
    articulation=G1_ARTICULATION_5,
)

G1_CFG.joint_symmetry_mapping = JOINT_SYMMETRY_MAP
G1_CFG.spatial_symmetry_mapping = SPATIAL_SYMMETRY_MAP
G1_CFG.joint_name_order = G1_JOINT_ORDER
G1_CFG_5.joint_symmetry_mapping = JOINT_SYMMETRY_MAP
G1_CFG_5.spatial_symmetry_mapping = SPATIAL_SYMMETRY_MAP
G1_CFG_5.joint_name_order = G1_JOINT_ORDER
