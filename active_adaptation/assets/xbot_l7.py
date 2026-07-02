from __future__ import annotations

import os
from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

from ._auto_symmetry import generate_auto_symmetry, load_spec_from_xml

ASSET_PATH = os.path.dirname(__file__)
XBOT_L7_XML = Path(ASSET_PATH) / "xbot_l7" / "mjcf" / "xbot_l7.xml"


def _get_spec() -> mujoco.MjSpec:
    return load_spec_from_xml(XBOT_L7_XML)


# Initial pose.
XBOT_L7_INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.93),
    joint_pos={".*": 0.0},
    joint_vel={".*": 0.0},
)


# Motor constants.
NATURAL_FREQ = 10.0 * 2.0 * 3.1415926535
DAMPING_RATIO = 2.0

ARMATURE_6508_100 = 0.137
ARMATURE_5005 = 0.01
ARMATURE_10520 = 0.16473
ARMATURE_9015 = 0.088
ARMATURE_15017 = 0.0968
ARMATURE_6008_30 = 0.0225

STIFFNESS_6508_100 = ARMATURE_6508_100 * NATURAL_FREQ**2
STIFFNESS_5005 = ARMATURE_5005 * NATURAL_FREQ**2
STIFFNESS_10520 = ARMATURE_10520 * NATURAL_FREQ**2
STIFFNESS_9015 = ARMATURE_9015 * NATURAL_FREQ**2
STIFFNESS_15017 = ARMATURE_15017 * NATURAL_FREQ**2
STIFFNESS_6008_30 = ARMATURE_6008_30 * NATURAL_FREQ**2

DAMPING_6508_100 = 2.0 * DAMPING_RATIO * ARMATURE_6508_100 * NATURAL_FREQ
DAMPING_5005 = 2.0 * DAMPING_RATIO * ARMATURE_5005 * NATURAL_FREQ
DAMPING_10520 = 2.0 * DAMPING_RATIO * ARMATURE_10520 * NATURAL_FREQ
DAMPING_9015 = 2.0 * DAMPING_RATIO * ARMATURE_9015 * NATURAL_FREQ
DAMPING_15017 = 2.0 * DAMPING_RATIO * ARMATURE_15017 * NATURAL_FREQ
DAMPING_6008_30 = 2.0 * DAMPING_RATIO * ARMATURE_6008_30 * NATURAL_FREQ

EFFORT_LIMIT_6508_100_WAIST = 110.0
EFFORT_LIMIT_6508_100_UPPER = 95.0
EFFORT_LIMIT_5005 = 35.0
EFFORT_LIMIT_10520 = 255.0
EFFORT_LIMIT_9015 = 100.0
EFFORT_LIMIT_15017 = 350.0
EFFORT_LIMIT_6008_30 = 50.0


# Actuators.
XBOT_L7_ACTUATOR_HIP_ROLL = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_hip_roll_joint",),
    armature=ARMATURE_10520,
    stiffness=200.0,
    damping=12.6,
    effort_limit=EFFORT_LIMIT_10520,
)
XBOT_L7_ACTUATOR_HIP_YAW = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_hip_yaw_joint",),
    armature=ARMATURE_9015,
    stiffness=80.0,
    damping=5.1,
    effort_limit=EFFORT_LIMIT_9015,
)
XBOT_L7_ACTUATOR_HIP_PITCH_KNEE = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_hip_pitch_joint", ".*_knee_joint"),
    armature=ARMATURE_15017,
    stiffness=200.0,
    damping=12.6,
    effort_limit=EFFORT_LIMIT_15017,
)
XBOT_L7_ACTUATOR_ANKLE = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
    armature=ARMATURE_6008_30 * 2.0,
    stiffness=30.0 * 2.0,
    damping=1.8 * 2.0,
    effort_limit=EFFORT_LIMIT_6008_30 * 2.0,
)
XBOT_L7_ACTUATOR_WAIST_YAW = BuiltinPositionActuatorCfg(
    target_names_expr=("waist_yaw_joint",),
    armature=ARMATURE_6508_100,
    stiffness=80.0,
    damping=5.1,
    effort_limit=EFFORT_LIMIT_6508_100_WAIST,
)
XBOT_L7_ACTUATOR_WAIST = BuiltinPositionActuatorCfg(
    target_names_expr=("waist_roll_joint", "waist_pitch_joint"),
    armature=ARMATURE_6508_100 * 2.0,
    stiffness=500.0,
    damping=20.0,
    effort_limit=EFFORT_LIMIT_6508_100_WAIST * 2.0,
)
XBOT_L7_ACTUATOR_UPPER_LARGE = BuiltinPositionActuatorCfg(
    target_names_expr=(
        ".*_shoulder_pitch_joint",
        ".*_shoulder_roll_joint",
        ".*_arm_yaw_joint",
        ".*_elbow_pitch_joint",
    ),
    armature=ARMATURE_6508_100,
    stiffness=200.0,
    damping=10.0,
    effort_limit=EFFORT_LIMIT_6508_100_UPPER,
)
XBOT_L7_ACTUATOR_UPPER_SMALL = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_elbow_yaw_joint", ".*_wrist_pitch_joint", ".*_wrist_roll_joint"),
    armature=ARMATURE_5005,
    stiffness=100.0,
    damping=5.0,
    effort_limit=EFFORT_LIMIT_5005,
)


# Collision config.
FULL_COLLISION = CollisionCfg(
    geom_names_expr=(".*_collision",),
    condim=3,
    priority=1,
    friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)


XBOT_L7_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        XBOT_L7_ACTUATOR_HIP_ROLL,
        XBOT_L7_ACTUATOR_HIP_YAW,
        XBOT_L7_ACTUATOR_HIP_PITCH_KNEE,
        XBOT_L7_ACTUATOR_ANKLE,
        XBOT_L7_ACTUATOR_WAIST_YAW,
        XBOT_L7_ACTUATOR_WAIST,
        XBOT_L7_ACTUATOR_UPPER_LARGE,
        XBOT_L7_ACTUATOR_UPPER_SMALL,
    ),
    soft_joint_pos_limit_factor=0.9,
)

XBOT_L7_JOINT_ORDER = (
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_hip_pitch_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_arm_yaw_joint",
    "left_elbow_pitch_joint",
    "left_elbow_yaw_joint",
    "left_wrist_pitch_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_arm_yaw_joint",
    "right_elbow_pitch_joint",
    "right_elbow_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
)

JOINT_SYMMETRY_MAP, SPATIAL_SYMMETRY_MAP = generate_auto_symmetry(
    xml_path=XBOT_L7_XML,
    spec_fn=_get_spec,
    body_pos_tol=1.0e-3,
    body_quat_tol=1.0e-6,
    fk_body_pos_tol=1.0e-3,
    fk_body_quat_tol=1.0e-6,
)

XBOT_L7_CFG = EntityCfg(
    init_state=XBOT_L7_INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=_get_spec,
    articulation=XBOT_L7_ARTICULATION,
)

XBOT_L7_CFG.joint_symmetry_mapping = JOINT_SYMMETRY_MAP
XBOT_L7_CFG.spatial_symmetry_mapping = SPATIAL_SYMMETRY_MAP
XBOT_L7_CFG.joint_name_order = XBOT_L7_JOINT_ORDER
