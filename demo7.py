# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab 3.0+: SO-101 アームで「ずれたパレットを突いて枠内に戻す」デモ。

PLC で組まれた自動化ラインの嵌め込み (圧入・嵌合) 工程では、搬送されてきたパレットが
位置決め枠から少しずれて止まり、センサが NG を出してラインが止まる — というトラブルがよく起きる。
このデモはその復旧作業を SO-101 にやらせる。

* 台 (テーブル) の上に SO-101 を固定し、その前に **位置決め枠** (黄色いテープの四角) を引いてある。
* パレットは平べったい直方体。初期状態で枠からずれた位置・向きに置かれている (``--offset_*``)。
* アームはパレットを **挟まず**、閉じたグリッパの指先でパレットの側面を **ゆっくり押して** 枠内に戻す。
* 枠の横の信号灯 (アンドン) が 赤 = ライン停止 (ずれ検出) / 黄 = 自動復旧中 / 緑 = 位置決め OK を示す。

押し方 (スクリプト制御)
------------------------

毎回「計測 → 押し方を決める → 1 回押す → 離れて計測し直す」を繰り返すクローズドループ:

1. **計測**: パレットの位置・ヨー角をシミュレータから取得する (実機ならカメラ/変位センサの役)。
   枠中心からの位置ずれ ``e`` と角度ずれ ``e_yaw`` (パレットは 180 deg 対称なので ±90 deg に畳む) を出す。
2. **判断**:

   * 角度ずれが :data:`YAW_TOL_DEG` を超えていたら **回転押し**: パレットの側面を、中心から外れた点
     (角寄り) で押してモーメントを作る。8 通りの候補 (4 側面 × 左右の端) のうち、回したい向きの
     モーメントが出て、かつ押す方向が位置ずれも一緒に減らす向きのものを選ぶ。
   * 角度が揃っていたら **並進押し**: パレットの軸 (長辺/短辺) ごとに、ずれの大きい方から
     側面の中央を押して真っ直ぐ動かす。
   * アームが届かない候補 (根元に近すぎる / 遠すぎる) は除外する。

3. **押す**: 押し始め位置の上空 → 降下 → 押し方向に :data:`PUSH_SPEED` でゆっくり前進。
   押している間もパレットを監視し、**残りずれ ≤ パレット速度 × サーボ遅れ** になった時点で止める
   (止めた後もアームが遅れぶん進むので、その分を先読みしてオーバーシュートを防ぐ)。
4. 少し引いて上昇し、1 に戻る。位置・角度とも許容値に入るか、4 隅すべてが枠の内側に
   :data:`DONE_MARGIN` 以上の余裕で収まったら完了してホームに戻る。

この「状態 (ずれ量) → 行動 (どこを・どの向きに・どれだけ押すか)」という形は、そのまま
強化学習の観測と行動の設計に流用できるようにしてある (次のステップで RL 化する想定)。

.. code-block:: bash

    # 既定の 1 パターン (手前寄り + 左ずれ + 14 deg 回転) で実行
    isaaclab.bat -p demo7.py --viz kit
    ./isaaclab.sh -p demo7.py --viz kit

    # ずれ方・重さを変えてみる
    isaaclab.bat -p demo7.py --viz kit --offset_x 0.02 --offset_y -0.015 --offset_yaw -10 --pallet_mass 0.5

    # 画面なしでログだけ
    isaaclab.bat -p demo7.py --viz none
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="SO-101 nudges a misaligned pallet back into its locating frame.")
parser.add_argument(
    "--offset_x", type=float, default=-0.015, help="Initial pallet offset [m] from the frame centre along world x."
)
parser.add_argument(
    "--offset_y", type=float, default=0.022, help="Initial pallet offset [m] from the frame centre along world y."
)
parser.add_argument("--offset_yaw", type=float, default=14.0, help="Initial pallet yaw offset [deg] from the frame.")
parser.add_argument("--pallet_mass", type=float, default=0.25, help="Mass [kg] of the pallet.")
parser.add_argument(
    "--friction", type=float, default=0.45, help="Static friction coefficient between the pallet and the table."
)
parser.add_argument("--push_speed", type=float, default=0.012, help="Speed [m/s] the fingertip pushes the pallet at.")
parser.add_argument("--max_pushes", type=int, default=15, help="Give up after this many pushes.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# demos should open the Kit visualizer by default
parser.set_defaults(visualizer=["kit"])
# parse the arguments
args_cli = parser.parse_args()

# the visualizer selection is resolved after the headless flag, so drop the Kit default when the user
# asks for --headless (otherwise a window is still requested).
if args_cli.headless:
    args_cli.visualizer = ["none"]

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import enum
import math
import numpy as np
import torch
from dataclasses import dataclass

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_apply, quat_apply_inverse, subtract_frame_transforms

from isaaclab_assets.robots.so101 import SO101_HIGH_PD_CFG

from isaaclab_tasks.contrib.stack.config.so101.pose_ik_controller import (
    SO101PoseIKController,
    SO101PoseIKControllerCfg,
)

##
# 物理・描画レート
##

PHYSICS_RATE = 120.0
"""Physics steps per second [Hz]."""
RENDER_INTERVAL = 2
"""Physics steps per rendered frame (60 Hz viewer at :data:`PHYSICS_RATE` = 120 Hz)."""

ARM_STIFFNESS = 600.0
"""Position-loop stiffness [N·m/rad] of the arm joints. Stiff enough that the push reaction barely deflects the tip."""
ARM_DAMPING = 60.0
"""Position-loop damping [N·m·s/rad] of the arm joints.

``damping / stiffness`` = 0.1 s servo lag (see :data:`SERVO_LAG_S`).
"""
SERVO_LAG_S = 0.1
"""Approximate lag [s] between the commanded and the achieved fingertip position.

Used by the predictive stop: once the push command halts, the fingertip still travels about
``speed * SERVO_LAG_S`` before it comes to rest, and the pallet goes with it.
"""

##
# シーンの寸法 [m] (ワールド座標)
##

TABLE_SIZE = (0.70, 0.60, 0.40)
"""Size (x, y, z) [m] of the work table."""
TABLE_CENTER_XY = (0.15, 0.0)
"""Table centre (x, y) [m] in world coordinates."""
TABLE_TOP = TABLE_SIZE[2]
"""Height [m] of the table surface."""
SO101_MOUNT_Z = -0.03008
"""Offset [m] from the table surface to the SO-101 root so its column foot sits on the table
(the USD base has its foot ~30 mm above the root; same value as the SO-101 cube-stack task)."""

PALLET_SIZE = (0.08, 0.06, 0.03)
"""Pallet edge lengths (x, y, z) [m] in its own frame. Flat box, long side along local x."""
FRAME_CENTER_XY = (0.22, 0.0)
"""Centre (x, y) [m] of the locating frame in world coordinates (in front of the arm)."""
FRAME_YAW_DEG = 90.0
"""Yaw [deg] of the locating frame in world coordinates.

90 deg puts the pallet's long side across the front of the arm (the line runs left to right), which
keeps every pushing face inside the SO-101's short reach: the near face is not too close to the
base and the far face is not too far.
"""
FRAME_CLEARANCE = 0.005
"""Gap [m] per side between the pallet and the inside edge of the frame tape."""
FRAME_LINE_WIDTH = 0.005
"""Width [m] of the frame tape."""

LAMP_POS = (0.30, -0.20, TABLE_TOP + 0.14)
"""Position [m] of the signal lamp (andon) beside the frame."""

##
# SO-101 の関節とツール点
##

ARM_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
"""Actuated arm joints, in the order the IK controller sees them."""
ORIENTATION_JOINT_NAMES = ("wrist_flex", "wrist_roll")
"""Joints allowed to serve the orientation rows of the IK task, so ``shoulder_pan`` serves position only."""
EE_BODY_NAME = "gripper"
"""Static wrist/gripper body the IK tracks."""
GRIPPER_JOINT_NAME = "gripper"
"""The single revolute jaw joint, held shut: the arm pushes with the closed jaws, it never grasps."""
GRIPPER_CLOSED_RAD = 0.0
"""Jaw angle [rad] we hold the gripper at."""
TCP_OFFSET = (0.0, 0.0, -0.092)
"""Tool point [m] in the ``gripper`` body frame: on the closed jaws, ~12 mm above the fingertips.

Measured from the SO-101 USD: the jaws extend to about -0.104 m along the gripper body's local -z.
"""
SO101_INIT_JOINT_POS: dict[str, float] = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -0.6,
    "elbow_flex": 0.8,
    "wrist_flex": 0.6,
    "wrist_roll": 0.0,
    "gripper": GRIPPER_CLOSED_RAD,
}
"""Starting joint pose [rad] (from the SO-101 cube-stack task)."""
SO101_BASE_ROT = (0.0, 0.0, 0.70710678, 0.70710678)
"""Root orientation (x, y, z, w): 90 deg about +z so the arm (which reaches along its base -y) faces world +x."""

##
# 押し動作のパラメータ
##

PUSH_HEIGHT = 0.017
"""Height [m] of the tool point above the table while pushing (fingertips ~5 mm above the table)."""
TRAVEL_HEIGHT = 0.085
"""Height [m] of the tool point above the table when moving between pushes (clears the pallet top)."""
HOME_OFFSET = (-0.06, 0.0, 0.13)
"""Home tool point [m] relative to the frame centre on the table surface."""
FINGER_TIP_HALF = 0.010
"""Half-thickness [m] of the closed jaws at push height, measured in sim (contact ~50 mm into a 60 mm approach)."""
STANDOFF = 0.035
"""Distance [m] from the pallet face to the tool point where the descent happens (~25 mm of air to the jaws)."""
APPROACH_SPEED = 0.03
"""Speed [m/s] for the air gap before contact. Slows to ``--push_speed`` shortly before the face."""
SLOWDOWN_BEFORE_FACE = 0.008
"""Distance [m] before the expected face where the approach slows down to the push speed."""
CONTACT_SEARCH = STANDOFF + 0.012
"""Longest distance [m] the fingertip advances looking for contact before the push is abandoned."""
CONTACT_MOVE = 0.0005
"""Pallet displacement [m] that counts as "the fingertip is now in contact"."""
CONTACT_TURN_DEG = 0.2
"""Pallet rotation [deg] that counts as "the fingertip is now in contact"."""
BACKOFF = 0.015
"""Distance [m] the fingertip retreats along the push axis before rising."""
TRAVEL_SPEED = 0.08
"""Speed [m/s] for free-space moves."""
DESCEND_SPEED = 0.04
"""Speed [m/s] for vertical moves."""
ROT_LEVER_FRACTION = 0.75
"""Where along a face the rotation push lands, as a fraction of the half-length from the face centre."""
POS_TOL = 0.0025
"""Position tolerance [m] (per pallet axis) the controller aims for."""
YAW_TOL_DEG = 1.5
"""Yaw tolerance [deg] the controller aims for."""
DONE_MARGIN = 0.0015
"""The job also counts as done once every pallet corner is at least this far [m] inside the frame tape."""
REACH_MIN = 0.11
"""Minimum horizontal distance [m] from the shoulder the tool point may be commanded to."""
REACH_MAX = 0.30
"""Maximum horizontal distance [m] from the shoulder the tool point may be commanded to."""
INSPECT_S = 0.4
"""Settle time [s] before every measurement."""
FINAL_HOLD_S = 2.0
"""Time [s] to hold at home after the pallet is in place."""


##
# ヘルパ
##


def yaw_quat(yaw: float) -> np.ndarray:
    """Quaternion (x, y, z, w) for a rotation of ``yaw`` [rad] about +z."""
    return np.array([0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw)])


def quat_to_yaw(quat: np.ndarray) -> float:
    """Yaw [rad] of a quaternion (x, y, z, w)."""
    x, y, z, w = (float(v) for v in quat)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap_half_turn(angle: float) -> float:
    """Wrap ``angle`` [rad] to [-pi/2, pi/2): a rectangle looks the same after a half turn."""
    return (angle + 0.5 * math.pi) % math.pi - 0.5 * math.pi


def rot2(angle: float) -> np.ndarray:
    """2-D rotation matrix for ``angle`` [rad]."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s], [s, c]])


def quat_from_z_axis(vec: np.ndarray) -> np.ndarray:
    """Shortest-arc quaternion (x, y, z, w) rotating +z onto ``vec`` (used to orient the cone marker)."""
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0])
    unit = vec / norm
    cos_angle = float(unit[2])
    if cos_angle < -1.0 + 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0])
    quat = np.array([-unit[1], unit[0], 0.0, 1.0 + cos_angle])
    return quat / float(np.linalg.norm(quat))


@dataclass
class PalletError:
    """Pallet pose relative to the locating frame, as the controller sees it."""

    pos_w: np.ndarray
    """Pallet centre minus frame centre, in world (x, y) [m], shape (2,)."""
    pos_local: np.ndarray
    """The same offset expressed along the pallet's own (x, y) axes [m], shape (2,)."""
    yaw: float
    """Pallet yaw minus frame yaw, folded to [-90, 90) deg, in [rad]."""
    margin: float
    """Smallest distance [m] from any pallet corner to the inner edge of the frame tape (negative = sticking out)."""

    @property
    def in_frame(self) -> bool:
        """Whether all four pallet corners lie inside the inner edge of the frame tape."""
        return self.margin >= 0.0

    def summary(self) -> str:
        """One-line human readable description."""
        return (
            f"ずれ x={self.pos_w[0] * 1000:+6.1f} mm, y={self.pos_w[1] * 1000:+6.1f} mm,"
            f" yaw={math.degrees(self.yaw):+6.2f} deg  -> {'枠内' if self.in_frame else '枠外'}"
            f" (余裕 {self.margin * 1000:+5.1f} mm)"
        )


@dataclass
class PushPlan:
    """One push primitive: where the fingertip touches the pallet, which way it pushes, and why."""

    kind: str
    """``rotate`` or ``translate``."""
    contact_xy: np.ndarray
    """Point [m] on the pallet face the push aims at, world (x, y), shape (2,)."""
    direction: np.ndarray
    """Unit push direction in world (x, y), shape (2,)."""
    follow_travel: float
    """Longest distance [m] the fingertip may advance once it is in contact with the pallet."""
    label: str
    """Description for the log."""

    @property
    def start_xy(self) -> np.ndarray:
        """Standoff point [m] where the fingertip descends, world (x, y), shape (2,)."""
        return self.contact_xy - self.direction * STANDOFF


class Phase(enum.Enum):
    """States of the nudging mission."""

    GO_HOME = enum.auto()
    INSPECT = enum.auto()
    MOVE_ABOVE = enum.auto()
    DESCEND = enum.auto()
    PUSH = enum.auto()
    BACKOFF = enum.auto()
    RISE = enum.auto()
    RETURN_HOME = enum.auto()
    DONE = enum.auto()


##
# シーン定義
##


def _frame_tape_cfgs() -> dict[str, AssetBaseCfg]:
    """Four thin, visual-only boxes drawing the locating frame on the table."""
    inner_x = PALLET_SIZE[0] + 2.0 * FRAME_CLEARANCE
    inner_y = PALLET_SIZE[1] + 2.0 * FRAME_CLEARANCE
    w = FRAME_LINE_WIDTH
    yaw = math.radians(FRAME_YAW_DEG)
    rot = rot2(yaw)
    quat = tuple(float(v) for v in yaw_quat(yaw))
    pieces = {
        # (local centre, size)
        "frame_px": ((0.5 * inner_x + 0.5 * w, 0.0), (w, inner_y + 2.0 * w)),
        "frame_nx": ((-0.5 * inner_x - 0.5 * w, 0.0), (w, inner_y + 2.0 * w)),
        "frame_py": ((0.0, 0.5 * inner_y + 0.5 * w), (inner_x, w)),
        "frame_ny": ((0.0, -0.5 * inner_y - 0.5 * w), (inner_x, w)),
    }
    cfgs: dict[str, AssetBaseCfg] = {}
    for name, (center_local, size_xy) in pieces.items():
        center = np.array(FRAME_CENTER_XY) + rot @ np.array(center_local)
        cfgs[name] = AssetBaseCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Frame/{name}",
            # no collision props: purely painted on the table
            spawn=sim_utils.CuboidCfg(
                size=(size_xy[0], size_xy[1], 0.0008),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.80, 0.05), roughness=0.9),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(float(center[0]), float(center[1]), TABLE_TOP + 0.0004), rot=quat
            ),
        )
    return cfgs


_FRAME_TAPES = _frame_tape_cfgs()


@configclass
class NudgeSceneCfg(InteractiveSceneCfg):
    """Ground, light, work table, table-mounted SO-101, locating frame, pallet and signal post."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.92, 0.92, 0.96))
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=TABLE_SIZE,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=args_cli.friction, dynamic_friction=args_cli.friction * 0.85, restitution=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.57, 0.60), roughness=0.8),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(TABLE_CENTER_XY[0], TABLE_CENTER_XY[1], 0.5 * TABLE_SIZE[2])),
    )

    frame_px = _FRAME_TAPES["frame_px"]
    frame_nx = _FRAME_TAPES["frame_nx"]
    frame_py = _FRAME_TAPES["frame_py"]
    frame_ny = _FRAME_TAPES["frame_ny"]

    lamp_post = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/LampPost",
        spawn=sim_utils.CylinderCfg(
            radius=0.006,
            height=LAMP_POS[2] - TABLE_TOP,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.15, 0.15), roughness=0.6),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(LAMP_POS[0], LAMP_POS[1], 0.5 * (TABLE_TOP + LAMP_POS[2]))
        ),
    )

    # SO-101 seated on the table, facing world +x. High-PD variant (gravity off on the links) for
    # task-space tracking, with gains retuned for pushing (see ARM_STIFFNESS / ARM_DAMPING).
    robot = SO101_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=SO101_HIGH_PD_CFG.init_state.replace(
            pos=(0.0, 0.0, TABLE_TOP + SO101_MOUNT_Z),
            rot=SO101_BASE_ROT,
            joint_pos=dict(SO101_INIT_JOINT_POS),
        ),
    )
    robot.actuators["arm"].stiffness = ARM_STIFFNESS
    robot.actuators["arm"].damping = ARM_DAMPING

    pallet = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Pallet",
        spawn=sim_utils.CuboidCfg(
            size=PALLET_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=0.5),
            mass_props=sim_utils.MassPropertiesCfg(mass=args_cli.pallet_mass),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=args_cli.friction, dynamic_friction=args_cli.friction * 0.85, restitution=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.35, 0.75), roughness=0.6),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(
                FRAME_CENTER_XY[0] + args_cli.offset_x,
                FRAME_CENTER_XY[1] + args_cli.offset_y,
                TABLE_TOP + 0.5 * PALLET_SIZE[2] + 0.001,
            ),
            rot=tuple(float(v) for v in yaw_quat(math.radians(FRAME_YAW_DEG + args_cli.offset_yaw))),
        ),
    )


LAMP_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/Lamp",
    markers={
        "stop": sim_utils.SphereCfg(
            radius=0.022, visual_material=sim_utils.PreviewSurfaceCfg(emissive_color=(1.0, 0.05, 0.05))
        ),
        "recover": sim_utils.SphereCfg(
            radius=0.022, visual_material=sim_utils.PreviewSurfaceCfg(emissive_color=(1.0, 0.75, 0.0))
        ),
        "ok": sim_utils.SphereCfg(
            radius=0.022, visual_material=sim_utils.PreviewSurfaceCfg(emissive_color=(0.05, 1.0, 0.2))
        ),
    },
)
"""Signal lamp: red = line stopped, amber = automatic recovery in progress, green = pallet located."""
LAMP_STOP, LAMP_RECOVER, LAMP_OK = 0, 1, 2

PUSH_ARROW_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/PushArrow",
    markers={
        "arrow": sim_utils.ConeCfg(
            radius=0.008,
            height=0.025,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.25, 0.10), roughness=0.5),
        )
    },
)
"""Flat cone on the table showing the planned push point and direction."""


##
# ミッション実行
##


class NudgeDemo:
    """Owns the sim handles and runs the measure -> plan -> push loop."""

    def __init__(self, sim: SimulationContext, scene: InteractiveScene):
        self.sim = sim
        self.scene = scene
        self.dt = sim.get_physics_dt()
        self.device = sim.device
        self.robot: Articulation = scene["robot"]
        self.pallet: RigidObject = scene["pallet"]

        # --- resolve joints / bodies ------------------------------------------------------------
        self.arm_cfg = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES, body_names=[EE_BODY_NAME])
        self.arm_cfg.resolve(scene)
        self.gripper_cfg = SceneEntityCfg("robot", joint_names=[GRIPPER_JOINT_NAME])
        self.gripper_cfg.resolve(scene)
        self.ee_body_id = self.arm_cfg.body_ids[0]
        self.shoulder_body_id = self.robot.body_names.index("shoulder")
        # the root body is not in the Jacobian, so a fixed-base robot's body index shifts down by one
        self.ee_jacobi_idx = self.ee_body_id - 1 if self.robot.is_fixed_base else self.ee_body_id
        self.jacobi_joint_ids = [j + self.robot.num_base_dofs for j in self.arm_cfg.joint_ids]

        # --- IK controller ------------------------------------------------------------------------
        ik_cfg = SO101PoseIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="adaptive_dls",
            ik_params={"lambda_min": 0.05, "lambda_max": 0.2, "sigma_thresh": 0.02},
            orientation_weight=0.5,
            orientation_joint_names=ORIENTATION_JOINT_NAMES,
        )
        self.ik = SO101PoseIKController(ik_cfg, num_envs=scene.num_envs, device=self.device)
        resolved_joint_names = [self.robot.joint_names[i] for i in self.arm_cfg.joint_ids]
        mask = torch.tensor(
            [1.0 if name in ORIENTATION_JOINT_NAMES else 0.0 for name in resolved_joint_names], device=self.device
        )
        self.ik.set_orientation_joint_mask(mask)
        self.gripper_target = torch.full(
            (scene.num_envs, len(self.gripper_cfg.joint_ids)), GRIPPER_CLOSED_RAD, device=self.device
        )
        self.tcp_offset = torch.tensor([TCP_OFFSET], dtype=torch.float32, device=self.device)

        # --- markers ------------------------------------------------------------------------------
        self.lamp_marker = VisualizationMarkers(LAMP_MARKER_CFG)
        self.arrow_marker = VisualizationMarkers(PUSH_ARROW_CFG)
        self.render_enabled = not args_cli.headless and "none" not in (args_cli.visualizer or [])

        # --- task geometry ------------------------------------------------------------------------
        self.frame_xy = np.array(FRAME_CENTER_XY)
        self.frame_yaw = math.radians(FRAME_YAW_DEG)
        self.half = np.array(PALLET_SIZE[:2]) * 0.5
        self.home_tcp_w = np.array(
            [FRAME_CENTER_XY[0] + HOME_OFFSET[0], FRAME_CENTER_XY[1] + HOME_OFFSET[1], TABLE_TOP + HOME_OFFSET[2]]
        )
        self.shoulder_xy_w = np.zeros(2)

        # --- mission state ------------------------------------------------------------------------
        self.command_tcp_w = np.zeros(3)
        self.phase = Phase.GO_HOME
        self.phase_time = 0.0
        self.plan: PushPlan | None = None
        self.push_travel = 0.0
        self.contact_travel: float | None = None
        self._push_start_yaw = 0.0
        self._push_start_xy = np.zeros(2)
        self._backoff_goal = np.zeros(3)
        self.push_count = 0
        self.lamp_state = LAMP_STOP
        self.initial_error: PalletError | None = None
        self.final_error: PalletError | None = None
        self.success = False

    ##
    # 座標変換・計測
    ##

    def _root_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        root_pose = self.robot.data.root_pose_w.torch
        return root_pose[:, 0:3], root_pose[:, 3:7]

    def _tcp_world(self) -> np.ndarray:
        """Current tool point [m] in world coordinates, shape (3,)."""
        pose = self.robot.data.body_pose_w.torch[:, self.ee_body_id]
        tcp = pose[:, 0:3] + quat_apply(pose[:, 3:7], self.tcp_offset)
        return tcp[0].cpu().numpy().astype(float)

    def measure(self) -> PalletError:
        """Measure the pallet pose against the locating frame (stand-in for a vision sensor)."""
        pos = self.pallet.data.root_pos_w.torch[0].cpu().numpy().astype(float)
        quat = self.pallet.data.root_quat_w.torch[0].cpu().numpy().astype(float)
        pallet_yaw = quat_to_yaw(quat)
        err_w = pos[:2] - self.frame_xy
        yaw_err = wrap_half_turn(pallet_yaw - self.frame_yaw)
        # the axes the pusher can act along are the pallet's own axes (pushing face centres)
        err_local = rot2(pallet_yaw).T @ err_w

        # corners against the inner edge of the tape, in the frame's coordinates
        inner_half = self.half + FRAME_CLEARANCE
        corners_local = np.array([[sx * self.half[0], sy * self.half[1]] for sx in (-1, 1) for sy in (-1, 1)])
        corners_w = pos[:2] + corners_local @ rot2(pallet_yaw).T
        corners_frame = (corners_w - self.frame_xy) @ rot2(self.frame_yaw)
        margin = float(np.min(inner_half - np.abs(corners_frame)))
        return PalletError(pos_w=err_w, pos_local=err_local, yaw=yaw_err, margin=margin)

    def _pallet_pose_xy(self) -> tuple[np.ndarray, float]:
        pos = self.pallet.data.root_pos_w.torch[0].cpu().numpy().astype(float)
        quat = self.pallet.data.root_quat_w.torch[0].cpu().numpy().astype(float)
        return pos[:2], quat_to_yaw(quat)

    def _reachable(self, xy: np.ndarray) -> bool:
        reach = float(np.linalg.norm(xy - self.shoulder_xy_w))
        return REACH_MIN <= reach <= REACH_MAX

    ##
    # 押し方の決定
    ##

    @staticmethod
    def within_tolerance(err: PalletError) -> bool:
        """Whether the pallet is located well enough that no further push is needed."""
        centred = abs(math.degrees(err.yaw)) <= YAW_TOL_DEG and bool(np.all(np.abs(err.pos_local) <= POS_TOL))
        return centred or err.margin >= DONE_MARGIN

    def plan_push(self, err: PalletError) -> PushPlan | None:
        """Pick the next push from the current error, or ``None`` when no reachable push exists."""
        center, pallet_yaw = self._pallet_pose_xy()
        rot = rot2(pallet_yaw)
        axes = [rot[:, 0], rot[:, 1]]  # pallet local x / y in world

        # ---- 1) 角度ずれ: 角寄りを押して回す ------------------------------------------------------
        if abs(math.degrees(err.yaw)) > YAW_TOL_DEG:
            best: tuple[float, PushPlan] | None = None
            for normal_axis in (0, 1):
                tangent_axis = 1 - normal_axis
                for normal_sign in (1.0, -1.0):
                    n = normal_sign * axes[normal_axis]
                    for lever_sign in (1.0, -1.0):
                        lever = lever_sign * ROT_LEVER_FRACTION * self.half[tangent_axis] * axes[tangent_axis]
                        r = self.half[normal_axis] * n + lever
                        d = -n
                        torque = float(r[0] * d[1] - r[1] * d[0])  # z of r x d (per unit force)
                        # must turn the pallet back towards the frame
                        if torque * err.yaw >= 0.0:
                            continue
                        contact = center + r
                        plan = PushPlan(
                            kind="rotate",
                            contact_xy=contact,
                            direction=d,
                            follow_travel=1.6 * abs(err.yaw) * float(np.linalg.norm(lever)) + 0.010,
                            label=f"回転押し (yaw {math.degrees(err.yaw):+.1f} deg を戻す, 腕長 {abs(torque) * 1000:.0f} mm)",
                        )
                        end = contact + d * plan.follow_travel
                        if not (self._reachable(plan.start_xy) and self._reachable(end)):
                            continue
                        # prefer pushes that also take out the position error, then a longer lever
                        score = float(np.dot(d, -err.pos_w)) + 0.3 * abs(torque)
                        if best is None or score > best[0]:
                            best = (score, plan)
            if best is not None:
                return best[1]
            print("[WARN] 届く回転押しの候補がありません。並進補正を先に行います。")

        # ---- 2) 位置ずれ: ずれの大きい軸から、側面の中央を押す ---------------------------------
        order = np.argsort(-np.abs(err.pos_local))
        for axis_id in order:
            component = float(err.pos_local[axis_id])
            if abs(component) <= POS_TOL:
                continue
            d = -math.copysign(1.0, component) * axes[axis_id]
            contact = center - d * self.half[axis_id]
            plan = PushPlan(
                kind="translate",
                contact_xy=contact,
                direction=d,
                follow_travel=1.5 * abs(component) + 0.010,
                label=f"並進押し (パレット{'長' if axis_id == 0 else '短'}辺方向に {abs(component) * 1000:.1f} mm)",
            )
            end = contact + d * plan.follow_travel
            if self._reachable(plan.start_xy) and self._reachable(end):
                return plan
            print(f"[WARN] {plan.label} はアームの可動範囲外のためスキップします。")
        return None

    def push_should_stop(self, plan: PushPlan, err: PalletError) -> bool:
        """Predictive stop: halt once the remaining error is what the servo lag will still deliver."""
        lin_vel = self.pallet.data.root_lin_vel_w.torch[0, :2].cpu().numpy().astype(float)
        yaw_rate = float(self.pallet.data.root_ang_vel_w.torch[0, 2])
        if plan.kind == "rotate":
            remaining = abs(err.yaw)
            if err.yaw * self._push_start_yaw <= 0.0:
                return True  # overshot through zero
            closing_rate = -yaw_rate * math.copysign(1.0, err.yaw)
            return remaining <= max(closing_rate, 0.0) * SERVO_LAG_S + math.radians(0.3)
        remaining = float(np.dot(-err.pos_w, plan.direction))
        closing_rate = float(np.dot(lin_vel, plan.direction))
        return remaining <= max(closing_rate, 0.0) * SERVO_LAG_S + 0.0005

    ##
    # 指令生成
    ##

    def _move_command(self, goal_w: np.ndarray, speed: float) -> bool:
        """Advance the commanded tool point towards ``goal_w`` at ``speed`` [m/s]. Returns True on arrival."""
        delta = goal_w - self.command_tcp_w
        dist = float(np.linalg.norm(delta))
        step = speed * self.dt
        if dist <= step:
            self.command_tcp_w = goal_w.copy()
            return True
        self.command_tcp_w = self.command_tcp_w + delta * (step / dist)
        return False

    def _apply_ik(self) -> None:
        """Turn the commanded tool point into joint targets (gripper pointing straight down)."""
        root_pos, root_quat = self._root_pose()
        tcp_w = torch.tensor(self.command_tcp_w, dtype=torch.float32, device=self.device).unsqueeze(0)
        tcp_b = quat_apply_inverse(root_quat, tcp_w - root_pos)
        shoulder_w = self.robot.data.body_pos_w.torch[:, self.shoulder_body_id]
        shoulder_b = quat_apply_inverse(root_quat, shoulder_w - root_pos)
        # Down-pointing gripper: in the base frame that is a pure yaw about +z equal to the azimuth
        # of the target (gripper x = horizontal reach direction, gripper -z = straight down).
        rel = (tcp_b - shoulder_b)[0]
        azimuth = math.atan2(float(rel[1]), float(rel[0]))
        quat_des_b = torch.tensor(yaw_quat(azimuth), dtype=torch.float32, device=self.device).unsqueeze(0)
        body_pos_des_b = tcp_b - quat_apply(quat_des_b, self.tcp_offset)

        command = torch.cat([body_pos_des_b, quat_des_b], dim=-1)
        self.ik.set_command(command)

        ee_pose_w = self.robot.data.body_pose_w.torch[:, self.ee_body_id]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(root_pos, root_quat, ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        # the Jacobian is expressed in the world frame; rotate it into the base frame like the IK action does
        jacobian = self.robot.data.body_link_jacobian_w.torch[:, self.ee_jacobi_idx, :, self.jacobi_joint_ids].clone()
        jacobian[:, 0:3, :] = self._rotate_rows_to_base(root_quat, jacobian[:, 0:3, :])
        jacobian[:, 3:6, :] = self._rotate_rows_to_base(root_quat, jacobian[:, 3:6, :])
        joint_pos = self.robot.data.joint_pos.torch[:, self.arm_cfg.joint_ids]
        joint_pos_des = self.ik.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
        self.robot.set_joint_position_target_index(target=joint_pos_des, joint_ids=self.arm_cfg.joint_ids)
        self.robot.set_joint_position_target_index(target=self.gripper_target, joint_ids=self.gripper_cfg.joint_ids)

    @staticmethod
    def _rotate_rows_to_base(root_quat: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        """Rotate 3xN Jacobian rows from world into the base frame."""
        cols = rows.transpose(1, 2)  # (1, N, 3)
        quat = root_quat.unsqueeze(1).expand(-1, cols.shape[1], -1).reshape(-1, 4)
        out = quat_apply_inverse(quat, cols.reshape(-1, 3)).reshape(cols.shape)
        return out.transpose(1, 2)

    ##
    # 状態遷移
    ##

    def _enter(self, phase: Phase) -> None:
        self.phase = phase
        self.phase_time = 0.0

    def setup(self) -> None:
        """Let everything settle, record the starting state and aim the camera."""
        for _ in range(int(0.3 * PHYSICS_RATE)):
            joint_pos = self.robot.data.joint_pos.torch[:, self.arm_cfg.joint_ids]
            self.robot.set_joint_position_target_index(target=joint_pos, joint_ids=self.arm_cfg.joint_ids)
            self.robot.set_joint_position_target_index(
                target=self.gripper_target, joint_ids=self.gripper_cfg.joint_ids
            )
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(self.dt)
        self.command_tcp_w = self._tcp_world()
        self.shoulder_xy_w = self.robot.data.body_pos_w.torch[0, self.shoulder_body_id, :2].cpu().numpy().astype(float)
        self.initial_error = self.measure()

        print("[INFO] シーンの準備が完了しました")
        print("[INFO] ロボット: TheRobotStudio SO-101 (5-DOF) / 押し付けは閉じたグリッパの指先で行います")
        print(
            f"[INFO] パレット {PALLET_SIZE[0] * 1000:.0f} x {PALLET_SIZE[1] * 1000:.0f} x {PALLET_SIZE[2] * 1000:.0f} mm,"
            f" {args_cli.pallet_mass * 1000:.0f} g, 摩擦係数 {args_cli.friction:.2f}"
        )
        print(
            f"[INFO] 位置決め枠: 片側すき間 {FRAME_CLEARANCE * 1000:.1f} mm,"
            f" 目標精度 ±{POS_TOL * 1000:.1f} mm / ±{YAW_TOL_DEG:.1f} deg"
        )
        print(f"[PLC ] 嵌合ステーション: パレット位置を検査 ... {self.initial_error.summary()}")
        if not self.initial_error.in_frame:
            print("[PLC ] *** パレット位置ずれ検出: ライン停止 *** -> アームによる自動復旧を開始します")

        focus = np.array([FRAME_CENTER_XY[0], FRAME_CENTER_XY[1], TABLE_TOP])
        self.sim.set_camera_view(
            (float(focus[0]) + 0.42, float(focus[1]) - 0.42, float(focus[2]) + 0.30),
            (float(focus[0]) - 0.03, float(focus[1]), float(focus[2]) + 0.02),
        )

    def step_mission(self) -> None:
        """Advance the state machine by one control period."""
        self.phase_time += self.dt

        if self.phase == Phase.GO_HOME:
            if self._move_command(self.home_tcp_w, TRAVEL_SPEED) and self.phase_time > 1.0:
                self._enter(Phase.INSPECT)

        elif self.phase == Phase.INSPECT:
            if self.phase_time < INSPECT_S:
                return
            err = self.measure()
            print(f"[MEAS] 計測 #{self.push_count}: {err.summary()}")
            if self.within_tolerance(err):
                self.final_error = err
                self.success = err.in_frame
                self.lamp_state = LAMP_OK if err.in_frame else LAMP_STOP
                if err.in_frame:
                    print(f"[PLC ] 位置決め完了 ({self.push_count} 回押し) -> ライン再開")
                else:
                    print("[PLC ] 許容範囲内ですが枠外です。作業者を呼び出します。")
                self._enter(Phase.RETURN_HOME)
                return
            plan = self.plan_push(err)
            if plan is None:
                self.final_error = err
                print("[PLC ] アームの届く押し方が見つかりません。作業者を呼び出します。")
                self._enter(Phase.RETURN_HOME)
                return
            if self.push_count >= args_cli.max_pushes:
                self.final_error = err
                print(f"[PLC ] {args_cli.max_pushes} 回押しても収まりませんでした。作業者を呼び出します。")
                self._enter(Phase.RETURN_HOME)
                return
            self.plan = plan
            self.push_count += 1
            self.lamp_state = LAMP_RECOVER
            print(f"[PLAN] 押し #{self.push_count}: {plan.label}")
            print(
                f"       接触点 ({plan.contact_xy[0] * 1000:.0f}, {plan.contact_xy[1] * 1000:.0f}) mm,"
                f" 押し方向 ({plan.direction[0]:+.2f}, {plan.direction[1]:+.2f})"
            )
            self._enter(Phase.MOVE_ABOVE)

        elif self.phase == Phase.MOVE_ABOVE:
            start = self.plan.start_xy
            goal = np.array([start[0], start[1], TABLE_TOP + TRAVEL_HEIGHT])
            # lift first if we are low, then travel horizontally
            if self.command_tcp_w[2] < TABLE_TOP + TRAVEL_HEIGHT - 1e-4:
                lift = self.command_tcp_w.copy()
                lift[2] = TABLE_TOP + TRAVEL_HEIGHT
                self._move_command(lift, DESCEND_SPEED)
            elif self._move_command(goal, TRAVEL_SPEED):
                self._enter(Phase.DESCEND)

        elif self.phase == Phase.DESCEND:
            start = self.plan.start_xy
            goal = np.array([start[0], start[1], TABLE_TOP + PUSH_HEIGHT])
            if self._move_command(goal, DESCEND_SPEED) and self.phase_time > 0.3:
                self.push_travel = 0.0
                self.contact_travel = None
                self._push_start_yaw = self.measure().yaw
                self._push_start_xy, _ = self._pallet_pose_xy()
                self._enter(Phase.PUSH)

        elif self.phase == Phase.PUSH:
            err = self.measure()
            if self.contact_travel is None:
                # still closing in: contact = the pallet starts to move
                pallet_xy, _ = self._pallet_pose_xy()
                moved = float(np.linalg.norm(pallet_xy - self._push_start_xy))
                turned = abs(math.degrees(err.yaw - self._push_start_yaw))
                if moved > CONTACT_MOVE or turned > CONTACT_TURN_DEG:
                    self.contact_travel = self.push_travel
                    print(f"[PUSH] 接触 (指先前進 {self.push_travel * 1000:.1f} mm でパレットが動き始めた)")
                elif self.push_travel >= CONTACT_SEARCH:
                    print(f"[PUSH] 接触できませんでした (指先前進 {self.push_travel * 1000:.1f} mm)。計測し直します。")
                    self._enter(Phase.BACKOFF)
                    return
            if self.contact_travel is not None:
                followed = self.push_travel - self.contact_travel
                reached = self.push_should_stop(self.plan, err)
                if reached or followed >= self.plan.follow_travel:
                    reason = "目標到達" if reached else "最大押し量"
                    print(
                        f"[PUSH] 停止 ({reason}, 接触後の押し量 {followed * 1000:.1f} mm): {err.summary()}"
                    )
                    self._enter(Phase.BACKOFF)
                    return
            slow_from = STANDOFF - FINGER_TIP_HALF - SLOWDOWN_BEFORE_FACE
            in_air = self.contact_travel is None and self.push_travel < slow_from
            step = (APPROACH_SPEED if in_air else args_cli.push_speed) * self.dt
            self.command_tcp_w[:2] = self.command_tcp_w[:2] + self.plan.direction * step
            self.push_travel += step
            if int(self.phase_time / 0.5) != int((self.phase_time - self.dt) / 0.5):
                tcp = self._tcp_world()
                lag = float(np.linalg.norm(tcp - self.command_tcp_w)) * 1000.0
                print(
                    f"  t={self.phase_time:4.1f}s 押し中: 指先 ({tcp[0] * 1000:.0f}, {tcp[1] * 1000:.0f},"
                    f" {(tcp[2] - TABLE_TOP) * 1000:.0f}) mm 追従誤差 {lag:4.1f} mm | {err.summary()}"
                )

        elif self.phase == Phase.BACKOFF:
            if self.phase_time < self.dt * 1.5:
                self._backoff_goal = self.command_tcp_w.copy()
                self._backoff_goal[:2] -= self.plan.direction * BACKOFF
            if self._move_command(self._backoff_goal, args_cli.push_speed * 2.0):
                self._enter(Phase.RISE)

        elif self.phase == Phase.RISE:
            goal = self.command_tcp_w.copy()
            goal[2] = TABLE_TOP + TRAVEL_HEIGHT
            if self._move_command(goal, DESCEND_SPEED):
                self.plan = None
                self._enter(Phase.INSPECT)

        elif self.phase == Phase.RETURN_HOME:
            if self.command_tcp_w[2] < TABLE_TOP + TRAVEL_HEIGHT - 1e-4:
                lift = self.command_tcp_w.copy()
                lift[2] = TABLE_TOP + TRAVEL_HEIGHT
                self._move_command(lift, DESCEND_SPEED)
            elif self._move_command(self.home_tcp_w, TRAVEL_SPEED):
                self._enter(Phase.DONE)

        elif self.phase == Phase.DONE:
            pass

    def _draw_markers(self) -> None:
        self.lamp_marker.visualize(
            translations=np.array([LAMP_POS], dtype=np.float32), marker_indices=[self.lamp_state]
        )
        show_arrow = self.plan is not None and self.phase in (Phase.MOVE_ABOVE, Phase.DESCEND, Phase.PUSH)
        self.arrow_marker.set_visibility(show_arrow)
        if show_arrow:
            d3 = np.array([self.plan.direction[0], self.plan.direction[1], 0.0])
            tip = self.plan.contact_xy - self.plan.direction * 0.018
            pos = np.array([[tip[0], tip[1], TABLE_TOP + 0.009]], dtype=np.float32)
            self.arrow_marker.visualize(
                translations=pos, orientations=quat_from_z_axis(d3).astype(np.float32).reshape(1, 4)
            )

    def run(self) -> None:
        """Run the mission until it finishes (or the window is closed)."""
        self.setup()
        self.ik.reset()
        step = 0
        elapsed = 0.0
        done_time = 0.0
        while simulation_app.is_running():
            self.step_mission()
            self._apply_ik()
            render_this_step = self.render_enabled and step % RENDER_INTERVAL == 0
            if render_this_step:
                self._draw_markers()
            self.scene.write_data_to_sim()
            self.sim.step(render=render_this_step)
            self.scene.update(self.dt)
            step += 1
            elapsed += self.dt
            if self.phase == Phase.DONE:
                done_time += self.dt
                if done_time >= FINAL_HOLD_S:
                    break
        self.print_report(elapsed)

    def print_report(self, elapsed: float) -> None:
        """Print the end-of-demo summary."""
        final = self.final_error if self.final_error is not None else self.measure()
        print("")
        print("=" * 78)
        print(" パレット位置ずれ自動復旧 — 結果")
        print("=" * 78)
        if self.initial_error is not None:
            print(f" 初期状態 : {self.initial_error.summary()}")
        print(f" 最終状態 : {final.summary()}")
        print(f" 押し回数 : {self.push_count} 回 / 所要時間 {elapsed:.1f} s")
        print(f" 判定     : {'OK (ライン再開)' if self.success else 'NG (作業者呼び出し)'}")
        print("=" * 78)


def main() -> None:
    """Build the scene and run the pallet-nudging mission."""
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=1.0 / PHYSICS_RATE, render_interval=RENDER_INTERVAL)
    sim = SimulationContext(sim_cfg)

    scene_cfg = NudgeSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    demo = NudgeDemo(sim, scene)
    demo.run()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
