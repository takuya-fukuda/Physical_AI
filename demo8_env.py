# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""demo8 共通: SO-101 が「ずれたパレットを突いて枠内に戻す」強化学習環境。

``demo8_train.py`` (学習) と ``demo8.py`` (学習済み方策での実践) の両方から使う共有モジュール。
Isaac Sim を起動 (``AppLauncher``) した **後で** import すること。

demo7 との関係
--------------

demo7 は「計測 → 押し方を決める → 1 回押す → 計測」をルールベースで回していた。
demo8 はこの **「押し方を決める」部分を強化学習の方策に置き換える**。押す動作そのもの
(上空へ移動 → 降下 → 押す → 引いて上昇) は demo7 と同じ手続きで、方策は 1 回の押しごとに
そのパラメータだけを決める (パラメータ化された動作プリミティブ方式)。

* **RL の 1 ステップ = 1 回の押し**。物理は :data:`DECIMATION` サブステップ (= 3.5 s) 進む。
* 1 エピソードは最大 :data:`MAX_PUSHES` 回。枠内に収まれば成功で終了する。

観測 (方策が見るもの, 16 次元)
------------------------------

* 枠座標でのパレットの位置ずれ (x, y) と角度ずれ
* パレット自身の座標で見た位置ずれ (どの面を押せば効くかが分かりやすい)
* 前回の行動、前回の押しで **実際に** パレットが動いた量 (x, y, yaw)
* 前回の押しで指先が接触した位置 (予定より手前/奥) と、押し力のピーク [N]
  (パレットにかかる法線方向の接触力の水平成分。台からの反力は真上向きなので、これがそのまま
  指先の押し力になる。実機なら指先の力センサやサーボの負荷値に相当する)
* 残りの押し回数

方策には **重さ・摩擦は見せない**。前回の「押し込み量に対して実際どれだけ動いたか」と
「押し力」から、重い/滑りにくいパレットかどうかを推し量らせる (実機でも重量は普通分からない)。
価値関数 (critic) だけは重さ・摩擦を特権情報として見る (asymmetric actor-critic)。

行動 (5 次元, いずれも [-1, 1])
-------------------------------

* ``a0, a1``: パレット座標での方向ベクトル。中心からこの向きに伸ばした線が当たる側面を、
  その点で押す (2 次元にしているのは角度の折り返しで不連続にならないようにするため)
* ``a2``: 押す向きの傾き。側面の法線から ±:data:`MAX_TILT_DEG` (斜め押し)
* ``a3``: 押し込み量。指先が面に触れる位置から 0 〜 :data:`MAX_DEPTH` [m] 先まで押す
* ``a4``: 押す速さ :data:`MIN_PUSH_SPEED` 〜 :data:`MAX_PUSH_SPEED` [m/s]

``a3`` と ``a4`` がいわゆる「力加減」: 重いパレットはアームがたわむので深めに押す必要があり、
軽いパレットを速く押すと勢いで行き過ぎる。強く当てすぎると報酬で減点される。

報酬 (1 回の押しごと)
---------------------

* ずれ (正規化した距離) が **減った分** を加点 (進捗報酬)
* 成功 (枠の内側に :data:`DONE_MARGIN` 以上の余裕で収まる) で大きく加点して終了
* 1 回押すごとに小さく減点 (少ない回数で直すほど得)
* 押し力が「そのパレットを動かすのに必要な力」の 1.5 倍 + 1 N を超えたら減点 (乱暴な押し)
* アームの届かない押し方を選んだら減点 (押しは実行されない)
* パレットが傾いた/台から外れたら大きく減点して終了

ドメインランダム化
------------------

* 初期ずれ: 枠中心から 8 〜 28 mm を **全方向** (上下左右・斜め) に、角度 ±20 deg
* パレット質量: :data:`MASS_RANGE` [kg] をエピソードごとに
* パレットの摩擦係数: :data:`FRICTION_RANGE` を環境ごとに
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import warp as wp

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_apply, quat_apply_inverse, subtract_frame_transforms

from isaaclab_assets.robots.so101 import SO101_HIGH_PD_CFG

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from isaaclab_tasks.contrib.stack.config.so101.pose_ik_controller import (
    SO101PoseIKController,
    SO101PoseIKControllerCfg,
)

##
# シーンの寸法 [m] (各環境のローカル座標。demo7 と同じ配置)
##

TABLE_SIZE = (0.70, 0.60, 0.40)
"""Size (x, y, z) [m] of the work table."""
TABLE_CENTER_XY = (0.15, 0.0)
"""Table centre (x, y) [m] in env-local coordinates."""
TABLE_TOP = TABLE_SIZE[2]
"""Height [m] of the table surface."""
TABLE_FRICTION = 0.5
"""Friction coefficient of the table surface (PhysX averages it with the pallet's)."""
SO101_MOUNT_Z = -0.03008
"""Offset [m] from the table surface to the SO-101 root so its column foot sits on the table."""
SO101_BASE_ROT = (0.0, 0.0, 0.70710678, 0.70710678)
"""Root orientation (x, y, z, w): 90 deg about +z so the arm faces env-local +x."""

PALLET_SIZE = (0.08, 0.06, 0.03)
"""Pallet edge lengths (x, y, z) [m] in its own frame."""
FRAME_CENTER_XY = (0.22, 0.0)
"""Centre (x, y) [m] of the locating frame in env-local coordinates."""
FRAME_YAW_DEG = 90.0
"""Yaw [deg] of the locating frame (pallet long side runs across the front of the arm)."""
FRAME_CLEARANCE = 0.005
"""Gap [m] per side between the pallet and the inside edge of the frame tape."""
FRAME_LINE_WIDTH = 0.005
"""Width [m] of the frame tape."""

##
# SO-101
##

ARM_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
"""Actuated arm joints, in the order the IK controller sees them."""
ORIENTATION_JOINT_NAMES = ("wrist_flex", "wrist_roll")
"""Joints allowed to serve the orientation rows of the IK task."""
EE_BODY_NAME = "gripper"
"""Static wrist/gripper body the IK tracks."""
GRIPPER_JOINT_NAME = "gripper"
"""The jaw joint, held shut: the arm pushes, it never grasps."""
TCP_OFFSET = (0.0, 0.0, -0.092)
"""Tool point [m] in the ``gripper`` body frame, ~12 mm above the fingertips."""
SO101_INIT_JOINT_POS: dict[str, float] = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -0.6,
    "elbow_flex": 0.8,
    "wrist_flex": 0.6,
    "wrist_roll": 0.0,
    "gripper": 0.0,
}
"""Joint pose [rad] the arm is spawned in before it is driven to :data:`HOME_TCP`."""
ARM_STIFFNESS = 600.0
"""Position-loop stiffness [N·m/rad] of the arm joints."""
ARM_DAMPING = 60.0
"""Position-loop damping [N·m·s/rad] of the arm joints."""
ARM_EFFORT_LIMIT = 4.0
"""Joint torque limit [N·m]. Close to a real STS3215 servo, so heavy pallets make the arm yield."""
JOINT_LIMIT_MARGIN = 0.02
"""Margin [rad] kept between the IK joint targets and the joint limits."""

##
# 押し動作 (プリミティブ) のパラメータ
##

PHYSICS_DT = 0.01
"""Physics step [s]."""
PUSH_HEIGHT = 0.017
"""Height [m] of the tool point above the table while pushing (fingertips ~5 mm above the table)."""
TRAVEL_HEIGHT = 0.085
"""Height [m] of the tool point above the table between pushes."""
HOME_TCP = (0.19, 0.0, TABLE_TOP + 0.10)
"""Home tool point [m] (env-local) the arm waits at between episodes."""
FINGER_TIP_HALF = 0.010
"""Half-thickness [m] of the closed jaws at push height (measured in demo7)."""
STANDOFF = 0.035
"""Distance [m] from the pallet face to the tool point where the descent happens."""
CONTACT_TRAVEL = STANDOFF - FINGER_TIP_HALF
"""Nominal fingertip travel [m] from the standoff point to touching the face."""
APPROACH_SPEED = 0.05
"""Speed [m/s] for the air gap before contact."""
SLOWDOWN_BEFORE_FACE = 0.008
"""Distance [m] before the nominal face where the push switches to the policy's speed."""
BACKOFF = 0.012
"""Distance [m] the fingertip retreats along the push axis before rising."""
ROT_FACE_FRACTION = 0.85
"""Contact points are kept within this fraction of a face's half-length (no pushing right on a corner)."""
MAX_TILT_DEG = 25.0
"""Largest angle [deg] between the push direction and the face's inward normal."""
MAX_DEPTH = 0.03
"""Largest push depth [m] past the nominal contact."""
MIN_PUSH_SPEED = 0.008
"""Slowest push speed [m/s] the policy may pick."""
MAX_PUSH_SPEED = 0.040
"""Fastest push speed [m/s] the policy may pick."""
REACH_MIN = 0.11
"""Minimum horizontal distance [m] from the shoulder the tool point may be commanded to."""
REACH_MAX = 0.30
"""Maximum horizontal distance [m] from the shoulder the tool point may be commanded to."""
CONTACT_FORCE_THRESHOLD = 0.3
"""Push force [N] above which the fingertip counts as touching the pallet."""

# Sub-step timeline of one push [physics steps]
N_MOVE = 50
"""Steps to fly (min-jerk) from the previous position to above the standoff point."""
N_DESCEND = 40
"""Steps to descend to push height (long enough for the servo lag to settle before the push)."""
N_PUSH = 200
"""Steps available for the push itself (approach + policy-speed push, then hold)."""
N_BACKOFF = 20
"""Steps to retreat along the push axis."""
N_RISE = 25
"""Steps to rise back to travel height."""
N_SETTLE = 15
"""Steps to let the pallet settle before it is measured."""
DECIMATION = N_MOVE + N_DESCEND + N_PUSH + N_BACKOFF + N_RISE + N_SETTLE
"""Physics steps per RL step (one push)."""

##
# タスク
##

MAX_PUSHES = 8
"""Pushes per episode."""
POS_TOL = 0.0025
"""Position tolerance [m] (per pallet axis) for a "centred" pallet."""
YAW_TOL_DEG = 1.5
"""Yaw tolerance [deg] for a "centred" pallet."""
DONE_MARGIN = 0.0015
"""Success also once every pallet corner is at least this far [m] inside the frame tape."""
INIT_OFFSET_RANGE = (0.008, 0.028)
"""Range [m] of the initial pallet offset from the frame centre (any direction)."""
INIT_YAW_RANGE_DEG = 20.0
"""Initial pallet yaw offset is uniform in ± this [deg]."""
MASS_RANGE = (0.10, 1.20)
"""Pallet mass range [kg], resampled every episode."""
FRICTION_RANGE = (0.30, 0.80)
"""Pallet static friction range, sampled per environment at start-up."""
ERROR_SCALE_POS = 0.01
"""Position error [m] that counts as one unit of normalised error."""
ERROR_SCALE_YAW_DEG = 5.0
"""Yaw error [deg] that counts as one unit of normalised error."""
LOST_DISTANCE = 0.08
"""Pallet offset [m] beyond which the episode is aborted (pushed off the station)."""
TIPPED_COS = math.cos(math.radians(10.0))
"""Pallet counts as tipped when its up axis is more than 10 deg from vertical."""

OBS_DIM = 16
"""Size of the policy observation."""
PRIV_DIM = 2
"""Size of the privileged (critic-only) observation: mass, friction."""
ACT_DIM = 5
"""Size of the action."""


##
# シーン
##


def _yaw_quat_tuple(yaw: float) -> tuple[float, float, float, float]:
    return (0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw))


def _frame_tape_cfgs() -> dict[str, AssetBaseCfg]:
    """Four thin, visual-only boxes drawing the locating frame on the table."""
    inner_x = PALLET_SIZE[0] + 2.0 * FRAME_CLEARANCE
    inner_y = PALLET_SIZE[1] + 2.0 * FRAME_CLEARANCE
    w = FRAME_LINE_WIDTH
    yaw = math.radians(FRAME_YAW_DEG)
    c, s = math.cos(yaw), math.sin(yaw)
    pieces = {
        "frame_px": ((0.5 * inner_x + 0.5 * w, 0.0), (w, inner_y + 2.0 * w)),
        "frame_nx": ((-0.5 * inner_x - 0.5 * w, 0.0), (w, inner_y + 2.0 * w)),
        "frame_py": ((0.0, 0.5 * inner_y + 0.5 * w), (inner_x, w)),
        "frame_ny": ((0.0, -0.5 * inner_y - 0.5 * w), (inner_x, w)),
    }
    cfgs: dict[str, AssetBaseCfg] = {}
    for name, ((lx, ly), size_xy) in pieces.items():
        cx = FRAME_CENTER_XY[0] + c * lx - s * ly
        cy = FRAME_CENTER_XY[1] + s * lx + c * ly
        cfgs[name] = AssetBaseCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Frame/{name}",
            spawn=sim_utils.CuboidCfg(
                size=(size_xy[0], size_xy[1], 0.0008),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.80, 0.05), roughness=0.9),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(cx, cy, TABLE_TOP + 0.0004), rot=_yaw_quat_tuple(yaw)),
        )
    return cfgs


_FRAME_TAPES = _frame_tape_cfgs()


@configclass
class PalletNudgeSceneCfg(InteractiveSceneCfg):
    """Table, table-mounted SO-101, locating frame and pallet."""

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
                static_friction=TABLE_FRICTION, dynamic_friction=TABLE_FRICTION * 0.85, restitution=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.57, 0.60), roughness=0.8),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(TABLE_CENTER_XY[0], TABLE_CENTER_XY[1], 0.5 * TABLE_SIZE[2])),
    )

    frame_px = _FRAME_TAPES["frame_px"]
    frame_nx = _FRAME_TAPES["frame_nx"]
    frame_py = _FRAME_TAPES["frame_py"]
    frame_ny = _FRAME_TAPES["frame_ny"]

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
    robot.actuators["arm"].effort_limit_sim = ARM_EFFORT_LIMIT

    pallet = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Pallet",
        spawn=sim_utils.CuboidCfg(
            size=PALLET_SIZE,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=0.5),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.25),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.45, dynamic_friction=0.38),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.35, 0.75), roughness=0.6),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(FRAME_CENTER_XY[0], FRAME_CENTER_XY[1], TABLE_TOP + 0.5 * PALLET_SIZE[2] + 0.0005),
            rot=_yaw_quat_tuple(math.radians(FRAME_YAW_DEG)),
        ),
    )

    # Net *normal* contact force on the pallet. The table only pushes straight up and friction is
    # not part of it, so its horizontal component is exactly the push force of the fingertip.
    pallet_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Pallet")


@configclass
class PalletNudgeEventCfg:
    """Domain randomisation: pallet friction per environment, pallet mass per episode."""

    pallet_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("pallet"),
            "static_friction_range": FRICTION_RANGE,
            "dynamic_friction_range": (FRICTION_RANGE[0] * 0.85, FRICTION_RANGE[1] * 0.85),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )

    pallet_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("pallet"),
            "mass_distribution_params": MASS_RANGE,
            "operation": "abs",
        },
    )


@configclass
class PalletNudgeEnvCfg(DirectRLEnvCfg):
    """Pallet-nudging task: one RL step = one parameterised push."""

    decimation = DECIMATION
    # (n - 0.5) pushes so the ceil() in max_episode_length cannot round up to n + 1
    episode_length_s = (MAX_PUSHES - 0.5) * DECIMATION * PHYSICS_DT
    action_space = ACT_DIM
    observation_space = OBS_DIM
    state_space = 0

    viewer: ViewerCfg = ViewerCfg(eye=(0.64, -0.42, TABLE_TOP + 0.30), lookat=(0.19, 0.0, TABLE_TOP + 0.02))
    sim: SimulationCfg = SimulationCfg(dt=PHYSICS_DT, render_interval=2)
    scene: PalletNudgeSceneCfg = PalletNudgeSceneCfg(num_envs=2048, env_spacing=1.0, replicate_physics=True)
    events: PalletNudgeEventCfg = PalletNudgeEventCfg()

    # randomisation of the initial pallet pose
    init_offset_range: tuple[float, float] = INIT_OFFSET_RANGE
    init_yaw_range_deg: float = INIT_YAW_RANGE_DEG

    # reward weights
    rew_progress = 1.0
    rew_success = 10.0
    rew_per_push = -0.1
    rew_force = -0.3
    rew_unreachable = -0.3
    rew_failure = -5.0


##
# 環境
##


class PalletNudgeEnv(DirectRLEnv):
    """SO-101 pushes a misaligned pallet back into its locating frame, one parameterised push per step."""

    cfg: PalletNudgeEnvCfg

    def __init__(self, cfg: PalletNudgeEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        n, dev = self.num_envs, self.device

        self.robot: Articulation = self.scene["robot"]
        self.pallet: RigidObject = self.scene["pallet"]
        self.contact: ContactSensor = self.scene["pallet_contact"]

        # --- joints / bodies ------------------------------------------------------------------
        self.arm_cfg = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES, body_names=[EE_BODY_NAME])
        self.arm_cfg.resolve(self.scene)
        self.gripper_cfg = SceneEntityCfg("robot", joint_names=[GRIPPER_JOINT_NAME])
        self.gripper_cfg.resolve(self.scene)
        self.ee_body_id = self.arm_cfg.body_ids[0]
        self.shoulder_body_id = self.robot.body_names.index("shoulder")
        self.ee_jacobi_idx = self.ee_body_id - 1 if self.robot.is_fixed_base else self.ee_body_id
        self.jacobi_joint_ids = [j + self.robot.num_base_dofs for j in self.arm_cfg.joint_ids]

        # --- IK (same set-up as demo7) --------------------------------------------------------
        ik_cfg = SO101PoseIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="adaptive_dls",
            ik_params={"lambda_min": 0.05, "lambda_max": 0.2, "sigma_thresh": 0.02},
            orientation_weight=0.5,
            orientation_joint_names=ORIENTATION_JOINT_NAMES,
        )
        self.ik = SO101PoseIKController(ik_cfg, num_envs=n, device=dev)
        resolved = [self.robot.joint_names[i] for i in self.arm_cfg.joint_ids]
        self.ik.set_orientation_joint_mask(
            torch.tensor([1.0 if name in ORIENTATION_JOINT_NAMES else 0.0 for name in resolved], device=dev)
        )
        self.gripper_target = torch.zeros((n, len(self.gripper_cfg.joint_ids)), device=dev)
        self.tcp_offset = torch.tensor([TCP_OFFSET], device=dev).expand(n, 3)
        limits = self.robot.data.soft_joint_pos_limits.torch[:, self.arm_cfg.joint_ids]
        self.joint_lower = limits[..., 0] + JOINT_LIMIT_MARGIN
        self.joint_upper = limits[..., 1] - JOINT_LIMIT_MARGIN

        # --- task geometry (env-local) ---------------------------------------------------------
        self.frame_xy = torch.tensor(FRAME_CENTER_XY, device=dev)
        self.frame_yaw = math.radians(FRAME_YAW_DEG)
        self.half = torch.tensor(PALLET_SIZE[:2], device=dev) * 0.5
        self.home_tcp = torch.tensor(HOME_TCP, device=dev).expand(n, 3).clone()

        # --- per-env buffers --------------------------------------------------------------------
        self.cmd_tcp = self.home_tcp.clone()  # current commanded tool point (env-local)
        self.move_from = self.home_tcp.clone()  # command at the start of the current push
        self.plan_start = torch.zeros((n, 2), device=dev)  # standoff point (env-local xy)
        self.plan_dir = torch.zeros((n, 2), device=dev)  # unit push direction
        self.plan_depth = torch.zeros(n, device=dev)
        self.plan_speed = torch.full((n,), MIN_PUSH_SPEED, device=dev)
        self.plan_valid = torch.zeros(n, dtype=torch.bool, device=dev)
        self.actions = torch.zeros((n, ACT_DIM), device=dev)
        self.peak_force = torch.zeros(n, device=dev)
        self.contact_travel = torch.full((n,), -1.0, device=dev)
        self._substep = 0

        # pallet state as seen by the controller: x, y (env-local) and yaw [rad]
        self.pallet_pose = torch.zeros((n, 3), device=dev)
        self.prev_pallet_pose = torch.zeros((n, 3), device=dev)
        self.prev_error_norm = torch.zeros(n, device=dev)
        self.error_norm = torch.zeros(n, device=dev)
        self.margin = torch.zeros(n, device=dev)
        self.success = torch.zeros(n, dtype=torch.bool, device=dev)
        self.failed = torch.zeros(n, dtype=torch.bool, device=dev)
        self.last_disp = torch.zeros((n, 3), device=dev)  # last push's pallet motion in pallet frame
        self.pushes_done = torch.zeros(n, device=dev)

        # privileged: pallet friction (fixed after start-up randomisation) and mass (per episode)
        materials = wp.to_torch(self.pallet.root_view.get_material_properties())
        pallet_mu = materials[:, 0, 0].to(dev)
        self.friction_eff = 0.5 * (pallet_mu + TABLE_FRICTION)  # PhysX "average" combine mode

        # find a home joint configuration once (gripper pointing down above the station)
        self._settle_arm_at_home()

    ##
    # 初期化
    ##

    def _settle_arm_at_home(self) -> None:
        """Drive every arm to :data:`HOME_TCP` with the IK and remember that joint pose for resets.

        The articulation only receives its configured initial joint pose on the first reset, so it is
        written explicitly first. Starting the IK from the stretched zero pose would land it on the
        flipped-elbow solution with the wrist pinned at its joint limit.
        """
        init_pos = self.robot.data.default_joint_pos.torch.clone()
        self.robot.write_joint_position_to_sim_index(position=init_pos)
        self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(init_pos))
        self.robot.set_joint_position_target_index(target=init_pos)
        for _ in range(10):
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)
        start = self._tcp_local()
        n_move, n_hold = 150, 100
        for i in range(n_move + n_hold):
            tau = min((i + 1) / n_move, 1.0)
            self.cmd_tcp = start + (self.home_tcp - start) * (10 * tau**3 - 15 * tau**4 + 6 * tau**5)
            self._apply_ik()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)
        self.home_joint_pos = self.robot.data.joint_pos.torch.clone()
        err = torch.linalg.norm(self._tcp_local() - self.home_tcp, dim=-1).max().item()
        print(f"[INFO] ホーム姿勢を決定しました (指先誤差 最大 {err * 1000:.1f} mm)")

    ##
    # 座標・計測ヘルパ
    ##

    def _tcp_local(self) -> torch.Tensor:
        pose = self.robot.data.body_pose_w.torch[:, self.ee_body_id]
        tcp_w = pose[:, 0:3] + quat_apply(pose[:, 3:7], self.tcp_offset)
        return tcp_w - self.scene.env_origins

    def _read_pallet(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Pallet (x, y, yaw) in env-local coordinates, and the cosine of its tilt."""
        pos = self.pallet.data.root_pos_w.torch - self.scene.env_origins
        q = self.pallet.data.root_quat_w.torch  # x, y, z, w
        x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        up_z = 1.0 - 2.0 * (x * x + y * y)
        return torch.stack([pos[:, 0], pos[:, 1], yaw], dim=-1), up_z

    def _errors(self, pose: torch.Tensor) -> dict[str, torch.Tensor]:
        """Pallet error w.r.t. the frame: frame-axes offset, pallet-axes offset, folded yaw, normalised norm, margin."""
        d = pose[:, 0:2] - self.frame_xy
        cf, sf = math.cos(self.frame_yaw), math.sin(self.frame_yaw)
        err_frame = torch.stack([cf * d[:, 0] + sf * d[:, 1], -sf * d[:, 0] + cf * d[:, 1]], dim=-1)
        yaw_err = torch.remainder(pose[:, 2] - self.frame_yaw + 0.5 * math.pi, math.pi) - 0.5 * math.pi
        cp, sp = torch.cos(pose[:, 2]), torch.sin(pose[:, 2])
        err_pallet = torch.stack([cp * d[:, 0] + sp * d[:, 1], -sp * d[:, 0] + cp * d[:, 1]], dim=-1)
        norm = torch.sqrt(
            (err_frame / ERROR_SCALE_POS).square().sum(-1) + (torch.rad2deg(yaw_err) / ERROR_SCALE_YAW_DEG).square()
        )
        # corner margin against the inner edge of the tape (frame coordinates)
        cy, sy = torch.cos(yaw_err), torch.sin(yaw_err)
        hx, hy = self.half[0], self.half[1]
        corners = [(sx * hx, sy_ * hy) for sx in (1.0, -1.0) for sy_ in (1.0, -1.0)]
        ex = err_frame[:, 0:1] + torch.stack([cy * px - sy * py for px, py in corners], -1)
        ey = err_frame[:, 1:2] + torch.stack([sy * px + cy * py for px, py in corners], -1)
        margin = torch.minimum(
            (hx + FRAME_CLEARANCE - ex.abs()).min(-1).values, (hy + FRAME_CLEARANCE - ey.abs()).min(-1).values
        )
        return {"frame": err_frame, "pallet": err_pallet, "yaw": yaw_err, "norm": norm, "margin": margin}

    def _reachable(self, xy: torch.Tensor) -> torch.Tensor:
        shoulder = self.robot.data.body_pos_w.torch[:, self.shoulder_body_id, 0:2] - self.scene.env_origins[:, 0:2]
        reach = torch.linalg.norm(xy - shoulder, dim=-1)
        return (reach >= REACH_MIN) & (reach <= REACH_MAX)

    ##
    # 行動 -> 押し方
    ##

    def decode_actions(self, actions: torch.Tensor, pallet_pose: torch.Tensor) -> dict[str, torch.Tensor]:
        """Turn raw actions in [-1, 1] into a push plan (also used by demo8.py for display)."""
        a = actions.clamp(-1.0, 1.0)
        hx, hy = self.half[0], self.half[1]
        # ray from the pallet centre in pallet coordinates -> which face it hits
        phi = torch.atan2(a[:, 1], a[:, 0])
        c, s = torch.cos(phi), torch.sin(phi)
        t_x = hx / c.abs().clamp_min(1e-6)
        t_y = hy / s.abs().clamp_min(1e-6)
        on_x_face = t_x <= t_y
        t = torch.minimum(t_x, t_y)
        lim_x, lim_y = ROT_FACE_FRACTION * hx, ROT_FACE_FRACTION * hy
        px = torch.where(on_x_face, torch.sign(c) * hx, (t * c).clamp(-lim_x, lim_x))
        py = torch.where(on_x_face, (t * s).clamp(-lim_y, lim_y), torch.sign(s) * hy)
        nx = torch.where(on_x_face, torch.sign(c), torch.zeros_like(c))
        ny = torch.where(on_x_face, torch.zeros_like(s), torch.sign(s))
        # inward push direction, tilted
        tilt = a[:, 2] * math.radians(MAX_TILT_DEG)
        ct, st = torch.cos(tilt), torch.sin(tilt)
        dx_l, dy_l = -(ct * nx - st * ny), -(st * nx + ct * ny)
        # to env-local coordinates
        cp, sp = torch.cos(pallet_pose[:, 2]), torch.sin(pallet_pose[:, 2])
        contact = pallet_pose[:, 0:2] + torch.stack([cp * px - sp * py, sp * px + cp * py], dim=-1)
        direction = torch.stack([cp * dx_l - sp * dy_l, sp * dx_l + cp * dy_l], dim=-1)
        depth = 0.5 * (a[:, 3] + 1.0) * MAX_DEPTH
        speed = MIN_PUSH_SPEED + 0.5 * (a[:, 4] + 1.0) * (MAX_PUSH_SPEED - MIN_PUSH_SPEED)
        start = contact - direction * STANDOFF
        end = contact + direction * (depth - FINGER_TIP_HALF).unsqueeze(-1)
        valid = self._reachable(start) & self._reachable(end)
        return {
            "contact": contact,
            "direction": direction,
            "depth": depth,
            "speed": speed,
            "start": start,
            "valid": valid,
        }

    ##
    # DirectRLEnv の実装
    ##

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone().clamp(-1.0, 1.0)
        plan = self.decode_actions(self.actions, self.pallet_pose)
        self.plan_start = plan["start"]
        self.plan_dir = plan["direction"]
        self.plan_depth = plan["depth"]
        self.plan_speed = plan["speed"]
        self.plan_valid = plan["valid"]
        self.move_from = self.cmd_tcp.clone()
        self.prev_pallet_pose = self.pallet_pose.clone()
        self.peak_force.zero_()
        self.contact_travel.fill_(-1.0)
        self._substep = 0

    def _push_travel(self, t: float) -> torch.Tensor:
        """Fingertip travel [m] from the standoff point after ``t`` [s] of the push phase."""
        fast_len = CONTACT_TRAVEL - SLOWDOWN_BEFORE_FACE
        t_fast = fast_len / APPROACH_SPEED
        if t < t_fast:
            travel = torch.full_like(self.plan_speed, APPROACH_SPEED * t)
        else:
            travel = fast_len + self.plan_speed * (t - t_fast)
        return torch.minimum(travel, CONTACT_TRAVEL + self.plan_depth)

    def _apply_action(self) -> None:
        k = self._substep
        self._substep += 1
        dt = self.physics_dt
        travel_z = TABLE_TOP + TRAVEL_HEIGHT
        push_z = TABLE_TOP + PUSH_HEIGHT
        above = torch.cat([self.plan_start, torch.full_like(self.plan_start[:, :1], travel_z)], dim=-1)

        b_move = N_MOVE
        b_desc = b_move + N_DESCEND
        b_push = b_desc + N_PUSH
        b_back = b_push + N_BACKOFF
        b_rise = b_back + N_RISE
        if k < b_move:
            tau = (k + 1) / N_MOVE
            s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
            cmd = self.move_from + (above - self.move_from) * s
        elif k < b_desc:
            tau = (k + 1 - b_move) / N_DESCEND
            cmd = above.clone()
            cmd[:, 2] = travel_z + (push_z - travel_z) * tau
        elif k < b_push:
            travel = self._push_travel((k + 1 - b_desc) * dt)
            xy = self.plan_start + self.plan_dir * travel.unsqueeze(-1)
            cmd = torch.cat([xy, torch.full_like(xy[:, :1], push_z)], dim=-1)
        elif k < b_back:
            tau = (k + 1 - b_push) / N_BACKOFF
            travel = self._push_travel(N_PUSH * dt) - BACKOFF * tau
            xy = self.plan_start + self.plan_dir * travel.unsqueeze(-1)
            cmd = torch.cat([xy, torch.full_like(xy[:, :1], push_z)], dim=-1)
        elif k < b_rise:
            tau = (k + 1 - b_back) / N_RISE
            travel = self._push_travel(N_PUSH * dt) - BACKOFF
            xy = self.plan_start + self.plan_dir * travel.unsqueeze(-1)
            cmd = torch.cat([xy, torch.full_like(xy[:, :1], push_z + (travel_z - push_z) * tau)], dim=-1)
        else:
            cmd = self.cmd_tcp

        # unreachable plans: the arm just hovers where it was
        self.cmd_tcp = torch.where(self.plan_valid.unsqueeze(-1), cmd, self.move_from)

        # push force on the pallet (from the previous physics step)
        if b_desc <= k < b_back:
            force = self.measure_push_force()
            self.peak_force = torch.maximum(self.peak_force, force)
            if k < b_push:
                travel_now = self._push_travel((k + 1 - b_desc) * dt)
                first = (self.contact_travel < 0.0) & (force > CONTACT_FORCE_THRESHOLD)
                self.contact_travel = torch.where(first, travel_now, self.contact_travel)

        self._apply_ik()

    def measure_push_force(self) -> torch.Tensor:
        """Push force [N] of the fingertip on the pallet: horizontal part of the pallet's net normal force."""
        return torch.linalg.norm(self.contact.data.net_forces_w.torch[:, 0, 0:2], dim=-1)

    def _apply_ik(self) -> None:
        """Command the gripper body so the tool point reaches :attr:`cmd_tcp`, pointing straight down."""
        root_pose = self.robot.data.root_pose_w.torch
        root_pos, root_quat = root_pose[:, 0:3], root_pose[:, 3:7]
        tcp_b = quat_apply_inverse(root_quat, self.cmd_tcp + self.scene.env_origins - root_pos)
        shoulder_w = self.robot.data.body_pos_w.torch[:, self.shoulder_body_id]
        rel = tcp_b - quat_apply_inverse(root_quat, shoulder_w - root_pos)
        half_az = 0.5 * torch.atan2(rel[:, 1], rel[:, 0])
        zeros = torch.zeros_like(half_az)
        quat_des = torch.stack([zeros, zeros, torch.sin(half_az), torch.cos(half_az)], dim=-1)
        body_pos_des = tcp_b - quat_apply(quat_des, self.tcp_offset)
        self.ik.set_command(torch.cat([body_pos_des, quat_des], dim=-1))

        ee_pose_w = self.robot.data.body_pose_w.torch[:, self.ee_body_id]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(root_pos, root_quat, ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        jac_w = self.robot.data.body_link_jacobian_w.torch[:, self.ee_jacobi_idx, :, self.jacobi_joint_ids]
        jacobian = torch.cat(
            [self._rotate_rows_to_base(root_quat, jac_w[:, 0:3]), self._rotate_rows_to_base(root_quat, jac_w[:, 3:6])],
            dim=1,
        )
        joint_pos = self.robot.data.joint_pos.torch[:, self.arm_cfg.joint_ids]
        joint_pos_des = self.ik.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
        # Never command past a joint limit: a target behind the limit makes the servo push against
        # its end stop forever, which wastes torque and corrupts the torque-based force estimate.
        joint_pos_des = torch.clamp(joint_pos_des, self.joint_lower, self.joint_upper)
        self.robot.set_joint_position_target_index(target=joint_pos_des, joint_ids=self.arm_cfg.joint_ids)
        self.robot.set_joint_position_target_index(target=self.gripper_target, joint_ids=self.gripper_cfg.joint_ids)

    @staticmethod
    def _rotate_rows_to_base(root_quat: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        """Rotate 3xJ Jacobian rows from world into the base frame."""
        cols = rows.transpose(1, 2)  # (N, J, 3)
        quat = root_quat.unsqueeze(1).expand(-1, cols.shape[1], -1).reshape(-1, 4)
        return quat_apply_inverse(quat, cols.reshape(-1, 3)).reshape(cols.shape).transpose(1, 2)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        pose, up_z = self._read_pallet()
        self.pallet_pose = pose
        err = self._errors(pose)
        self.error_norm = err["norm"]
        self.margin = err["margin"]
        centred = (torch.rad2deg(err["yaw"]).abs() <= YAW_TOL_DEG) & (err["pallet"].abs() <= POS_TOL).all(-1)
        self.success = (centred | (err["margin"] >= DONE_MARGIN)) & (up_z > TIPPED_COS)
        lost = torch.linalg.norm(err["frame"], dim=-1) > LOST_DISTANCE
        fallen = (self.pallet.data.root_pos_w.torch[:, 2] - self.scene.env_origins[:, 2]) < TABLE_TOP
        self.failed = lost | fallen | (up_z <= TIPPED_COS)

        # how the pallet actually moved during this push, in its own (pre-push) frame
        d = pose - self.prev_pallet_pose
        cp, sp = torch.cos(self.prev_pallet_pose[:, 2]), torch.sin(self.prev_pallet_pose[:, 2])
        dyaw = torch.remainder(d[:, 2] + math.pi, 2.0 * math.pi) - math.pi
        self.last_disp = torch.stack([cp * d[:, 0] + sp * d[:, 1], -sp * d[:, 0] + cp * d[:, 1], dyaw], dim=-1)
        self.pushes_done += 1.0

        time_out = self.episode_length_buf >= self.max_episode_length
        return self.success | self.failed, time_out

    def _get_rewards(self) -> torch.Tensor:
        cfg = self.cfg
        progress = (self.prev_error_norm - self.error_norm).clamp(-5.0, 5.0)
        self.prev_error_norm = self.error_norm.clone()

        mass = self.pallet.data.body_mass.torch[:, 0]
        force_needed = self.friction_eff * mass * 9.81
        force_allowed = 1.5 * force_needed + 1.0
        excess = ((self.peak_force - force_allowed) / force_allowed).clamp(0.0, 3.0)

        reward = (
            cfg.rew_progress * progress
            + cfg.rew_per_push
            + cfg.rew_force * excess
            + cfg.rew_unreachable * (~self.plan_valid).float()
            + cfg.rew_success * self.success.float()
            + cfg.rew_failure * self.failed.float()
        )

        log = self.extras.setdefault("log", {})
        log["Push/peak_force_N"] = self.peak_force.mean().item()
        log["Push/excess_force"] = excess.mean().item()
        log["Push/unreachable_rate"] = (~self.plan_valid).float().mean().item()
        moved = (self.last_disp[:, 0:2].norm(dim=-1) > 0.0005) | (self.last_disp[:, 2].abs() > math.radians(0.3))
        log["Push/pallet_moved_rate"] = moved.float().mean().item()
        log["Push/missed_contact_rate"] = (self.plan_valid & (self.contact_travel < 0.0)).float().mean().item()
        idle = ~moved
        if idle.any():
            # force-estimate noise floor: peak "force" on pushes that never moved the pallet
            log["Push/peak_force_idle_N"] = self.peak_force[idle].mean().item()
        return reward

    def _get_observations(self) -> dict:
        err = self._errors(self.pallet_pose)
        contact_offset = torch.where(
            self.contact_travel >= 0.0,
            (self.contact_travel - CONTACT_TRAVEL) / 0.01,
            torch.full_like(self.contact_travel, -2.0),
        )
        obs = torch.cat(
            [
                err["frame"] / 0.03,
                (err["yaw"] / 0.3).unsqueeze(-1),
                err["pallet"] / 0.03,
                self.actions,
                self.last_disp[:, 0:2] / 0.02,
                (self.last_disp[:, 2] / 0.2).unsqueeze(-1),
                contact_offset.unsqueeze(-1),
                (self.peak_force / 5.0).clamp(max=5.0).unsqueeze(-1),
                (1.0 - self.episode_length_buf.float() / self.max_episode_length).unsqueeze(-1),
            ],
            dim=-1,
        )
        mass = self.pallet.data.body_mass.torch[:, 0]
        priv = torch.stack([mass / MASS_RANGE[1], self.friction_eff], dim=-1)
        return {"policy": obs, "critic": torch.cat([obs, priv], dim=-1)}

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        # episode statistics of the envs about to be reset (skip the very first reset)
        if self.common_step_counter > 0:
            log = self.extras.setdefault("log", {})
            mass = self.pallet.data.body_mass.torch[env_ids, 0]
            ok = self.success[env_ids].float()
            log["Metrics/success_rate"] = ok.mean().item()
            log["Metrics/failure_rate"] = self.failed[env_ids].float().mean().item()
            log["Metrics/pushes_per_episode"] = self.pushes_done[env_ids].mean().item()
            log["Metrics/final_error_norm"] = self.error_norm[env_ids].mean().item()
            light, heavy = mass < 0.45, mass > 0.85
            if light.any():
                log["Metrics/success_light"] = ok[light].mean().item()
            if heavy.any():
                log["Metrics/success_heavy"] = ok[heavy].mean().item()

        super()._reset_idx(env_ids)  # applies the "reset" events (pallet mass)
        n = len(env_ids)

        # --- arm back home --------------------------------------------------------------------
        joint_pos = self.home_joint_pos[env_ids].clone()
        self.robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(joint_pos), env_ids=env_ids)
        self.robot.set_joint_position_target_index(target=joint_pos, env_ids=env_ids)
        self.cmd_tcp[env_ids] = self.home_tcp[env_ids]

        # --- pallet: random offset in any direction + random yaw ---------------------------------
        r = torch.empty(n, device=self.device).uniform_(*self.cfg.init_offset_range)
        ang = torch.empty(n, device=self.device).uniform_(-math.pi, math.pi)
        yaw = self.frame_yaw + torch.deg2rad(
            torch.empty(n, device=self.device).uniform_(-self.cfg.init_yaw_range_deg, self.cfg.init_yaw_range_deg)
        )
        x = self.frame_xy[0] + r * torch.cos(ang)
        y = self.frame_xy[1] + r * torch.sin(ang)
        self.set_pallet_pose(env_ids, x, y, yaw)

        # --- bookkeeping -------------------------------------------------------------------------
        self.actions[env_ids] = 0.0
        self.last_disp[env_ids] = 0.0
        self.peak_force[env_ids] = 0.0
        self.contact_travel[env_ids] = -1.0
        self.pushes_done[env_ids] = 0.0
        self.success[env_ids] = False
        self.failed[env_ids] = False

    def set_pallet_pose(self, env_ids: torch.Tensor, x: torch.Tensor, y: torch.Tensor, yaw: torch.Tensor) -> None:
        """Teleport the pallets of ``env_ids`` to (x, y, yaw) in env-local coordinates, at rest."""
        n = len(env_ids)
        pose = torch.zeros((n, 7), device=self.device)
        pose[:, 0] = x
        pose[:, 1] = y
        pose[:, 2] = TABLE_TOP + 0.5 * PALLET_SIZE[2] + 0.0005
        pose[:, 0:3] += self.scene.env_origins[env_ids]
        pose[:, 5] = torch.sin(0.5 * yaw)
        pose[:, 6] = torch.cos(0.5 * yaw)
        self.pallet.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        self.pallet.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros((n, 6), device=self.device), env_ids=env_ids
        )
        # the controller's view of the pallet (the sim buffers refresh only after the next step)
        new_pose = torch.stack([x, y, yaw], dim=-1)
        self.pallet_pose[env_ids] = new_pose
        self.prev_pallet_pose[env_ids] = new_pose
        err = self._errors(new_pose)
        self.error_norm[env_ids] = err["norm"]
        self.prev_error_norm[env_ids] = err["norm"]
        self.margin[env_ids] = err["margin"]


##
# PPO 設定 (rsl_rl)
##


@configclass
class PalletNudgePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO for the pallet-nudging task. Short episodes (<= 8 pushes), so a short horizon and small nets."""

    num_steps_per_env = 8
    max_iterations = 150
    save_interval = 25
    experiment_name = "so101_pallet_nudge"
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    clip_actions = 1.0
    actor = RslRlMLPModelCfg(
        hidden_dims=[128, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.6),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[128, 128],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.95,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


__all__ = [
    "DECIMATION",
    "MAX_PUSHES",
    "PalletNudgeEnv",
    "PalletNudgeEnvCfg",
    "PalletNudgePPORunnerCfg",
]
