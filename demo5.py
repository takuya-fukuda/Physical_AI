# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab 3.0+: 倉庫AGV(無人搬送車)による荷物搬送のフィジカルAIデモ。

idealworks 社の AGV ``iw.hub`` (Isaac Sim の公式アセット) を使って、倉庫内のピッキング棚から
搬出棚までパレット状の荷物を運ぶサイクルを繰り返す。AGV は荷物の下に潜り込み、リフト(昇降天板)を
40 mm 上げて荷物をスタンドから持ち上げ、搬送先でリフトを下げて置いてくる — 実機の iw.hub と同じ
搬送方式をそのまま物理シミュレーションで再現している。

「フィジカルAI」として、以下の 知覚 → 判断 → 制御 のクローズドループを回している:

1. **知覚**: 車体前方に 2D LiDAR (レイキャスタ, 水平 200 deg / 2 deg 刻み) を取り付け、
   通路に置かれた障害物までの距離を毎ステップ計測する。
2. **判断 (経路)**: LiDAR の距離データを極座標ヒストグラムに変換し、車体幅ぶんの膨張処理をした上で
   「目標方向に最も近い、空いている方向」を選ぶ (VFH 系の回避)。前方クリアランスから速度上限も決める。
3. **判断 (積載)**: 荷台は金属で滑りやすい (摩擦係数 0.12) ため、加減速や旋回が強いと荷物が滑る。
   コントローラは荷台上の荷物の相対位置を監視し、**滑りを検知したらそのときのピーク加速度から
   摩擦係数を推定し直して**、許容加速度・速度・ヨーレート変化率をオンラインで締め直す。
   滑らない時間が続けば推定値は少しずつ戻る。実際に走らせると 1 サイクル目で
   0.30 → 0.15 → 0.12 → 0.07 と学習し、2 サイクル目は荷ズレ 0 回で運び切る。
   ``--no_adaptive`` を付けると同じ経路で荷物が何度も荷台から落ちる。
4. **制御**: 目標速度 (v, ω) を差動二輪の車輪角速度に変換し、リフトは位置指令をランプさせて与える。
   荷物は回転中心から 0.26 m 後ろの荷台に載っているので、角加速度 α による横方向の力
   (α × r) も効く。そのため速度だけでなくヨーレートの変化率も制限している。

搬送は状態機械 (ピック棚へ移動 → ドッキング → リフト上昇 → 搬出 → 搬送先へ移動 → ドッキング →
リフト下降 → 退出) で管理し、``--cycles`` 回ぶん繰り返したら走行距離・荷崩れ回数・学習された
安全速度などのサマリを表示して終了する。

.. code-block:: bash

    # ビューアありで 2 サイクル搬送する
    isaaclab.bat -p demo5.py --viz kit
    ./isaaclab.sh -p demo5.py --viz kit

    # 画面なしで実行 (ログだけ見る)
    isaaclab.bat -p demo5.py --viz none --cycles 1

    # 積載適応制御を切って、荷崩れしやすくなることを確認する
    isaaclab.bat -p demo5.py --viz kit --no_adaptive

    # 荷物を重く・荷台をもっと滑りやすくして、学習の効き方を変えてみる
    isaaclab.bat -p demo5.py --viz kit --payload_mass 40 --deck_friction 0.08
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Warehouse AGV transport demo with a lidar-guided payload-aware policy.")
parser.add_argument("--cycles", type=int, default=2, help="Number of pickup-to-dropoff transport cycles to run.")
parser.add_argument("--cruise_speed", type=float, default=1.4, help="Nominal cruise speed [m/s] of the AGV.")
parser.add_argument("--payload_mass", type=float, default=20.0, help="Mass [kg] of the transported box.")
parser.add_argument(
    "--deck_friction",
    type=float,
    default=0.12,
    help="Friction coefficient between the AGV deck and the box. Lower values make the payload slip sooner.",
)
parser.add_argument(
    "--friction_prior",
    type=float,
    default=0.30,
    help="Friction coefficient the controller assumes before it has observed any slip (optimistic prior).",
)
parser.add_argument(
    "--no_adaptive",
    action="store_true",
    help="Disable the payload-aware speed adaptation, so the AGV keeps driving at its nominal limits.",
)
parser.add_argument(
    "--max_steps", type=int, default=60000, help="Safety cap on the number of simulation steps before giving up."
)
parser.add_argument(
    "--visual_mode",
    type=str,
    default="proxy",
    choices=["proxy", "highres"],
    help=(
        "How the AGV is drawn. 'proxy' shows its collision shapes, 'highres' the meshes shipped with the USD."
        " The high-resolution meshes blank the RTX viewport on some setups (see use_collision_shape_visuals)."
    ),
)
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
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors.ray_caster import MultiMeshRayCaster, MultiMeshRayCasterCfg, patterns
from isaaclab.sim import SimulationContext
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

##
# AGV (idealworks iw.hub) の寸法。すべて USD の値から取っている。
##

# AGV の USD。Isaac Sim 6.0 のクラウドアセットに含まれる idealworks 製 AGV。
IW_HUB_USD_PATH = f"{ISAAC_NUCLEUS_DIR}/Robots/Idealworks/iwhub/iw_hub.usd"
"""USD path of the idealworks iw.hub AGV shipped with the Isaac Sim asset pack."""

WHEEL_RADIUS = 0.08
"""Radius [m] of the two driven wheels (from the wheel collision cylinder in the USD)."""
WHEEL_HALF_SEPARATION = 0.28963
"""Half of the distance [m] between the two driven wheels (from the wheel joint frames in the USD)."""
ROBOT_SPAWN_HEIGHT = 0.082
"""Height [m] the robot root is spawned at. The root frame sits at the wheel axle, i.e. one wheel radius."""
DECK_CENTER_X = -0.256
"""Longitudinal offset [m] of the lift deck center w.r.t. the robot root frame (the deck sits behind the axle)."""
DECK_TOP_Z = ROBOT_SPAWN_HEIGHT + 0.147
"""Height [m] of the lift deck surface above the ground while the lift joint is fully lowered."""
LIFT_STROKE = 0.04
"""Stroke [m] of the prismatic lift joint (0 = lowered, 0.04 = raised)."""
LIFT_SPEED = 0.03
"""Rate [m/s] the lift position target is ramped at."""
PHYSICS_RATE = 60.0
"""Physics steps per second [Hz].

120 Hz behaves the same but leaves no time budget for the renderer: a rendered frame costs ~23 ms
here, so the viewer ends up showing well under 30 frames per wall-clock second and the motion looks
choppy. At 60 Hz the mission plays back at roughly real time (verified: identical mission outcome).
"""
RENDER_INTERVAL = 2
"""Physics steps per rendered frame. At :data:`PHYSICS_RATE` = 60 Hz this renders the viewer at 30 Hz.

Raise it (4 -> 15 Hz) if the viewer still cannot keep up on your machine; lower it to 1 for a
smoother picture at the cost of the mission playing back slower than real time.
"""
CAMERA_DISTANCE = 6.0
"""Distance [m] the chase camera trails the AGV by."""
CAMERA_HEIGHT = 4.5
"""Height [m] of the chase camera above the ground."""
CAMERA_SMOOTHING = 0.08
"""Per-frame blend factor of the chase camera towards its desired pose (0 = frozen, 1 = rigid).

The camera is moved on every rendered frame, but it eases towards the target pose instead of being
snapped onto it: a rigidly attached camera copies every yaw wobble of the AGV, which reads as
juddering even when the frame rate is fine.
"""

##
# 倉庫レイアウト [m]。AGV は右回りのループを走る。
##

CARGO_SIZE = (1.0, 1.1, 0.4)
"""Size (length, width, height) [m] of the transported box."""
STAND_TOP_Z = 0.25
"""Height [m] of the station stands the box rests on. Slightly above the lowered deck (0.229 m),
so the AGV can drive underneath and take the load by raising the lift."""
STAND_OFFSET_Y = 0.45
"""Lateral offset [m] of each station stand from the station center. Wider than the AGV half width,
so the AGV can drive through the station."""

PICKUP_CENTER = (4.0, -3.5)
"""Center [m] of the pickup station. The AGV enters it driving towards +x."""
PICKUP_YAW = 0.0
"""Heading [rad] the AGV has while docking into the pickup station."""
DROPOFF_CENTER = (4.0, 3.5)
"""Center [m] of the dropoff station. The AGV enters it driving towards -x."""
DROPOFF_YAW = math.pi
"""Heading [rad] the AGV has while docking into the dropoff station."""

# 通路のウェイポイント。中央の棚をぐるりと回る右回りのループ。
ROUTE_PICKUP_TO_DROPOFF = [(8.6, -3.4), (9.3, -2.6), (9.3, 2.6), (8.0, 3.4)]
"""Coarse route [m] from the pickup station exit to the dropoff pre-dock point.

The straight leg between the two middle waypoints runs right through the pallet left in the right
aisle, so the reactive avoidance has to find its way around it. The waypoints themselves are kept
clear of the obstacles: a waypoint the AGV cannot legally stand on would deadlock the navigation.
"""
ROUTE_DROPOFF_TO_PICKUP = [(-0.6, 3.4), (-1.8, 2.6), (-1.8, -2.6), (0.0, -3.4)]
"""Coarse route [m] from the dropoff station exit back to the pickup pre-dock point."""

# 静的障害物 (LiDAR で見える): (中心x, 中心y, サイズx, サイズy, 高さ)
OBSTACLES = [
    (4.0, 0.0, 5.0, 1.2, 1.8),  # 中央の棚。これがあるのでループ状に迂回する
    (9.6, 0.0, 0.9, 0.9, 1.0),  # 右通路に置き去りにされたパレット
    (-1.8, 0.0, 0.8, 0.8, 1.0),  # 左通路に置き去りにされたパレット
    # 倉庫の外壁。通路幅を決めているので、AGV は置き去りパレットの脇をすり抜けるのではなく
    # 広いほうの通路へ回り込むことになる。
    (4.0, 5.6, 16.0, 0.3, 1.5),
    (4.0, -5.6, 16.0, 0.3, 1.5),
    (11.8, 0.0, 0.3, 11.5, 1.5),
    (-3.8, 0.0, 0.3, 11.5, 1.5),
]
"""Static obstacles the lidar can see, as (center x, center y, size x, size y, height) [m]."""

AGV_START_POSE = (0.0, -3.4, 0.0)
"""Initial (x [m], y [m], yaw [rad]) of the AGV."""

##
# 制御パラメータ
##

LIDAR_FOV = 100.0
"""Half of the horizontal field of view [deg] of the 2D lidar."""
LIDAR_RES = 2.0
"""Angular resolution [deg] of the 2D lidar."""
LIDAR_RANGE = 8.0
"""Maximum range [m] of the 2D lidar."""
AVOID_RADIUS = 0.85
"""Radius [m] the AGV footprint is inflated to during avoidance.

Sized for the *loaded* vehicle: the box is 1.1 m wide, so half of it (0.55 m) plus a safety margin,
not the 0.65 m wide chassis. With a smaller radius the AGV squeezes through gaps that fit the
chassis but scrape the payload off the deck.
"""
AVOID_LOOKAHEAD = 3.5
"""Distance [m] within which a lidar hit constrains the steering choice.

Obstacles further away than this are visible but not yet steered around; without this active window
a distant wall would block a wide arc of headings and the AGV would start swerving far too early.
"""
STEER_LIMIT = math.radians(60.0)
"""Largest steering deviation [rad] from the goal direction the avoidance is allowed to pick."""
YAW_GAIN = 2.2
"""Proportional gain [1/s] from heading error [rad] to yaw rate command [rad/s]."""
YAW_RATE_LIMIT = 1.1
"""Largest yaw rate [rad/s] commanded while driving."""
WAYPOINT_RADIUS = 0.7
"""Distance [m] at which a route waypoint counts as reached."""
DOCK_SPEED = 0.35
"""Speed [m/s] used for the final straight-in approach into a station."""
DOCK_TOLERANCE = 0.03
"""Longitudinal tolerance [m] for the docked position."""
STUCK_TIMEOUT = 10.0
"""Time [s] the AGV may stand still while navigating before it gives up on the current waypoint."""
DOCK_TIMEOUT = 40.0
"""Time [s] after which a docking attempt is aborted and retried from the pre-dock point."""
SLIP_WARN = 0.03
"""Additional payload displacement [m] on the deck that counts as a new slip event."""
SLIP_COOLDOWN = 1.5
"""Minimum time [s] between two slip events, so a jittering box is not reported over and over."""
SLIP_DROP = 0.35
"""Payload displacement [m] beyond which the box is considered fallen off."""
GRAVITY = 9.81
"""Gravitational acceleration [m/s^2]."""


class MissionState(enum.Enum):
    """States of the transport mission state machine."""

    GOTO_PICKUP = enum.auto()
    """Drive along the route towards the pickup station's pre-dock point."""
    DOCK_PICKUP = enum.auto()
    """Drive straight into the pickup station, underneath the box."""
    LIFT_UP = enum.auto()
    """Raise the lift so the box leaves the stands and rests on the deck."""
    EXIT_PICKUP = enum.auto()
    """Drive straight out of the pickup station with the box loaded."""
    GOTO_DROPOFF = enum.auto()
    """Drive along the route towards the dropoff station's pre-dock point."""
    DOCK_DROPOFF = enum.auto()
    """Drive straight into the dropoff station, above the stands."""
    LIFT_DOWN = enum.auto()
    """Lower the lift so the box is handed over to the stands."""
    EXIT_DROPOFF = enum.auto()
    """Drive straight out of the dropoff station without the box."""
    DONE = enum.auto()
    """All requested transport cycles are finished."""


class Station:
    """A drive-through station the AGV docks into to pick up or drop off a box.

    The station is defined by its center and the heading the AGV has while driving through it. The
    box rests on two stands placed left and right of the center, so the AGV can drive underneath it,
    take the load by raising its lift, and leave on the far side without reversing.
    """

    def __init__(self, name: str, center: tuple[float, float], yaw: float):
        """Initialize the station.

        Args:
            name: Human readable name, used in the log output.
            center: Center (x, y) [m] of the station, i.e. where the box sits.
            yaw: Heading [rad] the AGV drives through the station with.
        """
        self.name = name
        self.center = center
        self.yaw = yaw
        self.direction = (math.cos(yaw), math.sin(yaw))

    def point_at(self, along: float) -> tuple[float, float]:
        """Return the point [m] that is ``along`` meters down the station axis from the center."""
        return (self.center[0] + along * self.direction[0], self.center[1] + along * self.direction[1])

    @property
    def dock_point(self) -> tuple[float, float]:
        """Root position [m] the AGV must reach so that its deck is centered under the box."""
        return self.point_at(-DECK_CENTER_X)

    @property
    def pre_dock_point(self) -> tuple[float, float]:
        """Point [m] on the station axis the AGV lines up at before the straight-in approach."""
        return self.point_at(-3.2)

    @property
    def exit_point(self) -> tuple[float, float]:
        """Point [m] on the far side of the station the AGV leaves towards."""
        return self.point_at(3.0)

    def along(self, position: tuple[float, float]) -> float:
        """Return how far [m] a position sits down the station axis, measured from the center.

        The value is signed: it is negative while the AGV is still approaching and turns positive
        once it has driven past the center, which is what makes an overshoot detectable.
        """
        return (position[0] - self.center[0]) * self.direction[0] + (position[1] - self.center[1]) * self.direction[1]

    def lateral(self, position: tuple[float, float]) -> float:
        """Return the cross-track offset [m] of a position from the station axis (left is positive)."""
        return -(position[0] - self.center[0]) * self.direction[1] + (position[1] - self.center[1]) * self.direction[
            0
        ]


def wrap_to_pi(angle: float) -> float:
    """Wrap an angle [rad] into [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
    """Convert a yaw angle [rad] into a quaternion (x, y, z, w)."""
    return (0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw))


##
# シーン定義
##


@configclass
class WarehouseSceneCfg(InteractiveSceneCfg):
    """Scene with the AGV, the transported box, the ground and the front lidar.

    The static warehouse props (station stands and obstacles) are spawned separately by
    :func:`spawn_warehouse_props`, so they can be generated from plain Python lists.
    """

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.95))
    )

    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=IW_HUB_USD_PATH,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=16, solver_velocity_iteration_count=4
            ),
            # USD 側のドライブは "acceleration" 型で、ゲインが関節の実効慣性倍されるため、
            # ここで力(トルク)型に変えておく。そうしないと下の damping / effort_limit_sim が
            # N-m 単位として効かず、車輪がまったく差動しない (直進しかできない) 。
            joint_drive_props=sim_utils.schemas.JointDriveBaseCfg(drive_type="force"),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(AGV_START_POSE[0], AGV_START_POSE[1], ROBOT_SPAWN_HEIGHT),
            rot=yaw_to_quat(AGV_START_POSE[2]),
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
        actuators={
            # 差動二輪の駆動輪。速度制御なので剛性は 0、減衰だけを与える。
            "wheels": ImplicitActuatorCfg(
                joint_names_expr=["left_wheel_joint", "right_wheel_joint"],
                stiffness=0.0,
                damping=25.0,
                effort_limit_sim=30.0,
                velocity_limit_sim=40.0,
            ),
            # 昇降天板。位置制御。
            "lift": ImplicitActuatorCfg(
                joint_names_expr=["lift_joint"],
                # 過減衰にしておく。ここが振動すると荷台が上下に暴れ、荷物の垂直抗力が
                # 抜けて摩擦係数と無関係に滑り出してしまう。
                stiffness=1.5e5,
                damping=3.0e4,
                effort_limit_sim=3000.0,
                velocity_limit_sim=0.1,
            ),
            # 自由に回る従動キャスタ。ゲインは 0 にして完全に受動にする。
            "casters": ImplicitActuatorCfg(
                joint_names_expr=[".*_swivel_joint", ".*_caster_joint"],
                stiffness=0.0,
                damping=0.0,
            ),
        },
    )

    cargo = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cargo",
        spawn=sim_utils.CuboidCfg(
            size=CARGO_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=args_cli.payload_mass),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            # 荷台と荷物の間の摩擦。低いほど加減速・旋回で荷崩れしやすい。
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=args_cli.deck_friction,
                dynamic_friction=args_cli.deck_friction * 0.85,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.5, 0.12), roughness=0.85),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(PICKUP_CENTER[0], PICKUP_CENTER[1], STAND_TOP_Z + 0.5 * CARGO_SIZE[2]),
        ),
    )

    lidar = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/chassis",
        # 荷台より下・車体前端に取り付けるので、積んだ荷物で視界が塞がらない。
        offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.30, 0.0, 0.06)),
        mesh_prim_paths=[
            MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="/World/Warehouse/Obstacles", track_mesh_transforms=False)
        ],
        ray_alignment="base",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0.0, 0.0),
            horizontal_fov_range=(-LIDAR_FOV, LIDAR_FOV),
            horizontal_res=LIDAR_RES,
        ),
        max_distance=LIDAR_RANGE,
        update_period=0.0,
        debug_vis=not args_cli.headless,
    )


IW_HUB_HIGH_RES_VISUALS = (
    "Body",
    "Left_Wheel",
    "Right_Wheel",
    "Lift",
    "Left_Swivel",
    "Right_Swivel",
    "Inner_Left_Wheel",
    "Inner_Right_Wheel",
)
"""Names of the prims in ``iw_hub.usd`` that reference the high-resolution meshes under ``HighResProps/``."""


def use_collision_shape_visuals(robot_prim_path: str) -> None:
    """Draw the AGV with its collision shapes instead of its high-resolution meshes.

    ``iw_hub.usd`` references a set of high-resolution meshes (``HighResProps/*.usd``) that all share
    the same MDL material. On some setups, loading them puts the RTX renderer into a state where the
    viewport only shows the dome-light background: the scene renders correctly for the first frames
    and then goes uniformly white, while the physics keeps running normally. Hiding those meshes and
    showing the (simple, already authored) collision boxes and cylinders instead avoids the problem,
    costs nothing in physics fidelity, and renders far more cheaply.

    Pass ``--visual_mode highres`` to keep the original meshes if the renderer handles them.

    Args:
        robot_prim_path: Prim path of the spawned AGV.
    """
    from pxr import Usd, UsdGeom

    stage = sim_utils.get_current_stage()
    materials = {
        "chassis": sim_utils.PreviewSurfaceCfg(diffuse_color=(0.16, 0.18, 0.22), roughness=0.6, metallic=0.2),
        "deck": sim_utils.PreviewSurfaceCfg(diffuse_color=(0.75, 0.75, 0.78), roughness=0.35, metallic=0.6),
        "wheel": sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.05, 0.06), roughness=0.9),
    }
    for name, material_cfg in materials.items():
        material_cfg.func(f"/World/Materials/agv_{name}", material_cfg)

    # deactivate rather than hide: an inactive prim is pruned from composition, so Kit never loads the
    # referenced meshes and their material at all. Merely hiding them still streams them in, which is
    # enough to make the viewport flash white while they load.
    # collect paths first: deactivating a prim expires its descendants, and the referenced assets nest
    # prims of the same name (e.g. ".../chassis/Body/Body").
    to_deactivate = [
        prim.GetPath()
        for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_prim_path))
        if prim.GetName() in IW_HUB_HIGH_RES_VISUALS
    ]
    for path in to_deactivate:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            prim.SetActive(False)

    for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_prim_path)):
        name = prim.GetName()
        if name.startswith("Collision") or prim.GetTypeName() == "Cylinder":
            UsdGeom.Imageable(prim).MakeVisible()
            # the lift plate is the only collision box under the "lift" body
            is_deck = "/lift/" in str(prim.GetPath())
            kind = "wheel" if prim.GetTypeName() == "Cylinder" else ("deck" if is_deck else "chassis")
            sim_utils.bind_visual_material(str(prim.GetPath()), f"/World/Materials/agv_{kind}")


def spawn_warehouse_props() -> None:
    """Spawn the static warehouse geometry: the two station stands and the lidar obstacles.

    The stands are kept out of ``/World/Warehouse/Obstacles`` on purpose: a real AGV docks into a
    station it has in its map, so the reactive avoidance must not treat the station it is aiming for
    as something to swerve around.
    """
    sim_utils.create_prim("/World/Warehouse", prim_type="Xform")
    sim_utils.create_prim("/World/Warehouse/Stations", prim_type="Xform")
    sim_utils.create_prim("/World/Warehouse/Obstacles", prim_type="Xform")

    stand_cfg = sim_utils.CuboidCfg(
        size=(1.2, 0.12, STAND_TOP_Z),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.3, 0.38), roughness=0.7),
    )
    for name, center in (("pickup", PICKUP_CENTER), ("dropoff", DROPOFF_CENTER)):
        for side, sign in (("left", 1.0), ("right", -1.0)):
            stand_cfg.func(
                f"/World/Warehouse/Stations/{name}_{side}",
                stand_cfg,
                translation=(center[0], center[1] + sign * STAND_OFFSET_Y, 0.5 * STAND_TOP_Z),
            )

    for index, (pos_x, pos_y, size_x, size_y, height) in enumerate(OBSTACLES):
        obstacle_cfg = sim_utils.CuboidCfg(
            size=(size_x, size_y, height),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.45, 0.5), roughness=0.9),
        )
        obstacle_cfg.func(
            f"/World/Warehouse/Obstacles/obstacle_{index}", obstacle_cfg, translation=(pos_x, pos_y, 0.5 * height)
        )


##
# 知覚: LiDAR の距離データ
##


class LidarScan:
    """Polar scan of the 2D lidar, converted into ranges and body-frame bearings.

    Args:
        sensor: The ray-caster sensor mounted on the AGV.
    """

    def __init__(self, sensor: MultiMeshRayCaster):
        self.sensor = sensor
        num_rays = int(round(2.0 * LIDAR_FOV / LIDAR_RES)) + 1
        # LidarPatternCfg のレイ順と同じ並びで方位角を持っておく (0 deg が車体前方 +x)。
        self.bearings = torch.deg2rad(
            torch.linspace(-LIDAR_FOV, LIDAR_FOV, num_rays, device=sensor.device, dtype=torch.float32)
        )
        self.ranges = torch.full((num_rays,), LIDAR_RANGE, device=sensor.device, dtype=torch.float32)

    def update(self) -> None:
        """Refresh :attr:`ranges` from the latest ray-cast hits."""
        hits = self.sensor.data.ray_hits_w.torch[0]
        origin = self.sensor.data.pos_w.torch[0]
        distances = torch.linalg.norm(hits - origin, dim=-1)
        # ヒットしなかったレイは inf になるので最大距離に丸める
        self.ranges = torch.nan_to_num(distances, nan=LIDAR_RANGE, posinf=LIDAR_RANGE).clamp(max=LIDAR_RANGE)

    def nearest(self) -> tuple[float, float]:
        """Return the distance [m] and body-frame bearing [rad] of the closest measured hit."""
        index = int(torch.argmin(self.ranges).item())
        return float(self.ranges[index].item()), float(self.bearings[index].item())

    def forward_clearance(self, half_angle: float = math.radians(25.0)) -> float:
        """Return the smallest range [m] measured within ``half_angle`` of straight ahead."""
        mask = self.bearings.abs() < half_angle
        return float(self.ranges[mask].min().item())

    def pick_heading(self, goal_bearing: float) -> tuple[float, bool]:
        """Pick the drivable direction closest to the goal direction (a VFH-style search).

        Every measured hit blocks a wedge of directions whose half width grows as the hit gets
        closer, so that the AGV footprint (inflated to :data:`AVOID_RADIUS`) fits through whatever
        gap is chosen.

        Args:
            goal_bearing: Direction [rad] towards the current goal, in the body frame.

        Returns:
            A tuple of the chosen bearing [rad] in the body frame, and whether it is drivable. When
            nothing within the steering limit is free, the most open direction of the whole scan is
            returned with ``False``, so the caller can turn towards it on the spot.
        """
        # 候補方向は目標方位を中心に、操舵制限のなかから選ぶ
        candidates = self.bearings[self.bearings.abs() < STEER_LIMIT + math.radians(5.0)]
        # 各ヒットが塞ぐ方位の半幅。近いほど広く塞ぐ。
        safe_range = self.ranges.clamp(min=AVOID_RADIUS + 1e-3)
        blocked_half_width = torch.asin((AVOID_RADIUS / safe_range).clamp(max=1.0))
        # 候補 x レイ の行列で「塞がれているか」を判定する
        angular_gap = (candidates.unsqueeze(1) - self.bearings.unsqueeze(0)).abs()
        # 候補方向に進んだときに衝突しうるのは、障害物がその方向の手前にある場合だけ
        blocking = (angular_gap < blocked_half_width.unsqueeze(0)) & (self.ranges.unsqueeze(0) < AVOID_LOOKAHEAD)
        free = ~blocking.any(dim=1)
        if not bool(free.any().item()):
            # 行ける方向がない: スキャン全体で一番奥まで抜けている方向へ向き直る
            return float(self.bearings[int(torch.argmax(self.ranges).item())].item()), False
        free_candidates = candidates[free]
        # 目標方位に最も近い空き方向を選ぶ
        best = int(torch.argmin((free_candidates - goal_bearing).abs()).item())
        return float(free_candidates[best].item()), True


##
# 判断: 積載状態に適応する速度リミッタ (フィジカルAI の中核)
##


class PayloadAwareLimiter:
    """Speed limiter that learns how fast the AGV may drive without the box sliding off.

    The limiter keeps an estimate of the friction coefficient between the deck and the box. From it
    it derives the largest lateral acceleration (``mu * g``, with a safety factor) the payload can
    take, which in turn caps the speed in a turn (``a = v * omega``) and the longitudinal
    acceleration. Whenever the payload is observed to slide on the deck, the estimate is corrected
    downwards; if nothing slides for a while, it creeps back up towards the prior.

    Args:
        nominal_speed: Cruise speed [m/s] requested when nothing limits the AGV.
        friction_prior: Friction coefficient assumed before any slip has been observed.
        adaptive: Whether the estimate reacts to observed slip at all. When False, the limiter keeps
            the prior forever, which is the behaviour of a controller that ignores its payload.
    """

    def __init__(self, nominal_speed: float, friction_prior: float, adaptive: bool = True):
        self.nominal_speed = nominal_speed
        self.friction_prior = friction_prior
        self.adaptive = adaptive
        self.friction_estimate = friction_prior
        # 推定値どおりに攻めると滑り続けるので、推定した限界の半分だけ使う
        self.safety_factor = 0.5
        self.slip_events = 0
        self.max_payload_accel = 0.0
        self.slip_accel = 0.0
        """Peak acceleration [m/s^2] the payload saw just before the last recorded slip event."""
        self._slip_reference: tuple[float, float] | None = None
        self._reported_slip = 0.0
        self._time_since_slip = 0.0
        self._peak_accel = 0.0

    @property
    def accel_limit(self) -> float:
        """Largest horizontal acceleration [m/s^2] the payload is expected to tolerate."""
        return self.safety_factor * self.friction_estimate * GRAVITY

    def start_carrying(self, payload_local: tuple[float, float]) -> None:
        """Latch the payload's pose on the deck as the reference the slip is measured against.

        Args:
            payload_local: Payload position (x, y) [m] on the deck, in the AGV body frame.
        """
        self._slip_reference = payload_local
        self._reported_slip = 0.0
        self._time_since_slip = 0.0
        self._peak_accel = 0.0

    def stop_carrying(self) -> None:
        """Forget the payload reference, e.g. after the box has been handed to a station."""
        self._slip_reference = None
        self._reported_slip = 0.0

    def slip(self, payload_local: tuple[float, float]) -> float:
        """Return how far [m] the payload has moved on the deck since it was picked up."""
        if self._slip_reference is None:
            return 0.0
        return math.hypot(payload_local[0] - self._slip_reference[0], payload_local[1] - self._slip_reference[1])

    def observe(self, payload_local: tuple[float, float], payload_accel: float, dt: float) -> float | None:
        """Update the friction estimate from the payload's behaviour over the last step.

        A new slip event is only recorded when the payload has moved :data:`SLIP_WARN` *further* than
        at the previous event, and at least :data:`SLIP_COOLDOWN` seconds have passed since then. The
        reference pose is never reset, so a box that merely jitters on the deck does not keep
        retriggering; only a payload that genuinely keeps creeping does.

        Args:
            payload_local: Payload position (x, y) [m] on the deck, in the AGV body frame.
            payload_accel: Magnitude [m/s^2] of the horizontal acceleration the AGV is currently
                subjecting the payload to (longitudinal and lateral combined).
            dt: Time step [s].

        Returns:
            The payload displacement [m] when this step recorded a new slip event, otherwise None.
        """
        self.max_payload_accel = max(self.max_payload_accel, payload_accel)
        if self._slip_reference is None:
            return None
        # ずれは加速した「あと」に検知されるので、その区間のピーク加速度を憶えておき、
        # 滑り出した瞬間の加速度としてそれを使う。検知時点の加速度を使うと、
        # 等速走行中に前回の加速の結果が届いて「加速度 0 で滑った」ことになってしまう。
        self._peak_accel = max(self._peak_accel, payload_accel)
        self._time_since_slip += dt
        slip = self.slip(payload_local)
        if slip < self._reported_slip + SLIP_WARN or self._time_since_slip < SLIP_COOLDOWN:
            # しばらく滑っていないなら、推定摩擦係数を少しずつ元に戻して速度を取り戻す
            if self.adaptive and self._time_since_slip > 8.0:
                self.friction_estimate = min(self.friction_prior, self.friction_estimate + 0.001 * dt)
            return None
        # 滑った: 直前のピーク加速度が実際の限界だったとみなして摩擦の推定値を下げる
        self.slip_events += 1
        self._reported_slip = slip
        self._time_since_slip = 0.0
        self.slip_accel = self._peak_accel
        if self.adaptive:
            observed_friction = max(self._peak_accel, 0.15) / GRAVITY
            self.friction_estimate = max(0.05, min(self.friction_estimate * 0.8, observed_friction))
        self._peak_accel = 0.0
        return slip

    def limit_yaw_rate(self, yaw_rate_request: float, current_yaw_rate: float, dt: float) -> float:
        """Rate-limit the yaw command so the angular acceleration does not throw the payload sideways.

        The box sits :data:`DECK_CENTER_X` behind the turn center, so a yaw acceleration ``alpha``
        pushes it sideways with ``alpha * |DECK_CENTER_X|``. Snapping the steering from one extreme
        to the other is what slides the payload most, even at modest speeds.

        Args:
            yaw_rate_request: Yaw rate [rad/s] the navigation asks for.
            current_yaw_rate: Yaw rate [rad/s] currently commanded.
            dt: Time step [s].

        Returns:
            The admissible yaw rate [rad/s].
        """
        if self._slip_reference is None:
            yaw_accel_limit = 6.0
        else:
            yaw_accel_limit = self.accel_limit / abs(DECK_CENTER_X)
        return float(
            min(max(yaw_rate_request, current_yaw_rate - yaw_accel_limit * dt), current_yaw_rate + yaw_accel_limit * dt)
        )

    def limit(self, speed_request: float, yaw_rate: float, current_speed: float, dt: float) -> float:
        """Cap a speed request so that neither the turn nor the acceleration slides the payload.

        Args:
            speed_request: Speed [m/s] the navigation would like to drive at.
            yaw_rate: Yaw rate [rad/s] about to be commanded.
            current_speed: Speed [m/s] the AGV is currently commanding.
            dt: Time step [s].

        Returns:
            The admissible speed [m/s].
        """
        # 後退指令 (ドッキングの微調整) もあるので、大きさだけを制限して符号は保つ
        magnitude = min(abs(speed_request), self.nominal_speed)
        if self._slip_reference is not None:
            # 旋回中の横加速度 a = v * omega が限界を超えないように速度を抑える
            if abs(yaw_rate) > 1e-3:
                magnitude = min(magnitude, self.accel_limit / abs(yaw_rate))
            # 前後加速度も同じ限界でレートリミットする (急ブレーキでも荷物は前に滑る)
            accel_limit = self.accel_limit
        else:
            accel_limit = 2.5
        target = math.copysign(magnitude, speed_request)
        return float(min(max(target, current_speed - accel_limit * dt), current_speed + accel_limit * dt))


##
# 制御: 差動二輪の車輪速度
##


def wheel_targets(linear_speed: float, yaw_rate: float, device: torch.device | str) -> torch.Tensor:
    """Convert a body twist into the two wheel velocity targets.

    Args:
        linear_speed: Forward speed [m/s] of the AGV.
        yaw_rate: Yaw rate [rad/s] of the AGV, positive counter-clockwise.
        device: Device the target tensor is created on.

    Returns:
        Wheel angular velocity targets [rad/s] for (left, right), shape (1, 2).
    """
    left = (linear_speed - yaw_rate * WHEEL_HALF_SEPARATION) / WHEEL_RADIUS
    right = (linear_speed + yaw_rate * WHEEL_HALF_SEPARATION) / WHEEL_RADIUS
    return torch.tensor([[left, right]], device=device, dtype=torch.float32)


class TransportMission:
    """Sequences the pickup/dropoff cycles and produces the driving commands for each state.

    Args:
        pickup: The station the box is fetched from.
        dropoff: The station the box is delivered to.
        cycles: Number of transport cycles to run before finishing.
    """

    def __init__(self, pickup: Station, dropoff: Station, cycles: int):
        self.pickup = pickup
        self.dropoff = dropoff
        self.cycles = cycles
        self.state = MissionState.GOTO_PICKUP
        self.cycle = 0
        self.route: list[tuple[float, float]] = list(ROUTE_DROPOFF_TO_PICKUP[-1:]) + [pickup.pre_dock_point]
        self.waypoint = 0
        self.state_time = 0.0
        self.dock_retries = 0
        self.stuck_recoveries = 0
        self.stuck_time = 0.0
        self.carrying = False
        self.lift_target = 0.0

    @property
    def active_station(self) -> Station:
        """The station the current state revolves around."""
        if self.state in (MissionState.GOTO_PICKUP, MissionState.DOCK_PICKUP, MissionState.EXIT_PICKUP):
            return self.pickup
        return self.dropoff

    def _enter(self, state: MissionState, route: list[tuple[float, float]] | None = None) -> None:
        """Switch to ``state``, optionally installing a new route to follow."""
        self.state = state
        self.state_time = 0.0
        if route is not None:
            self.route = route
            self.waypoint = 0

    def goal(self) -> tuple[float, float]:
        """Return the point [m] the AGV is currently driving towards."""
        if self.state in (MissionState.GOTO_PICKUP, MissionState.GOTO_DROPOFF):
            return self.route[min(self.waypoint, len(self.route) - 1)]
        if self.state == MissionState.DOCK_PICKUP:
            return self.pickup.dock_point
        if self.state == MissionState.DOCK_DROPOFF:
            return self.dropoff.dock_point
        if self.state == MissionState.EXIT_PICKUP:
            return self.pickup.exit_point
        if self.state == MissionState.EXIT_DROPOFF:
            return self.dropoff.exit_point
        return self.active_station.dock_point

    def dock_error(self, position: tuple[float, float]) -> float:
        """Return the signed distance [m] left to the dock point, measured along the station axis.

        Positive means the AGV still has to drive forward, negative means it overshot the station.
        """
        return -DECK_CENTER_X - self.active_station.along(position)

    def advance(self, position: tuple[float, float], speed: float, dt: float) -> str | None:
        """Advance the state machine.

        Args:
            position: Current AGV root position (x, y) [m].
            speed: Current forward speed [m/s] of the AGV.
            dt: Time step [s].

        Returns:
            A log message when the state changed, otherwise None.
        """
        self.state_time += dt
        goal = self.goal()
        distance = math.hypot(goal[0] - position[0], goal[1] - position[1])

        if self.state in (MissionState.GOTO_PICKUP, MissionState.GOTO_DROPOFF):
            # 回避が袋小路に入って止まってしまったときの保険。しばらく動けていなければ、
            # そのウェイポイントは諦めて次へ進む (最後のウェイポイントならドッキングへ移る)。
            self.stuck_time = self.stuck_time + dt if abs(speed) < 0.08 else 0.0
            if self.stuck_time > STUCK_TIMEOUT:
                self.stuck_time = 0.0
                self.stuck_recoveries += 1
                if self.waypoint < len(self.route) - 1:
                    self.waypoint += 1
                    return "[WARN] 動けなくなったので、次のウェイポイントへ目標を切り替えます"
            if distance < WAYPOINT_RADIUS and self.waypoint < len(self.route) - 1:
                self.waypoint += 1
                return None
            if self.waypoint == len(self.route) - 1 and distance < 0.6:
                if self.state == MissionState.GOTO_PICKUP:
                    self._enter(MissionState.DOCK_PICKUP)
                    return f"[MISSION] {self.pickup.name} に進入します (荷物の下に潜り込む)"
                self._enter(MissionState.DOCK_DROPOFF)
                return f"[MISSION] {self.dropoff.name} に進入します (荷物を降ろす)"
            return None

        if self.state in (MissionState.DOCK_PICKUP, MissionState.DOCK_DROPOFF):
            # 駅の軸に沿った符号付き残距離で判定する。行き過ぎたら負になるので、乗り越しても検知できる。
            remaining = abs(self.dock_error(position))
            if remaining < DOCK_TOLERANCE + 0.03 and abs(speed) < 0.05:
                if self.state == MissionState.DOCK_PICKUP:
                    self.lift_target = LIFT_STROKE
                    self._enter(MissionState.LIFT_UP)
                    return "[MISSION] ドッキング完了。リフトを上げて荷物を持ち上げます"
                self.lift_target = 0.0
                self._enter(MissionState.LIFT_DOWN)
                return "[MISSION] ドッキング完了。リフトを下げて荷物を渡します"
            if self.state_time > DOCK_TIMEOUT:
                self.dock_retries += 1
                station = self.active_station
                if self.state == MissionState.DOCK_PICKUP:
                    self._enter(MissionState.GOTO_PICKUP, [station.pre_dock_point])
                else:
                    self._enter(MissionState.GOTO_DROPOFF, [station.pre_dock_point])
                return "[WARN] ドッキングに時間がかかりすぎました。手前からやり直します"
            return None

        if self.state == MissionState.LIFT_UP:
            if self.state_time > 2.5:
                self._enter(MissionState.EXIT_PICKUP)
                return "[MISSION] 積載完了。搬送先へ向かいます"
            return None

        if self.state == MissionState.LIFT_DOWN:
            if self.state_time > 2.5:
                self._enter(MissionState.EXIT_DROPOFF)
                self.cycle += 1
                return f"[MISSION] 荷降ろし完了 ({self.cycle}/{self.cycles} サイクル)"
            return None

        if self.state == MissionState.EXIT_PICKUP:
            if self.pickup.along(position) > 2.6:
                self._enter(MissionState.GOTO_DROPOFF, list(ROUTE_PICKUP_TO_DROPOFF) + [self.dropoff.pre_dock_point])
                return None
            return None

        if self.state == MissionState.EXIT_DROPOFF:
            if self.dropoff.along(position) > 2.6:
                if self.cycle >= self.cycles:
                    self._enter(MissionState.DONE)
                    return "[MISSION] 全サイクル終了"
                self._enter(MissionState.GOTO_PICKUP, list(ROUTE_DROPOFF_TO_PICKUP) + [self.pickup.pre_dock_point])
                return "[MISSION] 次の荷物を取りに戻ります"
            return None

        return None

    def drive_command(self, position: tuple[float, float], yaw: float, scan: LidarScan) -> tuple[float, float, bool]:
        """Compute the desired body twist for the current state.

        Args:
            position: Current AGV root position (x, y) [m].
            yaw: Current heading [rad] of the AGV.
            scan: The latest lidar scan.

        Returns:
            A tuple of desired speed [m/s], desired yaw rate [rad/s], and whether the avoidance is
            currently steering around something.
        """
        if self.state in (MissionState.LIFT_UP, MissionState.LIFT_DOWN, MissionState.DONE):
            return 0.0, 0.0, False

        goal = self.goal()
        goal_bearing = wrap_to_pi(math.atan2(goal[1] - position[1], goal[0] - position[0]) - yaw)
        distance = math.hypot(goal[0] - position[0], goal[1] - position[1])

        if self.state in (
            MissionState.DOCK_PICKUP,
            MissionState.DOCK_DROPOFF,
            MissionState.EXIT_PICKUP,
            MissionState.EXIT_DROPOFF,
        ):
            # ステーション出入りは駅の軸に沿った直線走行。横ずれは姿勢で詰める。
            station = self.active_station
            lateral_error = station.lateral(position)
            heading_error = wrap_to_pi(station.yaw - yaw - 1.2 * max(-0.6, min(0.6, lateral_error)))
            yaw_rate = max(-0.6, min(0.6, YAW_GAIN * heading_error))
            if self.state in (MissionState.DOCK_PICKUP, MissionState.DOCK_DROPOFF):
                # 残距離は符号付き。乗り越したら負になり、そのまま微速で後退して戻る。
                remaining = self.dock_error(position)
                speed = max(-0.15, min(DOCK_SPEED, 0.8 * remaining))
                if abs(remaining) < DOCK_TOLERANCE:
                    speed = 0.0
            else:
                speed = 0.8
            return speed, yaw_rate, False

        # 通常走行: LiDAR で空いている方向を選び、そちらへ向ける
        steer_bearing, is_free = scan.pick_heading(goal_bearing)
        avoiding = is_free and abs(steer_bearing - goal_bearing) > math.radians(6.0)
        if not is_free:
            # 全方向塞がっている: その場で一番開けている方向へ旋回して抜け道を探す
            return 0.0, max(-0.6, min(0.6, YAW_GAIN * steer_bearing)), True

        yaw_rate = max(-YAW_RATE_LIMIT, min(YAW_RATE_LIMIT, YAW_GAIN * steer_bearing))
        # 目標方向を向いているほど速く、前方が詰まっているほど遅く走る
        speed = max(0.0, math.cos(steer_bearing)) * args_cli.cruise_speed
        speed = min(speed, 0.6 * max(0.0, scan.forward_clearance() - 0.8))
        speed = min(speed, max(0.35, 1.2 * distance))
        return speed, yaw_rate, avoiding


def body_frame_offset(
    point: tuple[float, float], origin: tuple[float, float], yaw: float
) -> tuple[float, float]:
    """Express a world-frame point [m] in the body frame of a pose at ``origin`` with heading ``yaw``."""
    delta_x = point[0] - origin[0]
    delta_y = point[1] - origin[1]
    return (
        math.cos(yaw) * delta_x + math.sin(yaw) * delta_y,
        -math.sin(yaw) * delta_x + math.cos(yaw) * delta_y,
    )


def place_cargo_on_station(cargo: RigidObject, station: Station) -> None:
    """Teleport the box onto a station's stands and zero its velocity.

    Args:
        cargo: The transported box.
        station: The station to place the box on.
    """
    device = cargo.device
    pose = torch.tensor(
        [[station.center[0], station.center[1], STAND_TOP_Z + 0.5 * CARGO_SIZE[2], *yaw_to_quat(station.yaw)]],
        device=device,
        dtype=torch.float32,
    )
    cargo.write_root_pose_to_sim_index(root_pose=pose)
    cargo.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=device, dtype=torch.float32))


def run_simulator(sim: SimulationContext, scene: InteractiveScene) -> None:
    """Run the transport mission until all cycles are done or the step budget runs out.

    Args:
        sim: The simulation context.
        scene: The interactive scene holding the AGV, the box and the lidar.
    """
    robot: Articulation = scene["robot"]
    cargo: RigidObject = scene["cargo"]
    lidar: MultiMeshRayCaster = scene["lidar"]

    device = robot.device
    sim_dt = sim.get_physics_dt()

    wheel_ids, wheel_names = robot.find_joints(["left_wheel_joint", "right_wheel_joint"], preserve_order=True)
    lift_ids, _ = robot.find_joints(["lift_joint"])
    print(f"[INFO] 駆動輪ジョイント: {wheel_names} / リフトジョイント: lift_joint")

    pickup = Station("ピッキング棚", PICKUP_CENTER, PICKUP_YAW)
    dropoff = Station("搬出棚", DROPOFF_CENTER, DROPOFF_YAW)
    mission = TransportMission(pickup, dropoff, args_cli.cycles)
    scan = LidarScan(lidar)
    limiter = PayloadAwareLimiter(
        nominal_speed=args_cli.cruise_speed,
        friction_prior=args_cli.friction_prior,
        adaptive=not args_cli.no_adaptive,
    )

    commanded_speed = 0.0
    previous_speed = 0.0
    previous_forward_speed = 0.0
    previous_yaw_rate = 0.0
    longitudinal_accel = 0.0
    yaw_accel = 0.0
    commanded_yaw_rate = 0.0
    lift_command = 0.0
    camera_eye: tuple[float, float, float] | None = None
    camera_lookat: tuple[float, float, float] | None = None
    travelled = 0.0
    previous_position = (AGV_START_POSE[0], AGV_START_POSE[1])
    drops = 0
    step = 0
    sim_time = 0.0

    print("[INFO] シミュレーション開始: AGV がピッキング棚へ向かいます")
    while simulation_app.is_running() and step < args_cli.max_steps:
        # -- 知覚: 自己位置と LiDAR
        position_w = robot.data.root_pos_w.torch[0]
        position = (float(position_w[0].item()), float(position_w[1].item()))
        yaw = float(robot.data.heading_w.torch[0].item())
        body_velocity = robot.data.root_lin_vel_b.torch[0]
        forward_speed = float(body_velocity[0].item())
        yaw_rate_measured = float(robot.data.root_ang_vel_w.torch[0][2].item())
        scan.update()

        cargo_position_w = cargo.data.root_pos_w.torch[0]
        cargo_local = body_frame_offset(
            (float(cargo_position_w[0].item()), float(cargo_position_w[1].item())), position, yaw
        )

        # -- 判断: 状態機械を進め、走行指令を作る
        message = mission.advance(position, forward_speed, sim_dt)
        if message is not None:
            print(f"\n{message}")
            if mission.state == MissionState.EXIT_PICKUP:
                # 荷物が荷台に載りきったこの時点の相対位置を、荷ズレ計測の基準にする
                mission.carrying = True
                limiter.start_carrying(cargo_local)
            elif mission.state == MissionState.EXIT_DROPOFF:
                mission.carrying = False
                limiter.stop_carrying()
            elif mission.state == MissionState.GOTO_PICKUP and step > 0:
                # 次のサイクルぶんの荷物がピッキング棚に補充される
                place_cargo_on_station(cargo, pickup)
                print("[SIM] 新しい荷物がピッキング棚に入庫しました")
        if mission.state == MissionState.DONE:
            break

        speed_request, yaw_rate_request, avoiding = mission.drive_command(position, yaw, scan)

        # -- 判断: 積載状態からの速度制限 (フィジカルAI のフィードバック)
        # 荷物が載っている荷台は回転中心より DECK_CENTER_X だけ後ろにあるので、荷物が受ける
        # 水平加速度は 車体の前後加速度 + 遠心項 omega^2*r と、旋回の v*omega + 角加速度項 alpha*r。
        # 差分はノイズが乗るので、どちらも一次ローパスをかけてから使う。
        longitudinal_accel = 0.9 * longitudinal_accel + 0.1 * (forward_speed - previous_forward_speed) / sim_dt
        yaw_accel = 0.9 * yaw_accel + 0.1 * (yaw_rate_measured - previous_yaw_rate) / sim_dt
        previous_forward_speed = forward_speed
        previous_yaw_rate = yaw_rate_measured
        payload_accel = math.hypot(
            longitudinal_accel - yaw_rate_measured**2 * DECK_CENTER_X,
            forward_speed * yaw_rate_measured + yaw_accel * DECK_CENTER_X,
        )
        if mission.carrying:
            slip = limiter.observe(cargo_local, payload_accel, sim_dt)
            if slip is not None:
                print(
                    f"\n[PAYLOAD] 荷物が {slip * 100.0:.1f} cm ずれました"
                    f" (直前のピーク加速度 {limiter.slip_accel:.2f} m/s^2)。"
                    f" 摩擦推定値を {limiter.friction_estimate:.3f} に更新 →"
                    f" 許容加速度 {limiter.accel_limit:.2f} m/s^2"
                )
            if abs(cargo_local[1]) > SLIP_DROP or float(cargo_position_w[2].item()) < 0.2:
                drops += 1
                print("\n[PAYLOAD] 荷物が荷台から落下しました。積み直して再開します")
                pose = torch.tensor(
                    [[
                        position[0] + math.cos(yaw) * DECK_CENTER_X,
                        position[1] + math.sin(yaw) * DECK_CENTER_X,
                        DECK_TOP_Z + LIFT_STROKE + 0.5 * CARGO_SIZE[2] + 0.01,
                        *yaw_to_quat(yaw),
                    ]],
                    device=device,
                    dtype=torch.float32,
                )
                cargo.write_root_pose_to_sim_index(root_pose=pose)
                cargo.write_root_velocity_to_sim_index(
                    root_velocity=torch.zeros((1, 6), device=device, dtype=torch.float32)
                )
                limiter.start_carrying((DECK_CENTER_X, 0.0))

        commanded_yaw_rate = limiter.limit_yaw_rate(yaw_rate_request, commanded_yaw_rate, sim_dt)
        commanded_speed = limiter.limit(speed_request, commanded_yaw_rate, previous_speed, sim_dt)
        previous_speed = commanded_speed

        # -- 制御: 車輪速度とリフト位置
        robot.set_joint_velocity_target_index(
            target=wheel_targets(commanded_speed, commanded_yaw_rate, device), joint_ids=wheel_ids
        )
        lift_command += max(-LIFT_SPEED * sim_dt, min(LIFT_SPEED * sim_dt, mission.lift_target - lift_command))
        robot.set_joint_position_target_index(
            target=torch.tensor([[lift_command]], device=device, dtype=torch.float32), joint_ids=lift_ids
        )

        # -- 追従カメラ: 描画するフレームでだけ、描画の直前に更新する。
        # カメラの更新が描画より粗いと、視点が段階的に飛んで画面全体がカクついて見える。
        # 毎描画フレーム動かし、さらに一次ローパスで滑らかに追従させる。
        render_this_step = step % RENDER_INTERVAL == 0
        if render_this_step and not args_cli.headless:
            desired_eye = (
                position[0] - CAMERA_DISTANCE * math.cos(yaw),
                position[1] - CAMERA_DISTANCE * math.sin(yaw),
                CAMERA_HEIGHT,
            )
            desired_lookat = (position[0], position[1], 0.3)
            if camera_eye is None:
                camera_eye, camera_lookat = desired_eye, desired_lookat
            else:
                camera_eye = tuple(
                    now + CAMERA_SMOOTHING * (goal - now) for now, goal in zip(camera_eye, desired_eye)
                )
                camera_lookat = tuple(
                    now + CAMERA_SMOOTHING * (goal - now) for now, goal in zip(camera_lookat, desired_lookat)
                )
            sim.set_camera_view(camera_eye, camera_lookat)

        scene.write_data_to_sim()
        # 物理は毎ステップ、描画は RENDER_INTERVAL ステップに 1 回だけ。SimulationCfg.render_interval は
        # 標準ループ (DirectRLEnv など) 用の設定で、自前ループでは sim.step(render=...) で間引く必要がある。
        sim.step(render=render_this_step)
        sim_time += sim_dt
        step += 1
        scene.update(sim_dt)

        # -- 進捗表示
        travelled += math.hypot(position[0] - previous_position[0], position[1] - previous_position[1])
        previous_position = position
        if step % 30 == 0:
            nearest_distance, nearest_bearing = scan.nearest()
            print(
                f"[{mission.state.name:<13}] 位置:({position[0]:6.2f},{position[1]:6.2f})"
                f" 速度:{forward_speed:5.2f}m/s 指令:{commanded_speed:5.2f}m/s"
                f" 最近傍障害物:{nearest_distance:4.1f}m@{math.degrees(nearest_bearing):+4.0f}deg"
                f" {'回避中' if avoiding else '直進中'}"
                f" 荷ズレ:{limiter.slip(cargo_local) * 100.0:5.1f}cm"
                f" mu推定:{limiter.friction_estimate:.3f}   ",
                end="\r",
            )

    print("\n\n================ 搬送サマリ ================")
    print(f"  完了サイクル数      : {mission.cycle} / {args_cli.cycles}")
    print(f"  シミュレーション時間: {sim_time:.1f} s ({step} ステップ)")
    print(f"  走行距離            : {travelled:.1f} m")
    print(f"  荷ズレ検知回数      : {limiter.slip_events} 回")
    print(f"  荷物の落下          : {drops} 回")
    print(f"  ドッキングやり直し  : {mission.dock_retries} 回")
    print(f"  スタック復帰        : {mission.stuck_recoveries} 回")
    print(f"  荷物にかかった最大加速度: {limiter.max_payload_accel:.2f} m/s^2")
    print(
        f"  摩擦係数の推定      : {limiter.friction_estimate:.3f}"
        f" (初期値 {args_cli.friction_prior:.3f} / 実際の設定値 {args_cli.deck_friction:.3f})"
    )
    print(f"  学習された許容加速度: {limiter.accel_limit:.2f} m/s^2")
    print(f"  適応制御            : {'無効 (--no_adaptive)' if args_cli.no_adaptive else '有効'}")
    print("===========================================")


def main() -> None:
    """Build the warehouse scene and run the AGV transport mission."""
    # 物理は PHYSICS_RATE、描画は RENDER_INTERVAL に従って間引く (実際の間引きは run_simulator 側)。
    sim_cfg = sim_utils.SimulationCfg(
        device=args_cli.device, dt=1.0 / PHYSICS_RATE, render_interval=RENDER_INTERVAL
    )
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view((-4.0, -9.0, 6.0), (4.0, 0.0, 0.5))

    # 静的な倉庫設備は LiDAR がパースする前に置いておく
    spawn_warehouse_props()

    scene_cfg = WarehouseSceneCfg(num_envs=1, env_spacing=25.0)
    scene = InteractiveScene(scene_cfg)

    # 荷台を滑りやすい金属面にする。ここが荷崩れの物理的な源になる。
    deck_material_cfg = sim_utils.RigidBodyMaterialCfg(
        static_friction=args_cli.deck_friction,
        dynamic_friction=args_cli.deck_friction * 0.85,
        restitution=0.0,
    )
    deck_material_cfg.func("/World/Materials/deck", deck_material_cfg)
    sim_utils.bind_physics_material("/World/envs/env_0/Robot/lift", "/World/Materials/deck")

    # AGV の見た目。既定では USD の高解像度メッシュを隠して当たり判定形状を表示する (理由は関数の docstring)。
    if args_cli.visual_mode == "proxy":
        use_collision_shape_visuals("/World/envs/env_0/Robot")

    sim.reset()
    print("[INFO] シーンの準備が完了しました")
    print(f"[INFO] AGV: idealworks iw.hub ({IW_HUB_USD_PATH})")
    print(f"[INFO] 荷物: {CARGO_SIZE[0]}x{CARGO_SIZE[1]}x{CARGO_SIZE[2]} m / {args_cli.payload_mass} kg")
    print(f"[INFO] 荷台の摩擦係数: {args_cli.deck_friction} (コントローラの初期推定値: {args_cli.friction_prior})")

    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
