# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab 3.0+: SO-101 アームで「吊り荷の振れ止め (anti-sway)」を自律制御するデモ。

TheRobotStudio の 5-DOF アーム SO-101 のグリッパから、ワイヤ (長さ ``--rope_length``) で
キューブ状の荷物を吊り下げ、水平に搬送する。吊り荷は振り子なので、加減速するたびに振れる。
このデモは **その振れをアーム側の動きだけで打ち消す** 制御を実装し、制御なしの場合と定量比較する。

工場で天井クレーンやガントリーが機械を吊って移動させるときとまったく同じ問題で、
ここで使っている 2 つの手法も実機のクレーン制御でそのまま使われているものです。

制御の中身
----------

1. **入力整形 (ZV input shaper) — フィードフォワード**

   振り子の固有角振動数 :math:`\\omega_n = \\sqrt{g/L}` が分かっていれば、指令速度を
   「いま」と「半周期後」の 2 発に振り分けて与えるだけで、移動が終わった瞬間に振れがゼロになる
   (2 発目のインパルスが 1 発目の起こした振れをちょうど打ち消す)。半周期ぶん到着が遅れる代わりに
   残留振れがほぼ消える、クレーン業界では定番の手法。**外乱には効かない** (開ループなので)。

2. **振れ角フィードバック — クローズドループ**

   荷物の位置を計測し、吊り点 (グリッパ) からの水平ずれから振れ角 :math:`\\theta` を求めて、
   吊り点を「荷物が振れた方向へ」速度 :math:`v = k_\\theta \\theta` で動かす。
   振り子の運動方程式 :math:`L\\ddot\\theta = -g\\theta - \\ddot x_s` に
   :math:`\\dot x_s = k_\\theta\\theta` を入れると
   :math:`L\\ddot\\theta + k_\\theta\\dot\\theta + g\\theta = 0` となり、
   **速度指令がそのまま粘性減衰項になる**。目標減衰比 :math:`\\zeta` に対して
   :math:`k_\\theta = 2\\zeta\\sqrt{gL}` と置けばよい (``--damping_ratio``)。
   こちらは外乱 (荷物を横から突く / 風) にも反応して立て直せる。

``--controller`` で ``none`` / ``shaper`` / ``feedback`` / ``both`` を選べる。既定では
``none`` と ``both`` を続けて実行し、最後に最大振れ角・残留振れ・整定時間を表で比較する。

実測値 (既定パラメータ, ワイヤ 0.15 m / 荷物 50 g / 搬送 220 mm)::

    モード      搬送中の最大振れ   残留振れ   整定時間   外乱後の最大振れ  外乱からの復帰  着地誤差
    none           24.60 deg    20.067 deg   未整定      19.72 deg      未整定     56.8 mm
    shaper          5.31 deg     1.882 deg   未整定      15.70 deg      未整定      2.3 mm
    feedback        5.64 deg     0.002 deg   0.31 s       9.87 deg      0.32 s      7.3 mm
    both            3.07 deg     0.002 deg   0.26 s       9.24 deg      0.32 s      5.9 mm

入力整形だけ (``shaper``) でも搬送中の振れは 1/5 になるが、**外乱を与えると立て直せない**
(15.70 deg のまま整定しない) — 開ループなので当然で、クローズドループが要る理由がそのまま出る。

サーボ帯域について
------------------

振れ角フィードバックは「吊り点が振れに追従して動けること」が前提なので、アームの位置制御が
振り子より十分速くないと機能しない。SO-101 の標準の高PDゲイン (400/80) は時定数
``damping / stiffness`` = 0.2 s で、振り子の周期 0.78 s に対して遅すぎ、指令した補正が
約 60 deg 遅れて実現されるため減衰項が実質ばね項に化けてしまう (実効減衰比が設計値の 1/6 程度)。
このデモでは :data:`ARM_STIFFNESS` / :data:`ARM_DAMPING` で 800/25 (時定数 0.031 s) に
調整し直している。実機のクレーンでも同じ話で、**振れ止め制御の性能はトロリーの応答速度で決まる**。

ワイヤのモデル化
----------------

ワイヤは「引っ張る方向にしか力が出ないバネダンパ (unilateral cable)」として毎ステップ解析的に
計算し、:meth:`instantaneous_wrench_composer.set_forces_and_torques_index` で荷物とグリッパの
両方に作用・反作用として与えている。多リンクのロープを物理で組むより安定で、張力をそのまま
ログに出せる。力は荷物の重心に与えているので荷物は回転しない (質点振り子モデル)。

.. code-block:: bash

    # 既定: 制御なし -> 制御あり を続けて実行して比較表を出す
    isaaclab.bat -p demo6.py --viz kit
    ./isaaclab.sh -p demo6.py --viz kit

    # 4 モードすべて比較する
    isaaclab.bat -p demo6.py --viz kit --controller none shaper feedback both

    # 制御ありだけ、ワイヤを長く・荷物を重くする
    isaaclab.bat -p demo6.py --viz kit --controller both --rope_length 0.20 --payload_mass 0.08

    # 画面なしで数値だけ取る / 時系列を CSV に残す
    isaaclab.bat -p demo6.py --viz none --log_csv sway.csv

    # 外乱 (荷物を横から突く) を無効にする
    isaaclab.bat -p demo6.py --viz kit --kick_speed 0
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Anti-sway control of a cable-suspended payload on the SO-101 arm.")
parser.add_argument(
    "--controller",
    nargs="+",
    default=["none", "both"],
    choices=["none", "shaper", "feedback", "both"],
    help=(
        "Controller variants to run back to back, then compare. 'none' = raw trajectory, 'shaper' = ZV input"
        " shaping only (feed-forward), 'feedback' = sway-angle feedback only, 'both' = shaping + feedback."
    ),
)
parser.add_argument("--rope_length", type=float, default=0.15, help="Free length [m] of the suspension cable.")
parser.add_argument("--payload_mass", type=float, default=0.05, help="Mass [kg] of the suspended payload.")
parser.add_argument("--traverse_span", type=float, default=0.22, help="Horizontal traverse distance [m].")
parser.add_argument("--traverse_speed", type=float, default=0.25, help="Cruise speed [m/s] of the traverse.")
parser.add_argument(
    "--traverse_accel",
    type=float,
    default=1.5,
    help="Acceleration [m/s^2] of the traverse. This is what excites the sway: higher = worse.",
)
parser.add_argument(
    "--damping_ratio",
    type=float,
    default=0.7,
    help="Target closed-loop damping ratio [dimensionless] of the pendulum. Sets the feedback gain.",
)
parser.add_argument(
    "--cable_stiffness", type=float, default=400.0, help="Axial stiffness [N/m] of the cable spring-damper model."
)
parser.add_argument(
    "--cable_damping_ratio",
    type=float,
    default=0.25,
    help="Axial damping of the cable, as a fraction [dimensionless] of critical damping.",
)
parser.add_argument(
    "--kick_speed",
    type=float,
    default=0.35,
    help="Lateral velocity [m/s] imparted to the payload by the disturbance kick. 0 disables the kick.",
)
parser.add_argument("--log_csv", type=str, default=None, help="Write the per-step sway time series to this CSV file.")
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

import collections
import csv
import math
import numpy as np
import torch
from collections.abc import Callable
from dataclasses import dataclass, field

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

PHYSICS_RATE = 240.0
"""Physics steps per second [Hz].

The cable is a stiff spring (``--cable_stiffness``), so the step has to resolve its axial mode:
with the default 400 N/m and a 50 g payload the axial frequency is ~89 rad/s, i.e. ~16 steps per
axial period at 240 Hz. Dropping to 60 Hz makes the cable ring and eventually blow up.
"""
RENDER_INTERVAL = 4
"""Physics steps per rendered frame. At :data:`PHYSICS_RATE` = 240 Hz this renders the viewer at 60 Hz."""

GRAVITY = 9.81
"""Magnitude of gravity [m/s^2]. Used for the pendulum model; matches the simulation default."""

ARM_STIFFNESS = 800.0
"""Position-loop stiffness [N·m/rad] of the arm joints, overriding :obj:`SO101_HIGH_PD_CFG`."""
ARM_DAMPING = 25.0
"""Position-loop damping [N·m·s/rad] of the arm joints, overriding :obj:`SO101_HIGH_PD_CFG`.

The stock high-PD gains (400 / 80) are tuned for slow, accurate pick-and-place. They make the
joint servo heavily overdamped, with a dominant time constant of ``damping / stiffness`` = 0.2 s.
That is a quarter of the pendulum's 0.78 s period, so the gripper reaches a commanded anti-sway
correction roughly 60 deg out of phase -- which converts the intended damping term into a spring
term and leaves the measured closed-loop damping at a fraction of the design value.

Anti-sway feedback only works if the support tracks the swing mode, so the servo is retuned for
bandwidth instead: 800 / 25 gives ``damping / stiffness`` = 0.031 s, an order of magnitude below
the swing period, while staying comfortably overdamped for this arm's link inertias.
"""

##
# シーンの寸法 [m]
##

PEDESTAL_SIZE = (0.10, 0.10, 0.12)
"""Size (x, y, z) [m] of the pedestal the arm is bolted to. Lifts the arm so the payload clears the floor."""
PAYLOAD_SIZE = (0.035, 0.035, 0.035)
"""Edge lengths [m] of the suspended payload cube."""
MIN_PAYLOAD_CLEARANCE = 0.03
"""Minimum gap [m] we insist on between the hanging payload's underside and the floor."""

##
# 制御パラメータ
##

PLANT_DAMPING_RATIO = 0.02
"""Damping ratio [dimensionless] of the *uncontrolled* pendulum, used to tune the ZV shaper.

The cable damper acts along the cable axis only, so the swing mode is almost undamped. A small
non-zero value keeps the shaper's impulse amplitudes well conditioned.
"""
MAX_CORRECTION = 0.07
"""Maximum magnitude [m] the sway feedback may shift the end-effector away from its reference path."""
MAX_CORRECTION_SPEED = 0.40
"""Maximum magnitude [m/s] of the feedback velocity correction, before it is integrated."""
CORRECTION_LEAK_TAU = 4.0
"""Time constant [s] the accumulated correction decays back to zero with.

Without the leak the integrator keeps whatever offset it accumulated, so the payload would settle
next to the commanded drop point instead of on it. At 4 s the leak is far slower than the swing
mode, so it neither eats into the damping nor re-excites the pendulum on its way back.
"""
ANGLE_FILTER_TAU = 0.02
"""Low-pass time constant [s] on the measured swing angle. Rejects contact/solver chatter."""
SETTLED_ANGLE_DEG = 1.0
"""Swing amplitude [deg] below which the payload counts as settled."""
SETTLED_HOLD_S = 0.5
"""Duration [s] the swing must stay under :data:`SETTLED_ANGLE_DEG` before we call it settled."""

##
# ミッションの時間配分 [s]
##

PHASE_SETTLE_S = 1.2
"""Initial hold [s] that lets the payload hang still before the first traverse."""
PHASE_HOLD_S = 3.0
"""Hold [s] after each traverse. Long enough to read the residual sway and the settling time."""
PHASE_RECOVER_S = 4.5
"""Hold [s] after the disturbance kick, to watch the controller reject it."""
PHASE_FINAL_S = 1.5
"""Hold [s] at the end of the mission."""
KICK_DURATION_S = 0.06
"""Duration [s] the disturbance force is applied over."""

##
# SO-101 の関節
##

ARM_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
"""Actuated arm joints, in the order the IK controller sees them."""
ORIENTATION_JOINT_NAMES = ("wrist_flex", "wrist_roll")
"""Joints allowed to serve the orientation rows of the IK task (see :class:`SO101PoseIKControllerCfg`).

Keeping ``shoulder_pan`` out of this set is load bearing here: the traverse is a lateral move, i.e.
almost pure ``shoulder_pan``, and letting the base also chase the commanded orientation makes it
fight the traverse.
"""
EE_BODY_NAME = "gripper"
"""Body the cable is attached to. ``gripper`` is the static wrist body; ``moving_jaw_so101_v1`` is the jaw."""
GRIPPER_JOINT_NAME = "gripper"
"""The single revolute jaw joint. Held shut for the whole demo: the cable is a sling, not a grasp."""
GRIPPER_CLOSED_RAD = 0.0
"""Jaw angle [rad] we hold the gripper at."""
SO101_INIT_JOINT_POS: dict[str, float] = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -0.6,
    "elbow_flex": 0.8,
    "wrist_flex": 0.6,
    "wrist_roll": 0.0,
    "gripper": GRIPPER_CLOSED_RAD,
}
"""Starting joint pose [rad]. Mid-range (elbow/wrist bent) to stay off the 5-DOF arm's boundary
singularity; copied from the SO-101 cube-stack task."""


##
# ヘルパ
##


def quat_from_z_axis(vec: np.ndarray) -> np.ndarray:
    """Build the shortest-arc quaternion rotating +z onto ``vec``.

    Used to orient the cylinder marker that draws the cable.

    Args:
        vec: Direction to point +z at [m], shape (3,). Need not be normalized.

    Returns:
        The quaternion in (x, y, z, w), shape (4,).
    """
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0])
    unit = vec / norm
    cos_angle = float(unit[2])
    if cos_angle < -1.0 + 1e-6:
        # anti-parallel: any axis perpendicular to z works, pick x
        return np.array([1.0, 0.0, 0.0, 0.0])
    # cross([0, 0, 1], unit) == (-unit_y, unit_x, 0)
    quat = np.array([-unit[1], unit[0], 0.0, 1.0 + cos_angle])
    return quat / float(np.linalg.norm(quat))


def clip_norm(vec: np.ndarray, max_norm: float) -> np.ndarray:
    """Scale ``vec`` down so its norm does not exceed ``max_norm`` (direction preserved)."""
    norm = float(np.linalg.norm(vec))
    if norm <= max_norm or norm < 1e-12:
        return vec
    return vec * (max_norm / norm)


class TrapezoidProfile:
    """Single-axis trapezoidal (or triangular) velocity profile.

    The reference the crane would be given if nobody worried about the swinging load: ramp up at
    ``a_max``, cruise at ``v_max``, ramp down. Degenerates to a triangle when the move is too short
    to reach ``v_max``.
    """

    def __init__(self, distance: float, v_max: float, a_max: float):
        """
        Args:
            distance: Signed travel [m].
            v_max: Cruise speed [m/s].
            a_max: Acceleration and deceleration magnitude [m/s^2].
        """
        self.sign = 1.0 if distance >= 0.0 else -1.0
        self.accel = a_max
        travel = abs(distance)
        if travel <= v_max * v_max / a_max:
            # triangular: never reaches v_max
            self.ramp_time = math.sqrt(travel / a_max)
            self.peak_speed = a_max * self.ramp_time
            self.cruise_time = 0.0
        else:
            self.ramp_time = v_max / a_max
            self.peak_speed = v_max
            self.cruise_time = (travel - v_max * v_max / a_max) / v_max
        self.duration = 2.0 * self.ramp_time + self.cruise_time

    def velocity(self, t: float) -> float:
        """Reference velocity [m/s] at time ``t`` [s] since the start of the move."""
        if t < 0.0 or t >= self.duration:
            return 0.0
        if t < self.ramp_time:
            speed = self.accel * t
        elif t < self.ramp_time + self.cruise_time:
            speed = self.peak_speed
        else:
            speed = self.peak_speed - self.accel * (t - self.ramp_time - self.cruise_time)
        return self.sign * max(speed, 0.0)


class ZeroVibrationShaper:
    """Two-impulse (ZV) input shaper for a lightly damped second-order mode.

    Splits the command into an immediate part and a copy delayed by half a damped period. The
    delayed half cancels the oscillation the first half started, so the move ends with (ideally)
    zero residual swing, at the cost of finishing half a period later.

    Reference: Singer & Seering, "Preshaping Command Inputs to Reduce System Vibration" (1990).
    """

    def __init__(self, omega_n: float, zeta: float, dt: float):
        """
        Args:
            omega_n: Undamped natural frequency [rad/s] of the mode to cancel.
            zeta: Damping ratio [dimensionless] of that mode.
            dt: Control period [s].
        """
        zeta = min(max(zeta, 0.0), 0.95)
        damped = math.sqrt(1.0 - zeta * zeta)
        ratio = math.exp(-zeta * math.pi / damped)
        self.amplitude_now = 1.0 / (1.0 + ratio)
        self.amplitude_delayed = ratio / (1.0 + ratio)
        self.delay_s = math.pi / (omega_n * damped)
        self.delay_steps = max(1, int(round(self.delay_s / dt)))
        self._history: collections.deque[float] = collections.deque()
        self.reset()

    def reset(self) -> None:
        """Clear the delay line."""
        self._history = collections.deque((0.0 for _ in range(self.delay_steps)), maxlen=self.delay_steps)

    def __call__(self, command: float) -> float:
        """Shape one sample of the command and advance the delay line.

        Args:
            command: Reference speed [m/s] along the path.

        Returns:
            The shaped speed [m/s].
        """
        delayed = self._history[0]
        shaped = self.amplitude_now * command + self.amplitude_delayed * delayed
        self._history.append(float(command))
        return shaped


class CableModel:
    """Unilateral spring-damper cable between the gripper and the payload.

    A real sling only pulls. The model therefore produces a force along the cable axis that is
    clamped at zero, so the payload free-falls whenever the cable goes slack and is yanked back
    once it goes taut again.
    """

    def __init__(self, free_length: float, stiffness: float, damping_ratio: float, payload_mass: float):
        """
        Args:
            free_length: Unstretched cable length [m].
            stiffness: Axial stiffness [N/m].
            damping_ratio: Axial damping as a fraction [dimensionless] of critical damping.
            payload_mass: Payload mass [kg], used to size the critical damping.
        """
        self.free_length = free_length
        self.stiffness = stiffness
        self.damping = 2.0 * damping_ratio * math.sqrt(stiffness * payload_mass)

    def tension(
        self, anchor_pos: np.ndarray, anchor_vel: np.ndarray, payload_pos: np.ndarray, payload_vel: np.ndarray
    ) -> tuple[float, np.ndarray]:
        """Compute the cable tension and the force it exerts on the payload.

        Args:
            anchor_pos: Cable attachment point on the gripper, in world coordinates [m], shape (3,).
            anchor_vel: Velocity [m/s] of that attachment point, shape (3,).
            payload_pos: Payload centre of mass, in world coordinates [m], shape (3,).
            payload_vel: Payload velocity [m/s], shape (3,).

        Returns:
            A tuple of the tension magnitude [N] and the force on the payload [N], shape (3,).
        """
        offset = payload_pos - anchor_pos
        length = float(np.linalg.norm(offset))
        stretch = length - self.free_length
        if length < 1e-9 or stretch <= 0.0:
            return 0.0, np.zeros(3)
        axis = offset / length
        # positive when the payload is moving away from the anchor along the cable
        separation_rate = float(np.dot(payload_vel - anchor_vel, axis))
        tension = self.stiffness * stretch + self.damping * separation_rate
        if tension <= 0.0:
            return 0.0, np.zeros(3)
        return tension, -tension * axis


class SwayController:
    """Reference generator + anti-sway control for the suspended payload.

    Holds the two halves of the controller described in the module docstring and keeps them
    separable so the demo can switch either one off:

    * ``use_shaper`` -- run the reference speed through :class:`ZeroVibrationShaper`.
    * ``use_feedback`` -- add the sway-angle velocity correction.

    The end-effector command is built as ``path(s) + correction``, where ``s`` is the integral of
    the (possibly shaped) reference speed along the traverse path and ``correction`` is the leaky
    integral of the feedback velocity, clamped to :data:`MAX_CORRECTION`.
    """

    def __init__(
        self,
        path: Callable[[float], np.ndarray],
        rope_length: float,
        dt: float,
        use_shaper: bool,
        use_feedback: bool,
    ):
        """
        Args:
            path: Callable mapping an arc length [m] along the traverse to an end-effector position
                in the robot base frame [m], shape (3,). ``path(0.0)`` is the home pose.
            rope_length: Cable free length [m]. Sets the pendulum frequency and the feedback gain.
            dt: Control period [s].
            use_shaper: Whether to apply ZV input shaping to the reference.
            use_feedback: Whether to apply the sway-angle feedback.
        """
        self.path = path
        self.dt = dt
        self.use_shaper = use_shaper
        self.use_feedback = use_feedback
        self.omega_n = math.sqrt(GRAVITY / rope_length)
        self.period = 2.0 * math.pi / self.omega_n
        # v = k_theta * theta makes the support velocity act as a viscous damper on the swing mode,
        # with k_theta = 2 * zeta * sqrt(g * L). See the module docstring.
        self.feedback_gain = 2.0 * args_cli.damping_ratio * math.sqrt(GRAVITY * rope_length)
        self.shaper = ZeroVibrationShaper(self.omega_n, PLANT_DAMPING_RATIO, dt)
        self.arc_length = 0.0
        self.correction = np.zeros(3)
        self.angle = np.zeros(2)

    def reset(self) -> None:
        """Return the controller to its home state, ready for another run."""
        self.shaper.reset()
        self.arc_length = 0.0
        self.correction = np.zeros(3)
        self.angle = np.zeros(2)

    @property
    def shaper_delay_s(self) -> float:
        """Extra time [s] a move takes because of input shaping (0 when shaping is off)."""
        return self.shaper.delay_s if self.use_shaper else 0.0

    def update_measurement(self, swing_offset: np.ndarray, cable_length: float) -> np.ndarray:
        """Update the filtered swing angle from the payload's offset below the anchor.

        Args:
            swing_offset: Payload position minus anchor position, in the robot base frame [m], shape (3,).
            cable_length: Current anchor-to-payload distance [m].

        Returns:
            The filtered swing angle [rad] as its two horizontal components, shape (2,).
        """
        # small-angle: sin(theta) ~= horizontal offset / cable length, kept as a 2-vector so the
        # swing direction comes for free
        raw = swing_offset[:2] / max(cable_length, 1e-6)
        alpha = self.dt / (ANGLE_FILTER_TAU + self.dt)
        self.angle = self.angle + alpha * (raw - self.angle)
        return self.angle

    def step(self, reference_speed: float) -> np.ndarray:
        """Advance the command by one control period.

        Args:
            reference_speed: Unshaped reference speed [m/s] along the traverse path.

        Returns:
            The commanded end-effector position in the robot base frame [m], shape (3,).
        """
        shaped_speed = self.shaper(reference_speed) if self.use_shaper else reference_speed
        self.arc_length += shaped_speed * self.dt

        if self.use_feedback:
            feedback_velocity = np.zeros(3)
            feedback_velocity[:2] = self.feedback_gain * self.angle
            feedback_velocity = clip_norm(feedback_velocity, MAX_CORRECTION_SPEED)
            # leaky integrator: the correction decays back to zero so the payload ends up over the
            # commanded drop point rather than beside it
            self.correction = self.correction + (feedback_velocity - self.correction / CORRECTION_LEAK_TAU) * self.dt
            self.correction = clip_norm(self.correction, MAX_CORRECTION)
        else:
            self.correction = np.zeros(3)

        return self.path(self.arc_length) + self.correction


@dataclass
class RunMetrics:
    """Numbers one run of the mission produces, used for the end-of-demo comparison table."""

    mode: str
    """Controller variant this run used."""
    peak_traverse_deg: float = 0.0
    """Largest swing angle [deg] seen while the arm was traversing."""
    residual_deg: float = 0.0
    """Largest swing angle [deg] over the last 1.5 s of the hold after the traverse."""
    settle_time_s: float | None = None
    """Time [s] from the end of the traverse until the swing stayed under :data:`SETTLED_ANGLE_DEG`.
    ``None`` when it never settled within the hold."""
    peak_kick_deg: float = 0.0
    """Largest swing angle [deg] caused by the disturbance kick."""
    kick_recovery_s: float | None = None
    """Time [s] from the kick until the swing settled again, or ``None`` if it never did."""
    traverse_wall_s: float = 0.0
    """How long [s] the traverse phase took, including the delay input shaping adds."""
    placement_error_mm: float = 0.0
    """Horizontal distance [mm] between the payload and the commanded drop point at the end of the hold."""
    samples: list[tuple[float, float, float]] = field(default_factory=list)
    """Per-step ``(time [s], swing angle [deg], cable tension [N])`` series, for ``--log_csv``."""


##
# シーン定義
##


@configclass
class CraneSceneCfg(InteractiveSceneCfg):
    """Ground, light, pedestal-mounted SO-101 and the free-flying payload.

    The cable itself is not a scene asset: it is the analytic :class:`CableModel` plus a cylinder
    marker drawn between the two ends every frame.
    """

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.92, 0.92, 0.96))
    )

    pedestal = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Pedestal",
        spawn=sim_utils.CuboidCfg(
            size=PEDESTAL_SIZE,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.27, 0.30), roughness=0.7),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5 * PEDESTAL_SIZE[2])),
    )

    # SO-101 seated on top of the pedestal, unrotated. The high-PD variant is used because the
    # default low-stiffness gains track task-space (IK) targets poorly; its gains are then retuned
    # for servo bandwidth (see ARM_STIFFNESS / ARM_DAMPING).
    robot = SO101_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=SO101_HIGH_PD_CFG.init_state.replace(
            pos=(0.0, 0.0, PEDESTAL_SIZE[2]),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos=dict(SO101_INIT_JOINT_POS),
        ),
    )
    robot.actuators["arm"].stiffness = ARM_STIFFNESS
    robot.actuators["arm"].damping = ARM_DAMPING

    payload = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Payload",
        spawn=sim_utils.CuboidCfg(
            size=PAYLOAD_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=args_cli.payload_mass),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.35, 0.10), roughness=0.6),
        ),
        # placed properly once the end-effector home pose is known; this is only a safe first guess
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.2, 0.0, 0.2)),
    )


CABLE_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/Cable",
    markers={
        # unit-height cylinder: positioned at the cable midpoint and scaled along z to its length
        "cable": sim_utils.CylinderCfg(
            radius=0.0015,
            height=1.0,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.85, 0.20), roughness=1.0),
        )
    },
)
"""Marker that draws the cable as a thin cylinder between the gripper and the payload."""

STATION_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/Stations",
    markers={
        "pick": sim_utils.CylinderCfg(
            radius=0.035,
            height=0.002,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.20, 0.50, 0.90), roughness=1.0),
        ),
        "place": sim_utils.CylinderCfg(
            radius=0.035,
            height=0.002,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.20, 0.85, 0.40), roughness=1.0),
        ),
    },
)
"""Flat pads on the floor marking where the payload is supposed to hang at each end of the traverse."""


##
# ミッション実行
##


class CraneDemo:
    """Owns the sim handles and runs the anti-sway mission once per controller variant."""

    def __init__(self, sim: SimulationContext, scene: InteractiveScene):
        self.sim = sim
        self.scene = scene
        self.dt = sim.get_physics_dt()
        self.device = sim.device
        self.robot: Articulation = scene["robot"]
        self.payload: RigidObject = scene["payload"]

        # --- resolve the arm joints / end-effector body --------------------------------------
        self.arm_cfg = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES, body_names=[EE_BODY_NAME])
        self.arm_cfg.resolve(scene)
        self.gripper_cfg = SceneEntityCfg("robot", joint_names=[GRIPPER_JOINT_NAME])
        self.gripper_cfg.resolve(scene)
        self.ee_body_id = self.arm_cfg.body_ids[0]
        # the root body is not in the Jacobian, so a fixed-base robot's body index shifts down by one
        self.ee_jacobi_idx = self.ee_body_id - 1 if self.robot.is_fixed_base else self.ee_body_id
        self.jacobi_joint_ids = [j + self.robot.num_base_dofs for j in self.arm_cfg.joint_ids]

        # --- IK controller -------------------------------------------------------------------
        ik_cfg = SO101PoseIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            # manipulability-aware damped least squares: keeps the 5-DOF arm conditioned near
            # singularities, which matters because the traverse sweeps shoulder_pan
            ik_method="adaptive_dls",
            ik_params={"lambda_min": 0.05, "lambda_max": 0.2, "sigma_thresh": 0.02},
            orientation_weight=0.5,
            orientation_joint_names=ORIENTATION_JOINT_NAMES,
        )
        self.ik = SO101PoseIKController(ik_cfg, num_envs=scene.num_envs, device=self.device)
        # The mask indexes Jacobian columns, so it must follow the *resolved* joint order rather
        # than the order ARM_JOINT_NAMES happens to be written in.
        resolved_joint_names = [self.robot.joint_names[i] for i in self.arm_cfg.joint_ids]
        mask = torch.tensor(
            [1.0 if name in ORIENTATION_JOINT_NAMES else 0.0 for name in resolved_joint_names],
            device=self.device,
        )
        self.ik.set_orientation_joint_mask(mask)

        # --- markers -------------------------------------------------------------------------
        self.cable_marker = VisualizationMarkers(CABLE_MARKER_CFG)
        self.station_marker = VisualizationMarkers(STATION_MARKER_CFG)
        self.render_enabled = not args_cli.headless and "none" not in (args_cli.visualizer or [])

        # --- placeholders filled in by :meth:`calibrate` ---------------------------------------
        self.home_pos_b = np.zeros(3)
        self.home_quat_b = np.array([0.0, 0.0, 0.0, 1.0])
        self.rope_length = args_cli.rope_length
        self.cable: CableModel | None = None
        self.profile: TrapezoidProfile | None = None
        self.start_pos_b = np.zeros(3)
        self.goal_pos_b = np.zeros(3)
        self.pivot_radius = 1.0
        self.pivot_angle = 0.0
        self.default_joint_pos = self.robot.data.default_joint_pos.torch.clone()
        self.gripper_target = torch.full(
            (scene.num_envs, len(self.gripper_cfg.joint_ids)), GRIPPER_CLOSED_RAD, device=self.device
        )

    ##
    # セットアップ
    ##

    def calibrate(self) -> None:
        """Settle the arm at its start pose, measure the home end-effector pose, size the cable.

        Reading the home pose out of the simulation instead of hard-coding it keeps the demo honest
        about the SO-101's small workspace: the traverse and the cable are defined relative to
        wherever the arm actually parks.
        """
        self._reset_robot()
        for _ in range(int(0.5 * PHYSICS_RATE)):
            self._hold_current_joint_targets()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(self.dt)

        ee_pos_b, ee_quat_b = self._ee_pose_in_base()
        self.home_pos_b = ee_pos_b
        self.home_quat_b = ee_quat_b

        anchor_w = self._base_to_world(self.home_pos_b)
        lowest_allowed = 0.5 * PAYLOAD_SIZE[2] + MIN_PAYLOAD_CLEARANCE
        max_rope = float(anchor_w[2]) - lowest_allowed
        if self.rope_length > max_rope:
            print(
                f"[WARN] --rope_length {self.rope_length:.3f} m はグリッパ高さ {anchor_w[2]:.3f} m に対して長すぎます。"
                f" 床との干渉を避けるため {max_rope:.3f} m に切り詰めます。"
            )
            self.rope_length = max(max_rope, 0.05)

        self.cable = CableModel(
            free_length=self.rope_length,
            stiffness=args_cli.cable_stiffness,
            damping_ratio=args_cli.cable_damping_ratio,
            payload_mass=args_cli.payload_mass,
        )
        self.profile = TrapezoidProfile(
            distance=args_cli.traverse_span, v_max=args_cli.traverse_speed, a_max=args_cli.traverse_accel
        )

        # The traverse is an arc at constant reach around the base, i.e. a pure ``shoulder_pan``
        # slew, exactly like a jib crane swinging its boom. Moving along a base-frame axis instead
        # would change the reach along the way: the SO-101's home pose sits at only ~0.19 m from
        # the base and a 0.22 m straight move would run it past the edge of its workspace.
        self.pivot_radius = float(np.linalg.norm(self.home_pos_b[:2]))
        self.pivot_angle = math.atan2(float(self.home_pos_b[1]), float(self.home_pos_b[0]))
        self.start_pos_b = self.path(-0.5 * args_cli.traverse_span)
        self.goal_pos_b = self.path(+0.5 * args_cli.traverse_span)

        period = 2.0 * math.pi / math.sqrt(GRAVITY / self.rope_length)
        print("[INFO] シーンの準備が完了しました")
        print("[INFO] ロボット: TheRobotStudio SO-101 (5-DOF, 高PDゲイン構成)")
        print(f"[INFO] グリッパ原点 (ベース座標): {np.round(self.home_pos_b, 4).tolist()} m")
        print(f"[INFO] ワイヤ長 {self.rope_length:.3f} m / 荷物 {args_cli.payload_mass * 1000:.0f} g")
        print(f"[INFO] 振り子の固有周期 {period:.3f} s (固有角振動数 {math.sqrt(GRAVITY / self.rope_length):.2f} rad/s)")
        print(
            f"[INFO] 搬送: {args_cli.traverse_span * 1000:.0f} mm を 最大 {args_cli.traverse_speed:.2f} m/s /"
            f" 加速度 {args_cli.traverse_accel:.2f} m/s^2 で移動 (所要 {self.profile.duration:.2f} s)"
        )

        # aim the viewer at the workspace
        focus = self._base_to_world(self.home_pos_b)
        self.sim.set_camera_view(
            (float(focus[0]) + 0.55, float(focus[1]) - 0.60, float(focus[2]) + 0.25),
            (float(focus[0]), float(focus[1]), float(focus[2]) - 0.5 * self.rope_length),
        )

    def path(self, arc_length: float) -> np.ndarray:
        """Map an arc length along the traverse to an end-effector position in the base frame.

        The path is a circle of radius :attr:`pivot_radius` centred on the arm's base, at constant
        height, so the reach never changes and the move is a pure ``shoulder_pan`` slew.

        Args:
            arc_length: Distance travelled [m] along the arc from the home pose (signed).

        Returns:
            The end-effector position in the robot base frame [m], shape (3,).
        """
        angle = self.pivot_angle + arc_length / max(self.pivot_radius, 1e-6)
        return np.array(
            [self.pivot_radius * math.cos(angle), self.pivot_radius * math.sin(angle), self.home_pos_b[2]]
        )

    ##
    # 低レベルのヘルパ
    ##

    def _base_to_world(self, pos_b: np.ndarray) -> np.ndarray:
        """Transform a position [m] from the robot base frame into world coordinates."""
        root_pose = self.robot.data.root_pose_w.torch
        vec = torch.tensor(pos_b, dtype=torch.float32, device=self.device).unsqueeze(0)
        world = root_pose[:, 0:3] + quat_apply(root_pose[:, 3:7], vec)
        return world[0].cpu().numpy().astype(float)

    def _world_to_base_vector(self, vec_w: np.ndarray) -> np.ndarray:
        """Rotate a free vector from world coordinates into the robot base frame."""
        root_pose = self.robot.data.root_pose_w.torch
        vec = torch.tensor(vec_w, dtype=torch.float32, device=self.device).unsqueeze(0)
        return quat_apply_inverse(root_pose[:, 3:7], vec)[0].cpu().numpy().astype(float)

    def _base_to_world_vector(self, vec_b: np.ndarray) -> np.ndarray:
        """Rotate a free vector from the robot base frame into world coordinates."""
        root_pose = self.robot.data.root_pose_w.torch
        vec = torch.tensor(vec_b, dtype=torch.float32, device=self.device).unsqueeze(0)
        return quat_apply(root_pose[:, 3:7], vec)[0].cpu().numpy().astype(float)

    def _path_tangent(self, arc_length: float) -> np.ndarray:
        """Unit tangent of the traverse arc at ``arc_length``, in the robot base frame."""
        angle = self.pivot_angle + arc_length / max(self.pivot_radius, 1e-6)
        return np.array([-math.sin(angle), math.cos(angle), 0.0])

    def _ee_pose_in_base(self) -> tuple[np.ndarray, np.ndarray]:
        """Current end-effector position [m] and orientation (x, y, z, w) in the robot base frame."""
        ee_pose_w = self.robot.data.body_pose_w.torch[:, self.ee_body_id]
        root_pose_w = self.robot.data.root_pose_w.torch
        pos_b, quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        return pos_b[0].cpu().numpy().astype(float), quat_b[0].cpu().numpy().astype(float)

    def _anchor_state_world(self) -> tuple[np.ndarray, np.ndarray]:
        """World position [m] and linear velocity [m/s] of the cable attachment point on the gripper."""
        pos = self.robot.data.body_pos_w.torch[0, self.ee_body_id].cpu().numpy().astype(float)
        vel = self.robot.data.body_lin_vel_w.torch[0, self.ee_body_id].cpu().numpy().astype(float)
        return pos, vel

    def _payload_state_world(self) -> tuple[np.ndarray, np.ndarray]:
        """World position [m] and linear velocity [m/s] of the payload."""
        pos = self.payload.data.root_pos_w.torch[0].cpu().numpy().astype(float)
        vel = self.payload.data.root_lin_vel_w.torch[0].cpu().numpy().astype(float)
        return pos, vel

    def _hold_current_joint_targets(self) -> None:
        """Command the arm to stay where it is (used while settling before a run)."""
        joint_pos = self.robot.data.joint_pos.torch[:, self.arm_cfg.joint_ids]
        self.robot.set_joint_position_target_index(target=joint_pos, joint_ids=self.arm_cfg.joint_ids)
        self.robot.set_joint_position_target_index(
            target=self.gripper_target, joint_ids=self.gripper_cfg.joint_ids
        )

    def _reset_robot(self) -> None:
        """Teleport the arm back to its start joint pose with zero velocity."""
        joint_pos = self.default_joint_pos.clone()
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.write_joint_position_to_sim_index(position=joint_pos)
        self.robot.write_joint_velocity_to_sim_index(velocity=joint_vel)
        self.robot.reset()

    def _hang_payload_below(self, anchor_pos_b: np.ndarray) -> None:
        """Teleport the payload so it hangs straight down from ``anchor_pos_b`` with the cable taut."""
        anchor_w = self._base_to_world(anchor_pos_b)
        pose = torch.zeros((1, 7), device=self.device)
        pose[0, 0:3] = torch.tensor(anchor_w - np.array([0.0, 0.0, self.rope_length]), device=self.device)
        pose[0, 6] = 1.0
        self.payload.write_root_pose_to_sim_index(root_pose=pose)
        self.payload.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=self.device))
        self.payload.reset()

    ##
    # ミッション本体
    ##

    def run(self, mode: str) -> RunMetrics:
        """Run the full mission once with the given controller variant.

        Args:
            mode: One of ``none``, ``shaper``, ``feedback``, ``both``.

        Returns:
            The metrics collected over the run.
        """
        use_shaper = mode in ("shaper", "both")
        use_feedback = mode in ("feedback", "both")
        controller = SwayController(
            path=self.path,
            rope_length=self.rope_length,
            dt=self.dt,
            use_shaper=use_shaper,
            use_feedback=use_feedback,
        )
        metrics = RunMetrics(mode=mode)

        self._reset_robot()
        self._hang_payload_below(self.home_pos_b)
        self.ik.reset()
        # walk the command from the measured home pose over to the start of the traverse, so the
        # run opens with a real move rather than a jump
        approach = TrapezoidProfile(
            distance=-0.5 * args_cli.traverse_span,
            v_max=args_cli.traverse_speed,
            a_max=args_cli.traverse_accel,
        )

        traverse_duration = self.profile.duration + controller.shaper_delay_s
        # phase boundaries [s] along the mission timeline
        t_approach_end = approach.duration
        t_settle_end = t_approach_end + PHASE_SETTLE_S
        t_out_end = t_settle_end + traverse_duration
        t_hold_end = t_out_end + PHASE_HOLD_S
        t_kick_end = t_hold_end + KICK_DURATION_S
        t_recover_end = t_kick_end + PHASE_RECOVER_S
        t_back_end = t_recover_end + traverse_duration
        t_final_end = t_back_end + PHASE_FINAL_S

        kick_force = np.zeros(3)
        if args_cli.kick_speed > 0.0:
            # a short constant force that delivers the requested velocity change, pushed along the
            # traverse tangent so it excites the same swing direction the controller acts in
            tangent_w = self._base_to_world_vector(self._path_tangent(0.5 * args_cli.traverse_span))
            kick_force = tangent_w * (args_cli.payload_mass * args_cli.kick_speed / KICK_DURATION_S)

        print("")
        print("=" * 78)
        print(f"[RUN ] 制御モード: {mode}  (入力整形={'ON' if use_shaper else 'OFF'}, 振れ角FB={'ON' if use_feedback else 'OFF'})")
        if use_shaper:
            print(f"       入力整形の遅れ: {controller.shaper_delay_s:.3f} s (半周期)")
        if use_feedback:
            print(f"       フィードバックゲイン: {controller.feedback_gain:.3f} (m/s)/rad, 目標減衰比 {args_cli.damping_ratio}")
        print("=" * 78)

        # settling trackers, in mission time
        last_unsettled_out = t_out_end
        last_unsettled_kick = t_kick_end
        next_report = 0.0
        step = 0
        elapsed = 0.0

        while simulation_app.is_running() and elapsed < t_final_end:
            # ---- 計測 -----------------------------------------------------------------------
            anchor_w, anchor_vel_w = self._anchor_state_world()
            payload_w, payload_vel_w = self._payload_state_world()
            tension, cable_force_w = self.cable.tension(anchor_w, anchor_vel_w, payload_w, payload_vel_w)
            offset_w = payload_w - anchor_w
            cable_length = float(np.linalg.norm(offset_w))
            angle = controller.update_measurement(self._world_to_base_vector(offset_w), cable_length)
            angle_deg = math.degrees(float(np.linalg.norm(angle)))

            # ---- 軌道指令 -------------------------------------------------------------------
            reference_speed = 0.0
            if elapsed < t_approach_end:
                phase = "APPROACH"
                reference_speed = approach.velocity(elapsed)
            elif elapsed < t_settle_end:
                phase = "SETTLE"
            elif elapsed < t_out_end:
                phase = "TRAVERSE>"
                reference_speed = self.profile.velocity(elapsed - t_settle_end)
            elif elapsed < t_hold_end:
                phase = "HOLD"
            elif elapsed < t_kick_end:
                phase = "KICK"
            elif elapsed < t_recover_end:
                phase = "RECOVER"
            elif elapsed < t_back_end:
                phase = "TRAVERSE<"
                reference_speed = -self.profile.velocity(elapsed - t_recover_end)
            else:
                phase = "FINISH"

            command_pos_b = controller.step(reference_speed)

            # ---- 力を書き込む ----------------------------------------------------------------
            total_force_w = cable_force_w.copy()
            if phase == "KICK":
                total_force_w = total_force_w + kick_force
            self.payload.instantaneous_wrench_composer.set_forces_and_torques_index(
                forces=torch.tensor(total_force_w, dtype=torch.float32, device=self.device).view(1, 1, 3),
                torques=torch.zeros((1, 1, 3), device=self.device),
                is_global=True,
            )
            # Newton's third law: the cable pulls the gripper down towards the payload.
            self.robot.instantaneous_wrench_composer.set_forces_and_torques_index(
                forces=torch.tensor(-cable_force_w, dtype=torch.float32, device=self.device).view(1, 1, 3),
                torques=torch.zeros((1, 1, 3), device=self.device),
                body_ids=torch.tensor([self.ee_body_id], dtype=torch.int32, device=self.device),
                is_global=True,
            )

            # ---- IK -> 関節指令 ---------------------------------------------------------------
            command = torch.zeros((1, 7), device=self.device)
            command[0, 0:3] = torch.tensor(command_pos_b, dtype=torch.float32, device=self.device)
            command[0, 3:7] = torch.tensor(self.home_quat_b, dtype=torch.float32, device=self.device)
            self.ik.set_command(command)

            jacobian = self.robot.data.body_link_jacobian_w.torch[:, self.ee_jacobi_idx, :, self.jacobi_joint_ids]
            ee_pos_b, ee_quat_b = self._ee_pose_in_base()
            joint_pos = self.robot.data.joint_pos.torch[:, self.arm_cfg.joint_ids]
            joint_pos_des = self.ik.compute(
                torch.tensor(ee_pos_b, dtype=torch.float32, device=self.device).unsqueeze(0),
                torch.tensor(ee_quat_b, dtype=torch.float32, device=self.device).unsqueeze(0),
                jacobian,
                joint_pos,
            )
            self.robot.set_joint_position_target_index(target=joint_pos_des, joint_ids=self.arm_cfg.joint_ids)
            self.robot.set_joint_position_target_index(
                target=self.gripper_target, joint_ids=self.gripper_cfg.joint_ids
            )

            # ---- 統計 -----------------------------------------------------------------------
            metrics.samples.append((elapsed, angle_deg, tension))
            if phase == "TRAVERSE>":
                metrics.peak_traverse_deg = max(metrics.peak_traverse_deg, angle_deg)
            if t_settle_end <= elapsed < t_hold_end and angle_deg > SETTLED_ANGLE_DEG:
                last_unsettled_out = elapsed
            if elapsed >= t_hold_end - 1.5 and elapsed < t_hold_end:
                metrics.residual_deg = max(metrics.residual_deg, angle_deg)
            if t_hold_end <= elapsed < t_recover_end:
                metrics.peak_kick_deg = max(metrics.peak_kick_deg, angle_deg)
                if angle_deg > SETTLED_ANGLE_DEG:
                    last_unsettled_kick = elapsed

            # ---- 進行状況 --------------------------------------------------------------------
            if elapsed >= next_report:
                next_report = elapsed + 0.5
                tracking_mm = float(np.linalg.norm(ee_pos_b - command_pos_b)) * 1000.0
                achieved_mm = float(np.linalg.norm(ee_pos_b - self.path(controller.arc_length))) * 1000.0
                print(
                    f"  t={elapsed:5.2f}s {phase:<9s} 振れ角 {angle_deg:5.2f} deg"
                    f" | 張力 {tension:5.3f} N"
                    f" | 補正 指令 {np.linalg.norm(controller.correction) * 1000:5.1f} / 実現 {achieved_mm:5.1f} mm"
                    f" | 追従誤差 {tracking_mm:5.1f} mm"
                )

            # ---- 1 ステップ進める --------------------------------------------------------------
            render_this_step = self.render_enabled and step % RENDER_INTERVAL == 0
            if render_this_step:
                self._draw_markers(anchor_w, payload_w)
            self.scene.write_data_to_sim()
            self.sim.step(render=render_this_step)
            self.scene.update(self.dt)
            step += 1
            elapsed += self.dt

        # ---- 後処理 -------------------------------------------------------------------------
        metrics.traverse_wall_s = traverse_duration
        if last_unsettled_out < t_hold_end - SETTLED_HOLD_S:
            metrics.settle_time_s = max(last_unsettled_out - t_out_end, 0.0)
        if args_cli.kick_speed > 0.0 and last_unsettled_kick < t_recover_end - SETTLED_HOLD_S:
            metrics.kick_recovery_s = max(last_unsettled_kick - t_kick_end, 0.0)

        payload_w, _ = self._payload_state_world()
        drop_point_w = self._base_to_world(self.start_pos_b)
        metrics.placement_error_mm = float(np.linalg.norm(payload_w[:2] - drop_point_w[:2])) * 1000.0
        return metrics

    def _draw_markers(self, anchor_w: np.ndarray, payload_w: np.ndarray) -> None:
        """Redraw the cable cylinder and the two station pads."""
        offset = payload_w - anchor_w
        length = max(float(np.linalg.norm(offset)), 1e-4)
        self.cable_marker.visualize(
            translations=(0.5 * (anchor_w + payload_w)).astype(np.float32).reshape(1, 3),
            orientations=quat_from_z_axis(offset).astype(np.float32).reshape(1, 4),
            scales=np.array([[1.0, 1.0, length]], dtype=np.float32),
        )
        pads = np.stack([self._base_to_world(self.start_pos_b), self._base_to_world(self.goal_pos_b)], axis=0)
        pads[:, 2] = 0.001
        self.station_marker.visualize(translations=pads.astype(np.float32), marker_indices=[0, 1])


##
# レポート
##


def print_report(results: list[RunMetrics]) -> None:
    """Print the end-of-demo comparison table."""
    print("")
    print("=" * 96)
    print(" 吊り荷の振れ止め制御 — 比較結果")
    print("=" * 96)
    header = (
        f"{'モード':<10}{'搬送中の最大振れ':>18}{'残留振れ':>12}{'整定時間':>12}"
        f"{'外乱後の最大振れ':>18}{'外乱からの復帰':>16}{'搬送所要':>10}{'着地誤差':>10}"
    )
    print(header)
    print("-" * 96)
    for run in results:
        settle = f"{run.settle_time_s:.2f} s" if run.settle_time_s is not None else "未整定"
        recover = f"{run.kick_recovery_s:.2f} s" if run.kick_recovery_s is not None else "未整定"
        print(
            f"{run.mode:<10}{run.peak_traverse_deg:>15.2f} deg{run.residual_deg:>9.3f} deg"
            f"{settle:>12}{run.peak_kick_deg:>15.2f} deg{recover:>16}"
            f"{run.traverse_wall_s:>8.2f} s{run.placement_error_mm:>8.1f} mm"
        )
    print("-" * 96)
    print(" 残留振れ  : 搬送完了後の保持区間 最後の 1.5 秒での最大振れ角")
    print(" 整定時間  : 搬送完了から振れ角が 1.0 deg 未満に収まるまでの時間")
    print(" 着地誤差  : 最終的な荷物の水平位置と、指令した吊り点の水平距離")
    print("=" * 96)

    baseline = next((r for r in results if r.mode == "none"), None)
    best = next((r for r in results if r.mode == "both"), None)
    if baseline is not None and best is not None and baseline.residual_deg > 1e-6:
        reduction = 100.0 * (1.0 - best.residual_deg / baseline.residual_deg)
        print(
            f" => 制御なし {baseline.residual_deg:.2f} deg -> 制御あり {best.residual_deg:.3f} deg"
            f" (残留振れ {reduction:.2f} % 低減)"
        )


def write_csv(path: str, results: list[RunMetrics]) -> None:
    """Dump every run's time series into one CSV file."""
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mode", "time_s", "swing_angle_deg", "cable_tension_N"])
        for run in results:
            for time_s, angle_deg, tension in run.samples:
                writer.writerow([run.mode, f"{time_s:.4f}", f"{angle_deg:.4f}", f"{tension:.4f}"])
    print(f"[INFO] 時系列を {path} に書き出しました")


def main() -> None:
    """Build the scene and run the anti-sway mission once per requested controller variant."""
    sim_cfg = sim_utils.SimulationCfg(
        device=args_cli.device, dt=1.0 / PHYSICS_RATE, render_interval=RENDER_INTERVAL
    )
    sim = SimulationContext(sim_cfg)

    scene_cfg = CraneSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    demo = CraneDemo(sim, scene)
    demo.calibrate()

    results: list[RunMetrics] = []
    for mode in args_cli.controller:
        if not simulation_app.is_running():
            break
        results.append(demo.run(mode))

    if results:
        print_report(results)
        if args_cli.log_csv:
            write_csv(args_cli.log_csv, results)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
