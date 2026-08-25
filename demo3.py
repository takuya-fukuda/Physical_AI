# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab 6.0+: カメラ(RGB-D)を使ってキューブの位置を推定し、Frankaアームでピック&リフトするデモ。

固定視点の俯瞰カメラで作業台を撮影し、緑色に着色したキューブをRGB画像から色抽出でセグメンテーションした後、
深度画像とカメラの内部/外部パラメータを使って3D位置を逆投影で推定する。推定したキューブ位置を
`lift_cube_sm.py` と同じ warp ベースのピック&リフト・ステートマシンに入力することで、
「カメラを使ったピッキング」を実現する。

推定は「見てから動く (look-then-act)」方式で、ステートマシンが REST の間だけカメラ推定を更新し、
十分なサンプルが溜まったらその中央値をラッチする。アームが降りてキューブを遮蔽し始めても
推定値は動かないため、把持が安定する。

.. code-block:: bash

    isaaclab.bat -p demo3.py --viz kit
    ./isaaclab.sh -p demo3.py --viz kit

    # 解像度を上げる場合
    isaaclab.bat -p demo3.py --viz kit --camera_width 1280 --camera_height 960
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Camera-based pick and lift demo for a Franka arm.")
parser.add_argument(
    "--position_threshold", type=float, default=0.01, help="Position threshold [m] for the pick state machine."
)
parser.add_argument("--camera_width", type=int, default=640, help="Width [px] of the overhead camera image.")
parser.add_argument("--camera_height", type=int, default=480, help="Height [px] of the overhead camera image.")
parser.add_argument("--episode_length_s", type=float, default=8.0, help="Episode length [s] before the scene resets.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# demos should open Kit visualizer by default
parser.set_defaults(visualizer=["kit"])
# camera sensors require rendering to be enabled
parser.set_defaults(enable_cameras=True)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything else."""

from collections.abc import Sequence

import gymnasium as gym
import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.assets.rigid_object.rigid_object_data import RigidObjectData
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.sensors.camera.utils import create_pointcloud_from_depth

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.core.lift.lift_env_cfg import LiftEnvCfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# initialize warp
wp.init()

# overhead camera placement, expressed relative to the environment origin [m]
CAMERA_EYE = (0.5, 0.0, 1.15)
CAMERA_LOOK_AT = (0.5, 0.0, 0.0)
# nominal cube position used only while the cube has not been observed yet [m]
NOMINAL_CUBE_POSITION = (0.5, 0.0, 0.035)


class GripperState:
    """States for the gripper."""

    OPEN = wp.constant(1.0)
    CLOSE = wp.constant(-1.0)


class PickSmState:
    """States for the pick state machine."""

    REST = wp.constant(0)
    APPROACH_ABOVE_OBJECT = wp.constant(1)
    APPROACH_OBJECT = wp.constant(2)
    GRASP_OBJECT = wp.constant(3)
    LIFT_OBJECT = wp.constant(4)


class PickSmWaitTime:
    """Additional wait times (in s) for states for before switching."""

    REST = wp.constant(0.2)
    APPROACH_ABOVE_OBJECT = wp.constant(0.5)
    APPROACH_OBJECT = wp.constant(0.6)
    GRASP_OBJECT = wp.constant(0.3)
    LIFT_OBJECT = wp.constant(1.0)


@wp.func
def distance_below_threshold(current_pos: wp.vec3, desired_pos: wp.vec3, threshold: float) -> bool:
    return wp.length(current_pos - desired_pos) < threshold


@wp.kernel
def infer_state_machine(
    dt: wp.array(dtype=float),
    sm_state: wp.array(dtype=int),
    sm_wait_time: wp.array(dtype=float),
    ee_pose: wp.array(dtype=wp.transform),
    object_pose: wp.array(dtype=wp.transform),
    des_object_pose: wp.array(dtype=wp.transform),
    des_ee_pose: wp.array(dtype=wp.transform),
    gripper_state: wp.array(dtype=float),
    offset: wp.array(dtype=wp.transform),
    position_threshold: float,
):
    # retrieve thread id
    tid = wp.tid()
    # retrieve state machine state
    state = sm_state[tid]
    # decide next state
    if state == PickSmState.REST:
        des_ee_pose[tid] = ee_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.REST:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.APPROACH_ABOVE_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.APPROACH_ABOVE_OBJECT:
        des_ee_pose[tid] = wp.transform_multiply(offset[tid], object_pose[tid])
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            # wait for a while
            if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_OBJECT:
                # move to next state and reset wait time
                sm_state[tid] = PickSmState.APPROACH_OBJECT
                sm_wait_time[tid] = 0.0
    elif state == PickSmState.APPROACH_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_OBJECT:
                # move to next state and reset wait time
                sm_state[tid] = PickSmState.GRASP_OBJECT
                sm_wait_time[tid] = 0.0
    elif state == PickSmState.GRASP_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.GRASP_OBJECT:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.LIFT_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.LIFT_OBJECT:
        des_ee_pose[tid] = des_object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            # wait for a while
            if sm_wait_time[tid] >= PickSmWaitTime.LIFT_OBJECT:
                # move to next state and reset wait time
                sm_state[tid] = PickSmState.LIFT_OBJECT
                sm_wait_time[tid] = 0.0
    # increment wait time
    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]


class PickAndLiftSm:
    """A simple state machine in a robot's task space to pick and lift an object.

    The state machine is implemented as a warp kernel. It takes in the current state of
    the robot's end-effector and the object, and outputs the desired state of the robot's
    end-effector and the gripper. The state machine is implemented as a finite state
    machine with the following states:

    1. REST: The robot is at rest.
    2. APPROACH_ABOVE_OBJECT: The robot moves above the object.
    3. APPROACH_OBJECT: The robot moves to the object.
    4. GRASP_OBJECT: The robot grasps the object.
    5. LIFT_OBJECT: The robot lifts the object to the desired pose. This is the final state.
    """

    def __init__(self, dt: float, num_envs: int, device: torch.device | str = "cpu", position_threshold=0.01):
        """Initialize the state machine.

        Args:
            dt: The environment time step.
            num_envs: The number of environments to simulate.
            device: The device to run the state machine on.
        """
        # save parameters
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = device
        self.position_threshold = position_threshold
        # initialize state machine
        self.sm_dt = torch.full((self.num_envs,), self.dt, device=self.device)
        self.sm_state = torch.full((self.num_envs,), 0, dtype=torch.int32, device=self.device)
        self.sm_wait_time = torch.zeros((self.num_envs,), device=self.device)

        # desired state
        self.des_ee_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.des_gripper_state = torch.full((self.num_envs,), 0.0, device=self.device)

        # approach above object offset
        self.offset = torch.zeros((self.num_envs, 7), device=self.device)
        self.offset[:, 2] = 0.1
        self.offset[:, -1] = 1.0  # warp expects quaternion as (x, y, z, w)

        # convert to warp
        self.sm_dt_wp = wp.from_torch(self.sm_dt, wp.float32)
        self.sm_state_wp = wp.from_torch(self.sm_state, wp.int32)
        self.sm_wait_time_wp = wp.from_torch(self.sm_wait_time, wp.float32)
        self.des_ee_pose_wp = wp.from_torch(self.des_ee_pose, wp.transform)
        self.des_gripper_state_wp = wp.from_torch(self.des_gripper_state, wp.float32)
        self.offset_wp = wp.from_torch(self.offset, wp.transform)

    def reset_idx(self, env_ids: Sequence[int] = None):
        """Reset the state machine."""
        if env_ids is None:
            env_ids = slice(None)
        self.sm_state[env_ids] = 0
        self.sm_wait_time[env_ids] = 0.0

    def hold_at_rest(self):
        """Keep the state machine from leaving :attr:`PickSmState.REST` on the next step.

        Used to stall the robot until the vision pipeline has actually located the cube.
        """
        self.sm_wait_time[:] = 0.0

    def compute(self, ee_pose: torch.Tensor, object_pose: torch.Tensor, des_object_pose: torch.Tensor) -> torch.Tensor:
        """Compute the desired state of the robot's end-effector and the gripper."""

        # convert to warp
        ee_pose_wp = wp.from_torch(ee_pose.contiguous(), wp.transform)
        object_pose_wp = wp.from_torch(object_pose.contiguous(), wp.transform)
        des_object_pose_wp = wp.from_torch(des_object_pose.contiguous(), wp.transform)

        # run state machine
        wp.launch(
            kernel=infer_state_machine,
            dim=self.num_envs,
            inputs=[
                self.sm_dt_wp,
                self.sm_state_wp,
                self.sm_wait_time_wp,
                ee_pose_wp,
                object_pose_wp,
                des_object_pose_wp,
                self.des_ee_pose_wp,
                self.des_gripper_state_wp,
                self.offset_wp,
                self.position_threshold,
            ],
            device=self.device,
        )

        # convert to torch
        return torch.cat([self.des_ee_pose, self.des_gripper_state.unsqueeze(-1)], dim=-1)


def add_overhead_camera(env_cfg: LiftEnvCfg, width: int, height: int) -> None:
    """Attach a fixed overhead RGB-D camera to the scene, looking down at the table.

    The camera is used to visually locate the cube (see :func:`estimate_cube_position_from_camera`)
    instead of relying on the simulator's ground-truth object pose. The vertical aperture is derived
    from the horizontal one and the image aspect ratio, so changing the resolution while keeping the
    4:3 aspect ratio only changes the sampling density, not the field of view.

    Args:
        env_cfg: The lift environment config to extend.
        width: Image width [px].
        height: Image height [px].
    """
    env_cfg.scene.overhead_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/CameraOverhead",
        update_period=0.0,
        height=height,
        width=width,
        data_types=["rgb", "distance_to_image_plane"],
        # this demo aims the camera from the script, and back-projection needs the extrinsics that the
        # image was actually rendered with. without this, `camera.data.pos_w` keeps reporting the pose
        # the camera was spawned with until the next sensor reset, and the point cloud is meaningless.
        update_latest_camera_pose=True,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.01, 5.0)
        ),
    )


def colorize_cube_for_vision(env_cfg: LiftEnvCfg) -> None:
    """Give the cube a distinct, saturated color so it can be segmented from the RGB image.

    The material is kept rough and non-metallic so that specular highlights do not wash the hue
    out to white, which would shrink the segmented region.
    """
    env_cfg.scene.object.spawn.visual_material = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.02, 0.85, 0.02), roughness=0.9, metallic=0.0
    )


def hide_debug_markers(env_cfg: LiftEnvCfg) -> None:
    """Turn off the frame-axes gizmos so the viewer shows only the robot, the table and the cube.

    The lift task visualizes the ``object_pose`` command with two RGB axis triads, one for the goal
    pose and one for the current hand pose. They clutter the view without adding anything here, since
    this demo is about what the camera sees. Hiding them does not change the command itself.
    """
    env_cfg.commands.object_pose.debug_vis = False
    env_cfg.scene.ee_frame.debug_vis = False


def patch_franka_usd_path(env_cfg: LiftEnvCfg) -> None:
    """Point the Franka robot spawn at the relocated USD asset.

    ``FRANKA_PANDA_CFG`` (used by the lift task's Franka config) still points at
    ``Robots/FrankaEmika/panda_instanceable.usd``, but the Isaac 6.0 content server moved
    that file to a ``Legacy/`` subfolder, so the default path 404s. Reroute the local env
    config to the relocated file rather than editing the shared asset config.
    """
    usd_path = env_cfg.scene.robot.spawn.usd_path
    marker = "Robots/FrankaEmika/panda_instanceable.usd"
    if usd_path.endswith(marker):
        legacy_path = "Robots/FrankaEmika/Legacy/panda_instanceable.usd"
        env_cfg.scene.robot.spawn.usd_path = usd_path.replace(marker, legacy_path)


def segment_green(rgb: torch.Tensor, min_chroma: float = 0.5, min_intensity: float = 0.05) -> torch.Tensor:
    """Segment the green cube from an RGB image using normalized chromaticity.

    Comparing the green channel against the pixel's total intensity rather than against a fixed
    threshold makes the mask invariant to shading: a cube pixel in the gripper's shadow keeps the
    same chromaticity as a brightly lit one, while a neutral table pixel sits at ~1/3 regardless of
    how bright it is.

    Args:
        rgb: Image with channel values in [0, 1], shape (H, W, 3).
        min_chroma: Minimum green fraction of the total intensity for a pixel to count as cube.
        min_intensity: Minimum total intensity, to reject near-black pixels whose ratio is noise.

    Returns:
        Boolean mask of cube pixels, shape (H, W).
    """
    total = rgb.sum(dim=-1)
    green_fraction = rgb[..., 1] / total.clamp(min=1e-6)
    return (green_fraction > min_chroma) & (total > min_intensity)


def estimate_cube_position_from_camera(
    camera: Camera,
    min_pixel_fraction: float = 2.5e-4,
    top_face_tolerance: float = 0.005,
    cube_height_range: tuple[float, float] = (0.01, 0.2),
) -> torch.Tensor | None:
    """Estimate the cube's center (world frame) from the overhead camera's RGB-D image.

    The cube is segmented from the RGB image by its distinct green chromaticity, and the whole depth
    image is back-projected to world-frame points through the camera's intrinsic and extrinsic
    parameters. Two corrections turn the visible surface into a center estimate:

    * Only points on the cube's **top face** are kept. Anti-aliased pixels along the silhouette blend
      the cube's color with the *table's* depth, and a cube that is not exactly below the camera also
      exposes its side faces; averaging all segmented points would pull the centroid sideways and down.
    * The center height is taken halfway between the observed top face and the **supporting surface**
      measured in an annulus of non-cube pixels around it. Since the cube rests on the table, this
      recovers the center without hard-coding the cube's size. Averaging the raw points instead would
      place the estimate on the top face, roughly half a cube too high, which makes the gripper close
      above the cube instead of around it.

    Args:
        camera: The overhead camera sensor.
        min_pixel_fraction: Fraction of the image that must be segmented for the estimate to be trusted.
            Expressed as a fraction so the check is independent of the camera resolution.
        top_face_tolerance: Thickness [m] of the slab below the top of the cube that is treated as
            belonging to the top face.
        cube_height_range: Plausible range (min, max) [m] for the height of the segmented blob above
            its supporting surface. Acts as a sanity gate: a rendered frame that does not match the
            camera's current extrinsics produces a blob of implausible height, which is rejected here
            rather than being fed to the state machine.

    Returns:
        The estimated cube center in the world frame [m], shape (3,), or ``None`` when the cube is not
        visible, or not plausibly a cube, in the current frame.
    """
    output = camera.data.output
    if output is None:
        return None

    rgb = output["rgb"].torch[0, ..., :3]
    # the backend hands back either uint8 in [0, 255] or float in [0, 1]
    rgb = rgb.float() / 255.0 if not rgb.is_floating_point() else rgb.float()
    depth_img = output["distance_to_image_plane"].torch[0]
    if depth_img.dim() == 3 and depth_img.shape[-1] == 1:
        # some camera implementations keep a trailing singleton channel dim, others don't
        depth_img = depth_img[..., 0]

    mask = segment_green(rgb)
    min_pixels = max(25, int(min_pixel_fraction * mask.numel()))
    if int(mask.sum().item()) < min_pixels:
        return None

    # back-project the whole depth image into world-frame points
    points_w = create_pointcloud_from_depth(
        intrinsic_matrix=camera.data.intrinsic_matrices.torch[0],
        depth=depth_img,
        keep_invalid=True,
        position=camera.data.pos_w.torch[0],
        orientation=camera.data.quat_w_ros.torch[0],
        device=depth_img.device,
    )
    # `unproject_depth` flattens the (H, W) image with the width index varying slowest, i.e. the point
    # at index `u * H + v` corresponds to pixel (v, u), so transpose the mask to match that ordering.
    mask_flat = mask.transpose(0, 1).reshape(-1)
    finite = torch.isfinite(points_w).all(dim=-1)

    cube_points = points_w[mask_flat & finite]
    if cube_points.shape[0] < min_pixels:
        return None

    # isolate the top face and use its centroid for the horizontal position
    top_of_cube = torch.quantile(cube_points[:, 2], 0.9)
    top_face = cube_points[cube_points[:, 2] >= top_of_cube - top_face_tolerance]
    center_xy = top_face[:, :2].mean(dim=0)
    top_z = top_face[:, 2].median()

    # measure the surface the cube stands on, in a ring of non-cube pixels just outside its footprint
    support_points = points_w[(~mask_flat) & finite]
    radial_distance = torch.linalg.norm(support_points[:, :2] - center_xy, dim=-1)
    support_z = support_points[(radial_distance > 0.05) & (radial_distance < 0.15), 2]
    if support_z.numel() < min_pixels:
        return None

    cube_height = top_z - support_z.median()
    if not cube_height_range[0] < cube_height.item() < cube_height_range[1]:
        return None

    return torch.cat([center_xy, (top_z - 0.5 * cube_height).reshape(1)])


def main():
    # parse configuration
    env_cfg: LiftEnvCfg = parse_env_cfg(
        "IsaacContrib-Lift-Cube-Franka-IK-Abs",
        device=args_cli.device,
        num_envs=1,
    )
    # attach the overhead camera and make the cube visually distinct for the vision pipeline
    add_overhead_camera(env_cfg, args_cli.camera_width, args_cli.camera_height)
    colorize_cube_for_vision(env_cfg)
    hide_debug_markers(env_cfg)
    patch_franka_usd_path(env_cfg)
    # the default 5 s episode cuts the lift off before it finishes; give the sequence room to complete
    env_cfg.episode_length_s = args_cli.episode_length_s

    # create environment
    env = gym.make("IsaacContrib-Lift-Cube-Franka-IK-Abs", cfg=env_cfg)
    # reset environment at start
    env.reset()

    # aim the overhead camera at the table, in front of the robot. the camera sits high enough that the
    # cube's full randomization range (x +/- 0.1 m, y +/- 0.25 m) stays inside the frame with margin.
    overhead_cam: Camera = env.unwrapped.scene["overhead_cam"]
    env_origins = env.unwrapped.scene.env_origins
    device = env.unwrapped.device
    eye = env_origins + torch.tensor(CAMERA_EYE, device=device)
    look_at = env_origins + torch.tensor(CAMERA_LOOK_AT, device=device)
    overhead_cam.set_world_poses_from_view(eye, look_at)

    def hold_pose_action() -> torch.Tensor:
        """Build an absolute-IK action that keeps the end-effector where it currently is."""
        ee_data = env.unwrapped.scene["ee_frame"].data
        position = ee_data.target_pos_w.torch[..., 0, :] - env.unwrapped.scene.env_origins
        orientation = ee_data.target_quat_w.torch[..., 0, :]
        gripper = torch.full((env.unwrapped.num_envs, 1), GripperState.OPEN, device=device)
        return torch.cat([position, orientation, gripper], dim=-1)

    # let the renderer catch up with the camera pose we just set: for the first few frames the returned
    # image is still drawn from the camera's previous pose, and back-projecting it with the new
    # extrinsics would yield a point cloud that does not correspond to any real geometry.
    with torch.inference_mode():
        for _ in range(12):
            env.step(hold_pose_action())

    # create action buffers (position + quaternion + gripper)
    actions = hold_pose_action()
    # desired object orientation (we only do position control of object)
    desired_orientation = torch.zeros((env.unwrapped.num_envs, 4), device=device)
    desired_orientation[:, 1] = 1.0
    # create state machine
    pick_sm = PickAndLiftSm(
        env_cfg.sim.dt * env_cfg.decimation,
        env.unwrapped.num_envs,
        device,
        position_threshold=args_cli.position_threshold,
    )

    # ground truth is read for reporting only -- the state machine is driven purely by the camera
    object_data: RigidObjectData = env.unwrapped.scene["object"].data
    nominal_position_w = env_origins[0] + torch.tensor(NOMINAL_CUBE_POSITION, device=device)

    # vision state: samples are gathered while the arm is still at rest and the cube is unoccluded,
    # then the running median is frozen for the rest of the pick
    vision_samples: list[torch.Tensor] = []
    cube_position_w = nominal_position_w
    frames_since_reset = 0
    latch_reported = False
    # skip the first frames after a reset: the renderer lags the physics, so the image may still show
    # the cube at its pre-reset pose
    vision_warmup_frames = 10
    min_vision_samples = 5

    print("[INFO] シミュレーション開始。俯瞰カメラの映像からキューブ位置を推定してピッキングします...")
    print(f"[INFO] カメラ解像度: {args_cli.camera_width}x{args_cli.camera_height} px")

    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # step environment
            dones = env.step(actions)[-2]
            frames_since_reset += 1

            # observations
            # -- end-effector frame
            ee_frame_sensor = env.unwrapped.scene["ee_frame"]
            tcp_rest_position = (
                ee_frame_sensor.data.target_pos_w.torch[..., 0, :].clone() - env.unwrapped.scene.env_origins
            )
            tcp_rest_orientation = ee_frame_sensor.data.target_quat_w.torch[..., 0, :].clone()
            # -- object frame, estimated from the overhead camera instead of ground truth. the estimate is
            #    only refreshed while the arm is at rest: once it descends, the gripper occludes the cube
            #    and any new estimate would be worse than the one already latched.
            if pick_sm.sm_state[0].item() == PickSmState.REST:
                if frames_since_reset > vision_warmup_frames:
                    sample = estimate_cube_position_from_camera(overhead_cam)
                    if sample is not None:
                        vision_samples.append(sample)
                if len(vision_samples) < min_vision_samples:
                    # do not start reaching for a cube we have not seen yet
                    pick_sm.hold_at_rest()
                else:
                    cube_position_w = torch.stack(vision_samples).median(dim=0).values
            elif not latch_reported:
                error = torch.linalg.norm(cube_position_w - object_data.root_pos_w.torch[0])
                print(
                    f"[VISION] 推定位置(env-local): {(cube_position_w - env_origins[0]).cpu().numpy()}"
                    f" / 真値との誤差: {error.item() * 1000.0:.1f} mm"
                    f" ({len(vision_samples)} フレームの中央値)"
                )
                latch_reported = True

            object_position = (cube_position_w - env_origins[0]).unsqueeze(0)
            # -- target object frame
            desired_position = env.unwrapped.command_manager.get_command("object_pose")[..., :3]

            # advance state machine
            actions = pick_sm.compute(
                torch.cat([tcp_rest_position, tcp_rest_orientation], dim=-1),
                torch.cat([object_position, desired_orientation], dim=-1),
                torch.cat([desired_position, desired_orientation], dim=-1),
            )

            # reset state machine
            if dones.any():
                pick_sm.reset_idx(dones.nonzero(as_tuple=False).squeeze(-1))
                # start a fresh observation of the newly randomized cube pose
                vision_samples.clear()
                cube_position_w = nominal_position_w
                frames_since_reset = 0
                latch_reported = False

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
