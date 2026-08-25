"""Isaac Lab 6.0+: 環境マネージャーを使って学習済み強化学習モデル(rsl_rl)を適用し、指定座標を順に経由して(10,10)までたどり着くスクリプト(エラー・転倒対策済)"""
import argparse
import os
import torch
from isaaclab.app import AppLauncher

# 1. コマンドライン引数の設定とAppLauncherの起動
parser = argparse.ArgumentParser(description="Run ANYmal-C passing waypoints straight to (10,10).")
parser.add_argument(
    "--model_path",
    type=str,
    default="logs/rsl_rl/anymal_c_rough/2026-07-18_16-38-49/exported/policy.pt",
    help="Path to the serialized policy.pt"
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = False
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- これ以降にインポートを書く ---
import gymnasium as gym
import warp as wp
import isaaclab_tasks
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import euler_xyz_from_quat

def main():
    # 2. タスク名（Gym環境ID）の設定
    task_name = "IsaacContrib-Velocity-Rough-AnymalC"

    print(f"[INFO] タスク設定を読み込んでいます: {task_name}")
    try:
        env_cfg = parse_env_cfg(task_name)
    except Exception as e:
        print(f"[WARN] {task_name} の取得に失敗しました: {e}")
        return

    # 🌟【重要対策1】エピソードの制限時間（デフォルト20秒）を実質無限に延長し、タイムアウトを完全に防ぐ
    if hasattr(env_cfg, "episode_length_s"):
        env_cfg.episode_length_s = 10000.0

    # 地形設定を【完全な平坦（Plane）】に変更
    if hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "terrain"):
        env_cfg.scene.terrain.terrain_type = "plane"

    # 不要な地形カリキュラム設定を消去
    if hasattr(env_cfg, "curriculum") and hasattr(env_cfg.curriculum, "terrain_levels"):
        delattr(env_cfg.curriculum, "terrain_levels")

    # コマンドの自動更新を無効化
    if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "base_velocity"):
        env_cfg.commands.base_velocity.resampling_time_range = (10000.0, 10000.0)

    # 環境数を1にオーバーライド
    env_cfg.scene.num_envs = 1

    # Fabricを有効化 / ビューワー固定
    if hasattr(env_cfg, "sim"):
        env_cfg.sim.use_fabric = True
    if hasattr(env_cfg, "viewer"):
        env_cfg.viewer.env_index = 0

    # 🚧【障害物コーンの配置】エラー回避のため physics_cfg は除外し、目印（Visual）のみとして配置
    cone_positions = [(3.5, 3.5, 0.3), (7.0, 7.0, 0.3)]

    if hasattr(env_cfg, "scene"):
        for i, pos in enumerate(cone_positions):
            setattr(env_cfg.scene, f"cone_{i}", AssetBaseCfg(
                prim_path=f"{{ENV_REGEX_NS}}/cone_{i}",
                spawn=sim_utils.ConeCfg(
                    radius=0.25,
                    height=0.6,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.3, 0.0)), # オレンジ色
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
            ))
            print(f"[INFO] 視覚用の障害物コーンを追加しました: 位置 {pos[:2]}")

    print("[INFO] クラス設定を適用してGym環境を生成中...")
    base_env = gym.make(task_name, cfg=env_cfg)
    env = RslRlVecEnvWrapper(base_env)

    # 3. 学習済みモデル（Policy）の読み込み
    if not os.path.exists(args_cli.model_path):
        raise FileNotFoundError(f"モデルファイルが見つかりません: {args_cli.model_path}")

    print(f"[INFO] モデルを読み込んでいます: {args_cli.model_path}")
    policy = torch.jit.load(args_cli.model_path, map_location=env.unwrapped.device)
    policy.eval()

    # 🎯【シンプルな座標ルート設定】
    # 直線 (0,0) -> (10,10) 上のコーン (3.5, 3.5), (7.0, 7.0) を綺麗に避けるための経由地を設定
    waypoints = torch.tensor([
        [2.5, 4.5],    # 経由地1: 1つ目のコーンの左上をすり抜ける
        [5.0, 5.0],    # 経由地2: 中間地点に戻る
        [8.0, 6.0],    # 経由地3: 2つ目のコーンの右下をすり抜ける
        [10.0, 10.0]   # 最終ゴール！
    ], device=env.unwrapped.device)

    waypoint_index = 0
    arrived = False

    # 環境のリセット
    obs, _ = env.reset()
    print("[INFO] シミュレーション開始。ウェイポイント経由で安全に目標へ向かいます...")

    cmd_term = env.unwrapped.command_manager._terms["base_velocity"]
    env_ids = torch.tensor([0], device=env.unwrapped.device)

    # コマンド平滑化用のバッファ
    smoothed_v_x = 0.0
    smoothed_v_z = 0.0
    filter_alpha = 0.08
    
    print(simulation_app.is_running())

    # 4. メインループ
    while simulation_app.is_running():

        robot = env.unwrapped.scene["robot"]

        root_pos_tensor = wp.to_torch(robot.data.root_pos_w)
        root_quat_tensor = wp.to_torch(robot.data.root_quat_w)

        current_pos = root_pos_tensor[0][:2] # XY座標
        _, _, root_yaw = euler_xyz_from_quat(root_quat_tensor)
        current_yaw = root_yaw[0]

        # 現在ターゲットにしているウェイポイントの取得
        target_pos = waypoints[waypoint_index]
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        distance = torch.sqrt(dx**2 + dy**2)

        # 判定: 現在の経由地に 0.6m 以内に近づいたら次の地点へ
        if not arrived and distance < 0.6:
            if waypoint_index < len(waypoints) - 1:
                waypoint_index += 1
                print(f"\n[INFO] 🚩 経由地を通過！ 次の目標地点 {waypoint_index}: ({waypoints[waypoint_index][0]:.1f}, {waypoints[waypoint_index][1]:.1f}) へ向かいます。")
            else:
                # 最後の地点（ゴール）に達したら停止
                print(f"\n[INFO] 🎉 ゴール地点 ({target_pos[0]:.1f}, {target_pos[1]:.1f}) に到着しました！安全にその場で停止します。")
                arrived = True

        if arrived:
            # ゴールに到達した後は停止
            target_lin_vel_x = 0.0
            target_ang_vel_z = 0.0
            yaw_error = 0.0
        else:
            # ターゲット方向への追従
            target_yaw = torch.atan2(dy, dx)
            yaw_error = target_yaw - current_yaw
            yaw_error = torch.atan2(torch.sin(yaw_error), torch.cos(yaw_error)) # [-pi, pi]正規化

            # 正面を向いている度合いに応じて前進
            forward_factor = torch.clamp(torch.cos(yaw_error), min=0.0)
            target_lin_vel_x = 0.45 * forward_factor

            # 旋回速度（転倒防止のためマイルドに）
            target_ang_vel_z = 1.2 * yaw_error
            target_ang_vel_z = torch.clamp(target_ang_vel_z, min=-0.45, max=0.45)

        raw_v_x = float(target_lin_vel_x.item()) if hasattr(target_lin_vel_x, "item") else float(target_lin_vel_x)
        raw_v_z = float(target_ang_vel_z.item()) if hasattr(target_ang_vel_z, "item") else float(target_ang_vel_z)

        # 指数移動平均フィルターで滑らかに出力を補正
        smoothed_v_x = filter_alpha * raw_v_x + (1.0 - filter_alpha) * smoothed_v_x
        smoothed_v_z = filter_alpha * raw_v_z + (1.0 - filter_alpha) * smoothed_v_z

        # マネージャーに値を注入
        cmd_term.cfg.ranges.lin_vel_x = (smoothed_v_x, smoothed_v_x)
        cmd_term.cfg.ranges.lin_vel_y = (0.0, 0.0)
        cmd_term.cfg.ranges.ang_vel_z = (smoothed_v_z, smoothed_v_z)
        cmd_term._resample(env_ids)

        # ステータス表示
        status_str = "【停止中】" if arrived else f"【経由地 {waypoint_index} へ移動中】"
        print(f"[NAV] {status_str} 残り距離:{distance:.2f}m | 角度誤差:{yaw_error:.2f} | 前進:{smoothed_v_x:.2f} | 旋回:{smoothed_v_z:.2f}      ", end="\r")

        # AI Policyによる推論とステップ実行
        raw_obs = obs["policy"] if (hasattr(obs, "get") or isinstance(obs, dict)) else obs
        with torch.no_grad():
            actions = policy(raw_obs)
        
        obs, rewards, dones, infos = env.step(actions)

        # リセット検知時にルート進行状況も初期化
        if dones[0]:
            print("\n[WARN] ⚠️ 環境リセット検知。ルートをスタート地点からやり直します。")
            waypoint_index = 0
            arrived = False
            smoothed_v_x = 0.0
            smoothed_v_z = 0.0

        #simulation_app.update()

if __name__ == "__main__":
    main()