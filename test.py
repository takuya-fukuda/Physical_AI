import numpy as np
import omni.kit.app  # アプリの実行状態を監視するため
from omni.isaac.core import SimulationContext
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.quadruped.robots import UnitreeA1
import omni.isaac.core.utils.stage as stage_utils

# 1. シミュレーション初期化
sim_context = SimulationContext()

# 2. ステージの作成（床、ライト、Cube、ロボット）
create_prim("/World/Ground", prim_type="GroundPlane")
create_prim("/World/Light", prim_type="DistantLight")
create_prim(
    "/World/Cube", 
    prim_type="Cube", 
    position=np.array([0.0, 0.0, 0.5]), 
    scale=np.array([1.0, 1.0, 1.0])
)

# ロボットを配置
my_a1 = UnitreeA1(prim_path="/World/a1", name="my_a1", position=np.array([2.0, 0.0, 0.0]))

# カメラの位置をロボットの近くに固定
stage_utils.set_camera_view(eye=np.array([4.0, 4.0, 2.0]), target=np.array([0.0, 0.0, 0.5]))

# 物理リセット
sim_context.reset()

# 3. 制御ループ
forward_velocity = 0.3   # 前進速度 (m/s)
turning_velocity = 0.4   # 旋回速度 (rad/s)

print("環境構築成功！シミュレーションを開始します。")

# アプリが起動している間ループを回す（Kit環境の標準的な書き方）
app_run = omni.kit.app.get_app_interface()
while app_run.is_running():
    if sim_context.is_playing():
        # ロボットに円軌道の速度コマンドを送信
        my_a1.set_world_velocity(np.array([forward_velocity, 0.0, turning_velocity]))
    
    # 物理演算と画面描画を1ステップ進める
    sim_context.step(render=True)