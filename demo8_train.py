# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab 3.0+: demo8 の学習スクリプト — ずれたパレットを突いて枠内に戻す方策を PPO で学習する。

環境の中身 (観測・行動・報酬・ランダム化) は ``demo8_env.py`` の docstring を参照。
ここでは環境を多数並列に立ち上げ、rsl_rl の PPO で学習し、重み (チェックポイント) と
推論用に書き出した方策 (TorchScript) を保存する。

保存先::

    logs/rsl_rl/so101_pallet_nudge/<日時>[_<run_name>]/
        model_<iter>.pt          # 学習途中のチェックポイント (--resume で再開できる)
        exported/policy.pt       # demo8.py が読み込む推論用の方策 (TorchScript, 観測正規化込み)
    logs/rsl_rl/so101_pallet_nudge/latest.txt   # 直近の run のパス (demo8.py の既定の読み込み先)

学習の進み具合は TensorBoard で見られる (``Metrics/success_rate``、重いパレットだけの
``Metrics/success_heavy`` 等)::

    tensorboard --logdir logs/rsl_rl/so101_pallet_nudge

.. code-block:: bash

    # 既定 (2048 環境, 150 イテレーション) で学習。RTX 5060 Ti で 1 イテレーション 約 80 秒
    isaaclab.bat -p demo8_train.py
    ./isaaclab.sh -p demo8_train.py

    # まず動作確認だけ (少ない環境・数イテレーション)
    isaaclab.bat -p demo8_train.py --num_envs 32 --max_iterations 3

    # 学習の様子を画面で見る (遅くなるので環境数は少なめに)
    isaaclab.bat -p demo8_train.py --num_envs 16 --viz kit

    # 直近の run から再開して追加で学習
    isaaclab.bat -p demo8_train.py --resume --max_iterations 100
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train the SO-101 pallet-nudging policy with PPO (rsl_rl).")
parser.add_argument("--num_envs", type=int, default=2048, help="Number of parallel environments.")
parser.add_argument("--max_iterations", type=int, default=None, help="PPO iterations (default: runner config).")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--run_name", type=str, default="", help="Suffix appended to the run directory name.")
parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint of the latest run.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# training runs headless unless a visualizer is requested explicitly
parser.set_defaults(visualizer=["none"])
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import glob
import importlib.metadata as metadata
import os
import time
from datetime import datetime

import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

from demo8_env import DECIMATION, PalletNudgeEnv, PalletNudgeEnvCfg, PalletNudgePPORunnerCfg

LOG_ROOT = os.path.abspath(os.path.join("logs", "rsl_rl", PalletNudgePPORunnerCfg().experiment_name))
"""Directory all runs of this experiment are written under."""


def find_latest_checkpoint(log_root: str) -> str | None:
    """Return the newest ``model_*.pt`` of the newest run under ``log_root``, or ``None``."""
    runs = sorted(d for d in glob.glob(os.path.join(log_root, "*")) if os.path.isdir(d))
    for run in reversed(runs):
        models = glob.glob(os.path.join(run, "model_*.pt"))
        if models:
            return max(models, key=lambda p: int(os.path.basename(p)[6:-3]))
    return None


def main() -> None:
    """Build the vectorised environment, train PPO, save and export the policy."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # --- configs ---------------------------------------------------------------------------------
    env_cfg = PalletNudgeEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.seed = args_cli.seed

    agent_cfg = PalletNudgePPORunnerCfg()
    agent_cfg.seed = args_cli.seed
    agent_cfg.device = env_cfg.sim.device
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    agent_cfg.run_name = args_cli.run_name
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    resume_path = find_latest_checkpoint(LOG_ROOT) if args_cli.resume else None
    if args_cli.resume and resume_path is None:
        print("[WARN] --resume が指定されましたが、再開できるチェックポイントが見つかりません。最初から学習します。")

    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(LOG_ROOT, log_dir)
    os.makedirs(log_dir, exist_ok=True)
    env_cfg.log_dir = log_dir

    # --- environment ------------------------------------------------------------------------------
    env = PalletNudgeEnv(env_cfg)
    push_s = DECIMATION * env_cfg.sim.dt
    samples_per_iter = agent_cfg.num_steps_per_env * env.num_envs
    print("[INFO] SO-101 パレット突き戻しタスクの学習を開始します")
    print(f"[INFO] 並列環境 {env.num_envs} / 1 ステップ = 1 回の押し ({push_s:.2f} s 相当の物理)")
    print(f"[INFO] 1 イテレーション = {samples_per_iter} 回の押し, 合計 {agent_cfg.max_iterations} イテレーション")
    print(f"[INFO] ログ: {log_dir}")

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    if resume_path is not None:
        print(f"[INFO] チェックポイントから再開します: {resume_path}")
        runner.load(resume_path)

    # --- train ------------------------------------------------------------------------------------
    start = time.time()
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    elapsed = time.time() - start
    print(f"Training time: {elapsed:.1f} seconds")

    # --- save & export ----------------------------------------------------------------------------
    final_path = os.path.join(log_dir, f"model_{runner.current_learning_iteration}.pt")
    if not os.path.exists(final_path):
        runner.save(final_path)
    export_dir = os.path.join(log_dir, "exported")
    runner.export_policy_to_jit(path=export_dir, filename="policy.pt")
    with open(os.path.join(LOG_ROOT, "latest.txt"), "w", encoding="utf-8") as handle:
        handle.write(log_dir + "\n")

    print("")
    print("=" * 78)
    print(" 学習完了")
    print("=" * 78)
    print(f" 所要時間       : {elapsed / 60.0:.1f} 分")
    print(f" チェックポイント: {final_path}")
    print(f" 推論用の方策   : {os.path.join(export_dir, 'policy.pt')}")
    print(" 成功率などの推移: tensorboard --logdir " + LOG_ROOT)
    print("=" * 78)

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
