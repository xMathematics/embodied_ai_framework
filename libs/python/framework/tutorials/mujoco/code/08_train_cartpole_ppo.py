#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 08_train_cartpole_ppo.py — 项目实战 (三): 用框架 PPO 训练倒立摆平衡
 ================================================================================
 这是"MuJoCo 与 RL 全栈对接"的一课: 把我们前几节学的 MuJoCo 环境,
 接入本框架的完整 RL 训练栈:
   MuJoCoBackend (仿真后端) → RobotEnv (Gym 风格环境)
   → DummyVecEnv (向量化) → PPOTrainer (策略梯度训练)

 训练目标: 让小车学会"把摆杆维持在直立位置" (经典 CartPole)。
   - 观测: [小车位置 x, 速度 ẋ, sinθ, cosθ, 摆杆角速度 θ̇]
   - 动作: 小车水平推力 (连续, [-1, 1])
   - 奖励: 存活每步 +1, 摆杆倒下或小车出界则终止
   - 策略: 高斯连续策略 (框架自带 ActorCritic)

 运行方式:
   python 08_train_cartpole_ppo.py            # 训练
   python 09_evaluate_cartpole.py             # 评估并录制视频
================================================================================
"""

import sys
from pathlib import Path

import numpy as np
import torch

# 允许从框架根目录导入 src.*
# 本文件位于 tutorials/mujoco/code/, 框架根目录是上三级
FRAMEWORK_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(FRAMEWORK_ROOT))

from src.simulation.backend import RobotEnv, SimBackendFactory
from src.simulation.vec_env import DummyVecEnv, make_vec_env
from src.algorithm.reinforcement.ppo import PPOTrainer

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
CART_XML = MODEL_DIR / "cartpole.xml"
DEVICE = "cpu"


# ═══════════════════════════════════════════════════════════════════════════════
# 1. 定义 CartPole 环境 (对接框架的 RobotEnv)
# ═══════════════════════════════════════════════════════════════════════════════

class CartPoleEnv(RobotEnv):
    """倒立摆环境: 继承框架 RobotEnv, 内部用 MuJoCoBackend 驱动物理"""
    def __init__(self, config=None):
        config = config or {}
        config.setdefault("model_path", str(CART_XML))
        config.setdefault("time_step", 0.02)
        # 用框架的仿真后端工厂创建 MuJoCo 后端 (default() 含内置后端)
        backend = SimBackendFactory.default().create("mujoco", config)
        super().__init__(backend, config)
        self.obs_dim = 5
        self.action_dim = 1
        self._t = 0

    def reset(self, *, seed=None, options=None):
        self.backend.reset()
        self._t = 0
        return self._get_obs(), {}

    def _get_obs(self):
        qpos = self.backend.get_observation({})["joint_pos"].numpy()
        qvel = self.backend.get_observation({})["joint_vel"].numpy()
        x, th = qpos[0], qpos[1]
        xd, thd = qvel[0], qvel[1]
        # 用 sin/cos 表示角度, 避免无界角度累积
        state = np.array([x, xd, np.sin(th), np.cos(th), thd], dtype=np.float32)
        return {"robot_state": torch.from_numpy(state)}

    def step(self, action):
        # action: (1,) tensor ∈ [-1,1] → 推力
        a = float(action.detach().cpu().numpy().reshape(-1)[0])
        a = float(np.clip(a, -1, 1)) * 15.0          # ±15N
        ctrl = torch.tensor([a / 10.0])               # motor gear=10 → ctrl
        self.backend.step({"ctrl": ctrl})
        self._t += 1

        obs = self._get_obs()
        s = obs["robot_state"].numpy()
        x, xd, sp, cp, thd = s
        # 终止: 摆杆偏离竖直 > 23° 或小车出界
        terminated = bool(cp < np.cos(np.deg2rad(23)) or abs(x) > 2.0)
        truncated = self._t >= 200
        reward = 1.0
        return obs, reward, terminated, truncated, {"x": x, "angle_deg": np.degrees(np.arctan2(sp, cp))}


# ═══════════════════════════════════════════════════════════════════════════════
# 2. 训练主循环
# ═══════════════════════════════════════════════════════════════════════════════

def train(num_iter=120, num_steps=1024, num_envs=4, save_path="cartpole_ppo.pt"):
    # 向量化环境: 4 个并行倒立摆
    env_fn = lambda: CartPoleEnv()                 # noqa: E731
    vec_env = make_vec_env(env_fn, num_envs=num_envs, mode="dummy")

    # PPO 训练器 (观测 5 维, 动作 1 维)
    trainer = PPOTrainer(
        obs_dim=5, action_dim=1,
        lr=3e-4, update_epochs=10, batch_size=256,
        device=DEVICE,
    )

    print(f"🚀 开始训练 PPO (共 {num_iter} 轮, 每轮 {num_steps} 步 × {num_envs} 环境)")
    best = -1e9
    for it in range(num_iter):
        # 1. 收集轨迹
        stats = trainer.collect(vec_env, num_steps)
        # 2. 策略更新
        metrics = trainer.train_on_buffer()
        # 3. 定期评估
        if it % 10 == 0 or it == num_iter - 1:
            eval_stats = trainer.evaluate(vec_env, num_episodes=5)
            avg_r = eval_stats["eval_avg_reward"]
            print(f"  轮 {it:3d} | 平均回报(评估)={avg_r:7.1f} | "
                  f"策略损失={metrics['policy_loss']:.4f}")
            if avg_r > best:
                best = avg_r
                torch.save(trainer.ac.state_dict(), save_path)
                print(f"    ✅ 新最优, 已保存: {save_path}")

    vec_env.close()
    print(f"\n🏁 训练完成, 最佳平均回报 = {best:.1f} (满分 {200 * num_envs})")
    return trainer


if __name__ == "__main__":
    train()
