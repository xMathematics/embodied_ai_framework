#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 09_evaluate_cartpole.py — 项目实战 (四): 评估训练好的策略并录制视频
 ================================================================================
 用 08 节训练好的 PPO 策略控制倒立摆, 评估性能并渲染成视频。
 演示: 加载模型 → 策略推理 (无梯度) → 环境步进 → 离屏渲染录制。

 运行方式:
   python 09_evaluate_cartpole.py
================================================================================
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import mujoco

FRAMEWORK_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(FRAMEWORK_ROOT))

from src.algorithm.reinforcement.ppo import ActorCritic
from src.simulation.backend import SimBackendFactory

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
CART_XML = MODEL_DIR / "cartpole.xml"
POLICY_PATH = Path(__file__).parent / "cartpole_ppo.pt"
OUT_VIDEO = Path(__file__).parent / "cartpole_balance.mp4"
FPS, W, H = 30, 640, 480

# ── 1. 加载策略 ──────────────────────────────────────────────────────────────
# 若没有训练好的权重, 提示先运行 08
if not POLICY_PATH.exists():
    raise FileNotFoundError(
        f"未找到策略权重 {POLICY_PATH}, 请先运行: python 08_train_cartpole_ppo.py")

ac = ActorCritic(obs_dim=5, action_dim=1)
ac.load_state_dict(torch.load(POLICY_PATH, map_location="cpu", weights_only=True))
ac.eval()
print("✅ 已加载训练好的策略:", POLICY_PATH)

# ── 2. 创建原始 MuJoCo 环境 (用于渲染) ──────────────────────────────────────
model = mujoco.MjModel.from_xml_path(str(CART_XML))
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=H, width=W)

def get_obs():
    """从 mjData 构造与训练一致的观测"""
    x, th = data.qpos[0], data.qpos[1]
    xd, thd = data.qvel[0], data.qvel[1]
    return torch.tensor([x, xd, np.sin(th), np.cos(th), thd], dtype=torch.float32)

# ── 3. 用策略控制并录制 ──────────────────────────────────────────────────────
video = cv2.VideoWriter(str(OUT_VIDEO), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
EPISODES, MAX_STEPS = 3, 200
steps_per_frame = max(1, int(1.0 / FPS / model.opt.timestep))

total_steps = 0
for ep in range(EPISODES):
    mujoco.mj_resetData(model, data)
    data.qpos[1] = 0.02         # 摆杆初始微倾 (1.1°, 与训练条件接近)
    mujoco.mj_forward(model, data)
    ep_steps = 0
    for t in range(MAX_STEPS):
        with torch.no_grad():
            obs = get_obs().unsqueeze(0)
            action, _, _ = ac(obs)
        a = float(np.clip(action.item(), -1, 1)) * 15.0
        data.ctrl[0] = a / 10.0   # gear=10

        # 推进若干物理步并录帧
        for _ in range(steps_per_frame):
            mujoco.mj_step(model, data)
            renderer.update_scene(data)
            frame = renderer.render()
            video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        ep_steps += 1

        # 终止判定 (与训练一致)
        x, xd, sp, cp, thd = get_obs().numpy()
        if cp < np.cos(np.deg2rad(23)) or abs(x) > 2.0:
            break

    total_steps += ep_steps
    print(f"  📊 回合 {ep+1}: 平衡了 {ep_steps} 步 (满分 {MAX_STEPS})")

video.release()
renderer.close()
print(f"\n✅ 视频已保存: {OUT_VIDEO}")
print(f"🎉 平均每回合平衡 {total_steps / EPISODES:.0f} 步"
      f" (满分 {MAX_STEPS})" if total_steps / EPISODES > MAX_STEPS * 0.8
      else f"📈 平均每回合平衡 {total_steps / EPISODES:.0f} 步, 可继续训练")
