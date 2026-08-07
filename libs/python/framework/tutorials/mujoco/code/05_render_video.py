#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 05_render_video.py — 离屏渲染与视频录制
 ================================================================================
 本节目标:
   1. 掌握 mujoco.Renderer 离屏渲染 (无需显示器，服务器/容器也能用)
   2. 学习相机 (camera) 设置: 自由视角 / 跟随视角 / 固定视角
   3. 把仿真过程录制成 mp4 视频 (用 OpenCV 编码，无需额外 ffmpeg)

 渲染管线
 --------
   Renderer(model, height, width)   → 创建离屏渲染器
   renderer.update_scene(data)      → 把当前仿真状态同步到渲染器
   renderer.render()                → 返回 (H, W, 3) uint8 的 RGB 图像
   renderer.add_overlay(...)        → (可选) 叠加文字

 运行方式:
   python 05_render_video.py
================================================================================
"""

import time
from pathlib import Path

import mujoco
import numpy as np

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
OUT_VIDEO = Path(__file__).parent / "swingup_video.mp4"

# ── 1. 加载单摆模型 ──────────────────────────────────────────────────────────
model = mujoco.MjModel.from_xml_path(str(MODEL_DIR / "pendulum.xml"))
data = mujoco.MjData(model)

# 物理常量 (复用 04 节的方法)
I = float(model.body_inertia[1, 1])
m = float(model.body_subtreemass[1])
L = float(np.linalg.norm(model.body_ipos[1]))
g = 9.81
E_star = m * g * L

KP, KPP, KDD = 8.0, 15.0, 2.0

def control(d):
    """能量整形起摆 + PD 锁定 (与 04 节一致, θ=0 为直立)"""
    theta, omega = d.qpos[0], d.qvel[0]
    theta_n = np.angle(np.exp(1j * theta))
    E = 0.5 * I * omega**2 + m * g * L * np.cos(theta)
    if abs(theta_n) < 0.4 and abs(E - E_star) < 0.25 * E_star:
        return -KPP * theta_n - KDD * omega
    return KP * (E_star - E) * np.sign(omega)

# ── 2. 给模型加一个跟随相机 ─────────────────────────────────────────────────
CAMERA_XML = """
    <camera name="track" pos="0 -2.5 1.2" xyaxes="1 0 0 0 0.3 0.95" mode="trackcom"
            target="pole"/>
"""
xml = (MODEL_DIR / "pendulum.xml").read_text(encoding="utf-8")
if "<camera" not in xml:
    # 把 camera 元素插入现有 <worldbody> 的末尾
    xml = xml.replace("</worldbody>", CAMERA_XML + "</worldbody>")
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

# ── 3. 创建渲染器 (必须在模型最终确定之后创建) ───────────────────────────────
W, H, FPS = 640, 480, 30
renderer = mujoco.Renderer(model, height=H, width=W)

# ── 4. 录制视频 ──────────────────────────────────────────────────────────────
import cv2

DURATION = 8.0                       # 8 秒视频
n_frames = int(DURATION * FPS)
steps_per_frame = max(1, int(1.0 / FPS / model.opt.timestep))

video = cv2.VideoWriter(
    str(OUT_VIDEO),
    cv2.VideoWriter_fourcc(*"mp4v"),
    FPS,
    (W, H),
)

print(f"🎬 开始录制 {DURATION}s 视频 (共 {n_frames} 帧, "
      f"每帧 {steps_per_frame} 物理步) ...")
t0 = time.time()

for f in range(n_frames):
    # 推进若干物理步
    for _ in range(steps_per_frame):
        data.ctrl[0] = control(data)
        mujoco.mj_step(model, data)

    # 渲染一帧
    renderer.update_scene(data, camera="track")
    frame = renderer.render()

    # OpenCV 需要 BGR 顺序
    video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

video.release()
renderer.close()
dt = time.time() - t0
print(f"✅ 视频已保存: {OUT_VIDEO}  ({dt:.1f}s 内完成 {n_frames} 帧渲染)")

print("\n💡 渲染技巧:")
print("   - 服务器环境没有显示器也能渲染 (EGL/OSMesa 离屏后端)")
print("   - update_scene(data, camera=...) 可切换相机")
print("   - 视频可用于: 训练可视化 / 论文插图 / Sim2Real 迁移对比")
