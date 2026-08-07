#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 07_scripted_pick_place.py — 项目实战 (二): 脚本化抓取搬运 Pick-and-Place
 ================================================================================
 基于 06 的 PickPlaceEnv，用"状态机 + 逆运动学 + 位置伺服"完成:
   1. 移动到方块上方      (approach)
   2. 下降到抓取高度      (descend)
   3. 闭合夹爪抓取        (grasp)
   4. 抬升方块            (lift)
   5. 平移到目标上方      (transfer)
   6. 下降到目标          (place)
   7. 张开夹爪释放        (release)
   8. 退回安全位          (retreat)

 并实时离屏渲染成 mp4 视频 (这是调试/展示抓取最直观的方式)。

 运行方式:
   python 07_scripted_pick_place.py
================================================================================
"""

import time
from pathlib import Path
import importlib.util

import cv2
import numpy as np
import mujoco

# 数字开头的模块名不能直接 import, 用 importlib 加载
def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

make_scene = _load_module("make_scene", "make_pick_place_scene.py")
pp_env = _load_module("pp_env", "06_pick_place_env.py")
PickPlaceEnv = pp_env.PickPlaceEnv
FINGER_OPEN, FINGER_CLOSE = pp_env.FINGER_OPEN, pp_env.FINGER_CLOSE

OUT_VIDEO = Path(__file__).parent / "pick_place_demo.mp4"
FPS, W, H = 30, 640, 480

# ── 1. 创建环境 ──────────────────────────────────────────────────────────────
env = PickPlaceEnv(seed=0)
obs = env.reset()
block_pos = obs["block_pos"]
target_pos = obs["target_pos"]
print(f"🎯 方块位置 = {np.round(block_pos, 3)}")
print(f"🎯 目标位置 = {np.round(target_pos, 3)}")

# 抓取高度: 腕部在方块中心上方 0.04, 球形指尖落在方块中平面
GRASP_Z = block_pos[2] + pp_env.GRASP_OFFSET
LIFT_Z = 0.78                # 抬升高度 (须高于桌面与方块, 肩高 0.85)
APPROACH_Z = 0.75            # 接近高度 (方块上方)
HOME = (0.15, 0.0, 0.80)     # 安全回缩位

# ── 2. 状态机定义 ────────────────────────────────────────────────────────────
# 每个阶段: (终点位置, 手指开度, 等待伺服收敛步数)
STAGES = [
    ("approach ", (block_pos[0], block_pos[1], APPROACH_Z), FINGER_OPEN, 250),
    ("descend  ", (block_pos[0], block_pos[1], GRASP_Z), FINGER_OPEN, 250),
    ("grasp    ", (block_pos[0], block_pos[1], GRASP_Z), FINGER_CLOSE, 120),
    ("lift     ", (block_pos[0], block_pos[1], LIFT_Z), FINGER_CLOSE, 250),
    ("transfer ", (target_pos[0], target_pos[1], LIFT_Z), FINGER_CLOSE, 300),
    ("place    ", (target_pos[0], target_pos[1], GRASP_Z), FINGER_CLOSE, 250),
    ("release  ", (target_pos[0], target_pos[1], GRASP_Z), FINGER_OPEN, 120),
    ("retreat  ", HOME, FINGER_OPEN, 250),
]

# ── 3. 录制视频并执行状态机 ──────────────────────────────────────────────────
video = cv2.VideoWriter(str(OUT_VIDEO), cv2.VideoWriter_fourcc(*"mp4v"),
                        FPS, (W, H))

def record(steps):
    """推进若干步并录帧 (虚拟抓取时同步方块)"""
    for _ in range(steps):
        mujoco.mj_step(env.model, env.data)
        env._sync_block()                     # 虚拟抓取时让方块跟随夹爪
        frame = env.render(camera="side")
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

print("🎬 开始执行抓取任务 ...")
t0 = time.time()

for i, (name, pos, fingers, hold) in enumerate(STAGES):
    # 计算 IK 并发送位置伺服指令
    env.ik_wrist(np.array(pos, dtype=np.float64))
    for aid, qadr in zip(env.arm_act_ids, env.arm_qadr):
        env.data.ctrl[aid] = float(env.data.qpos[qadr])
    env._set_fingers(fingers)

    # 关键节点: 闭合夹爪后 = 抓取成功, 建立虚拟抓取; 释放前解除
    if name.strip() == "grasp":
        record(hold)
        env.attach_block()                    # ── 虚拟抓取: 方块"焊"在夹爪上
        print("  🤝 已建立虚拟抓取 (方块将精确跟随夹爪)")
        continue
    elif name.strip() == "release":
        env.release_block(drop=True)          # ── 解除虚拟抓取, 方块下落放置
        record(hold)
        print("  ✋ 已释放方块")
        continue

    record(hold)
    ee = env.data.body("gripper_base").xpos
    block = env.data.body("block").xpos
    print(f"  [{name}] 夹爪={np.round(ee, 3)}  方块={np.round(block, 3)}")

video.release()
env.close()
dt = time.time() - t0

# ── 4. 结果判定 ──────────────────────────────────────────────────────────────
final_block = env.data.body("block").xpos
dist = np.linalg.norm(final_block[:2] - target_pos[:2])
print(f"\n✅ 视频已保存: {OUT_VIDEO}  ({dt:.1f}s)")
print(f"📦 最终方块位置 = {np.round(final_block, 3)}")
print(f"📏 距目标水平距离 = {dist * 100:.1f} cm")
if dist < 0.05:
    print("🎉 任务成功: 方块已放到目标点！")
else:
    print("⚠️ 任务未完全成功, 请检查抓取是否稳固 (可调 FINGER_CLOSE / 抬升高度)")
