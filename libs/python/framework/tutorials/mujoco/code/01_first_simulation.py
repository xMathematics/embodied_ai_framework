#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 01_first_simulation.py — 你的第一个 MuJoCo 仿真
 ================================================================================
 本节目标:
   1. 加载一个 MJCF 模型 (本框架自带的 5-DoF 机械臂)
   2. 理解 mjModel 与 mjData 两个核心对象
   3. 驱动物理引擎步进 (mj_step)
   4. 读取关节状态、末端执行器位姿
   5. 用 mujoco.viewer 打开**实时可视化窗口**，观察机械臂自然下垂与夹爪开合
   6. (可选) 保存一帧离屏截图验证

 核心概念
 --------
 MuJoCo 的一切都围绕两个结构体:
   - MjModel (模型):  静态描述 —— 机器人的质量、几何、关节、执行器等"模板"
   - MjData  (数据):  动态状态 —— 当前的位置(qpos)、速度(qvel)、力等"实例"
 二者关系: 一个 MjModel 可以派生任意多个 MjData (即多个并行仿真实例)。
 这是 MuJoCo 能极快并行 (MJX) 的根本原因。

 可视化 (mujoco.viewer)
 ----------------------
   viewer = mujoco.viewer.launch_passive(model, data)  # 非阻塞打开窗口
   主循环里每步 mj_step 后调用 viewer.sync() 把新状态推送到窗口
   while viewer.is_running(): ...   # 窗口会一直保持，直到你手动关闭
   鼠标操作: 左键旋转 · 右键平移 · 滚轮缩放 · 双击聚焦物体
   无显示器的服务器/容器环境会抛异常 → 脚本自动降级为无界面运行

 运行方式:
   python 01_first_simulation.py
================================================================================
"""

import time
from pathlib import Path

import mujoco
import numpy as np

# ── 1. 加载模型 ──────────────────────────────────────────────────────────────
# 注意: 从脚本所在目录定位模型文件，保证任何位置运行都能找到
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "robot_arm.xml"

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)
print(f"✅ 模型加载成功: {MODEL_PATH.name}")
print(f"   自由度 nq = {model.nq}   (广义坐标维度)")
print(f"   自由度 nv = {model.nv}   (广义速度维度)")
print(f"   执行器 nu = {model.nu}   (控制输入维度)")

# ── 2. 检查模型结构 ──────────────────────────────────────────────────────────
print("\n── 机器人关节 ──")
for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    # qpos 与 qvel 中的起始索引
    qadr = model.jnt_qposadr[i]
    vadr = model.jnt_dofadr[i]
    print(f"  [{i}] {name:<16} qpos_idx={qadr}  qvel_idx={vadr}")

print("\n── 机器人 body ──")
for i in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    print(f"  [{i}] {name}")

# ── 3. 前向动力学: 计算初始状态下所有几何体的位姿 ────────────────────────────
# mj_forward 会根据 qpos 更新所有 body 的世界坐标 (xpos/xmat)、雅可比等
mujoco.mj_forward(model, data)

# 查询末端执行器 (夹爪基座) 的世界位置
ee_pos = data.body("gripper_base").xpos.copy()
print(f"\n🟢 夹爪基座世界坐标 = {np.round(ee_pos, 3)}")

# ── 4. 交互式可视化仿真 (mujoco.viewer) ──────────────────────────────────────
# 打开一个实时仿真窗口: 左键旋转 / 右键平移 / 滚轮缩放 / 双击聚焦
print("\n── 开始交互式仿真 (打开实时窗口) ──")
print("   操作: 左键拖拽旋转 · 右键平移 · 滚轮缩放 · 关闭窗口即结束")
print("   窗口会一直保持打开，直到你手动关闭窗口")
SIM_DURATION = 5.0                      # 仅无界面降级模式的时间上限
n_steps = int(SIM_DURATION / model.opt.timestep)

# 夹爪开合演示 (position 执行器, ctrl = 目标位置)
# 夹爪关节范围 0~0.05: 0 = 张开(两指分离), 0.05 = 合拢(夹紧)
# 注意: 之前用 0.032 会令两指几乎合拢并互相重叠碰撞, 看起来"分不开";
#       要张开必须给 0.0。这里让夹爪周期性地 张开↔合拢, 直观看到分离。
FINGER_OPEN, FINGER_CLOSE = 0.00, 0.02          # 0.02 接近合拢但不接触
left_finger_act = model.actuator("left_finger_act").id
right_finger_act = model.actuator("right_finger_act").id
flip_steps = int(1.5 / model.opt.timestep)      # 每 1.5 秒切换一次开合

# 观察在重力下机械臂如何自然下垂 + 夹爪周期开合
qpos_history = []
try:
    import mujoco.viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("✅ 可视化窗口已打开，观察机械臂自然下垂 + 夹爪张开/合拢 ...")
        print("   (窗口保持打开，直到你手动关闭)")
        step = 0
        while viewer.is_running():              # 窗口不关闭就不结束
            # 根据时间在"张开/合拢"之间切换
            aperture = FINGER_OPEN if (step // flip_steps) % 2 == 0 else FINGER_CLOSE
            data.ctrl[left_finger_act] = aperture
            data.ctrl[right_finger_act] = aperture
            mujoco.mj_step(model, data)
            viewer.sync()                       # 把最新状态推送到窗口
            if step % 50 == 0:                  # 每 50 步记录一次
                qpos_history.append(data.qpos.copy())
            step += 1
        print("  用户关闭窗口，仿真结束")
except Exception as e:
    # 无显示器的环境 (SSH/服务器/容器) → 降级为无界面运行
    print(f"⚠️ 无法打开图形窗口 ({e})")
    print("   已切换为无界面模式运行 (只做物理仿真与截图) ...")
    for step in range(n_steps):
        aperture = FINGER_OPEN if (step // flip_steps) % 2 == 0 else FINGER_CLOSE
        data.ctrl[left_finger_act] = aperture
        data.ctrl[right_finger_act] = aperture
        mujoco.mj_step(model, data)
        if step % 50 == 0:
            qpos_history.append(data.qpos.copy())

if not qpos_history:
    qpos_history.append(data.qpos.copy())

# ── 5. 分析结果 ──────────────────────────────────────────────────────────────
print(f"\n仿真完成，共 {len(qpos_history) - 1} * 50 步，仿真时间 {data.time:.2f}s")
print("\n── 关节角度变化 (弧度) ──")
joint_names = ["shoulder_yaw", "shoulder_pitch", "elbow_pitch", "wrist_pitch"]
print(f"{'关节':<16}{'初始':>10}{'结束':>10}")
for name in joint_names:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    qadr = model.jnt_qposadr[jid]
    print(f"{name:<16}{qpos_history[0][qadr]:>10.3f}{qpos_history[-1][qadr]:>10.3f}")

# ── 6. 保存一帧离屏截图 (无论是否有窗口都执行) ────────────────────────────────
print("\n── 保存一帧截图 (first_sim_render.png) ──")
renderer = mujoco.Renderer(model, height=480, width=640)
renderer.update_scene(data)
img = renderer.render()
renderer.close()

out_png = Path(__file__).parent / "first_sim_render.png"
try:
    from PIL import Image
    Image.fromarray(img).save(out_png)
    print(f"✅ 图像已保存: {out_png}  形状={img.shape}")
except ImportError:
    print("未安装 PIL，跳过图像保存 (仅查看 numpy 数组)")

# 仿真耗时统计 (说明 MuJoCo 的"快")
print("\n💡 MuJoCo 性能提示: 单 CPU 核通常每秒可仿真数千步。"
      "配合 MJX (JAX GPU 加速) 可并行上千个环境。")
