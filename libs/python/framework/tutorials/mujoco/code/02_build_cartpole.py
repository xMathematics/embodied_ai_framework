#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 02_build_cartpole.py — MJCF 场景建模入门: 从零构建倒立摆
 ================================================================================
 本节目标:
   1. 学习 MJCF (MuJoCo XML) 的核心语法: <mujoco> <worldbody> <body> <geom> <joint>
   2. 用字符串或文件构建一个经典的"小车-倒立摆" (Cart-Pole) 模型
   3. 验证模型正确加载，并对比"带执行器"与"纯被动"两种情况

 MJCF 语法速览
 -------------
   <mujoco>            根节点 (可含 <option>/<default>/<worldbody>/<actuator>...)
   <option>            全局物理参数 (timestep 仿真步长, gravity 重力, ...)
   <default>           默认值 (可被子元素继承，如默认阻尼)
   <worldbody>         世界根节点 (不可移动的固定世界)
     <body>            刚体 (可含子 <body>，形成运动学树)
       <joint>         关节 (hinge 铰链 / slide 滑动 / free 自由 6-DoF)
       <geom>          几何体 (碰撞与渲染外观; box/capsule/sphere/cylinder...)
   <actuator>          执行器 (motor 力矩 / position 位置 / velocity 速度 ...)

 运行方式:
   python 02_build_cartpole.py
================================================================================
"""

import tempfile
from pathlib import Path

import mujoco
import numpy as np

# ── 1. 用三引号字符串定义 MJCF 场景 ──────────────────────────────────────────
# 建模要点:
#   1. 小车与地面之间必须有间隙 —— 若几何体互相穿透会产生巨大接触力
#      (本模型小车悬空, 只靠 slide 关节约束在 1D 轨道上, 地面上无接触)
#   2. 摆杆用 capsule 表示杆身 + sphere 表示顶端配重
CART_POLE_XML = """
<mujoco model="cartpole">
  <!-- 全局参数: 仿真步长 20ms (50Hz)，标准 RL 环境节奏 -->
  <option timestep="0.02"/>

  <!-- 默认值: 给所有关节加一点阻尼，避免数值振荡 -->
  <default>
    <joint damping="0.1"/>
  </default>

  <worldbody>
    <!-- 光源 (没有光照则渲染是黑的) -->
    <light name="light" pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>

    <!-- 地面 (纯视觉 + 兜底碰撞) -->
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.9 0.9 0.9 1"/>

    <!-- 小车 body: 悬空 0.15m, 包含一个滑动关节 (沿 X 轴平移) -->
    <body name="cart" pos="0 0 0.15">
      <joint name="slider" type="slide" axis="1 0 0" range="-1.2 1.2"/>
      <geom name="cart_geom" type="box" size="0.25 0.06 0.06" rgba="0.2 0.6 1 1"/>

      <!-- 摆杆 body (小车的子体): 包含一个铰链关节 (绕 Y 轴旋转) -->
      <body name="pole" pos="0 0 0.06">
        <joint name="hinge" type="hinge" axis="0 1 0" range="-180 180"/>
        <geom name="pole_geom" type="capsule" size="0.03 0.25" pos="0 0 0.25"
              rgba="1 0.3 0.3 1"/>
        <geom name="pole_mass" type="sphere" size="0.05" pos="0 0 0.5"
              rgba="1 0.8 0.3 1"/>
      </body>
    </body>
  </worldbody>

  <!-- 执行器: motor 表示"力矩控制"，gear 把 ctrl 放大 10 倍 (可理解为力臂) -->
  <actuator>
    <motor name="cart_motor" joint="slider" gear="10" ctrlrange="-3 3"/>
  </actuator>
</mujoco>
"""

# 保存到模型目录，方便后续章节 (RL 训练) 复用
model_dir = Path(__file__).resolve().parents[1] / "models"
model_dir.mkdir(parents=True, exist_ok=True)
xml_path = model_dir / "cartpole.xml"
xml_path.write_text(CART_POLE_XML, encoding="utf-8")
print(f"✅ MJCF 已写入: {xml_path}")

# ── 2. 从字符串加载模型 ──────────────────────────────────────────────────────
#   from_xml_string: 不落盘直接加载
#   from_xml_path:   从文件加载 (支持 <include> 引入外部子模型)
model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)

print(f"\n模型信息: 关节={model.njnt}, body={model.nbody}, "
      f"geom={model.ngeom}, 执行器={model.nu}")
print(f"初始 qpos = {np.round(data.qpos, 3)}")
print(f"  qpos[0] = 小车位置 x")
print(f"  qpos[1] = 摆杆角度 θ (弧度)")

# ── 3. 被动演化: 不加控制，看摆杆在重力下倒下 ───────────────────────────────
# 注意: 摆杆初始严格直立 (θ=0) 是不稳定平衡点，永远不会自己倒下。
#       所以这里给一个微小的初始偏角 (5°)，重力才会让它倒下来。
print("\n── 被动演化 (无控制, 初始偏角 5°) ──")
data.qpos[1] = np.deg2rad(5.0)
mujoco.mj_forward(model, data)         # 让 qpos 改动立即反映到世界坐标
data.ctrl[:] = 0.0
for _ in range(100):
    mujoco.mj_step(model, data)
print(f"  100 步后: 摆杆角度 = {np.degrees(data.qpos[1]):.1f}° "
      f"(应明显倒下, 说明重力生效)")

# ── 4. 主动控制: 给恒定力矩，观察小车被推动 ──────────────────────────────────
print("── 恒定力矩 10N (ctrl=1, gear=10) ──")
mujoco.mj_resetData(model, data)     # 重置回初始状态
data.ctrl[0] = 1.0
for _ in range(100):
    mujoco.mj_step(model, data)
print(f"  100 步后: 小车位置 = {data.qpos[0]:.3f} m (应被推离原点)")

# ── 5. 渲染一帧查看模型外观 ──────────────────────────────────────────────────
renderer = mujoco.Renderer(model, height=400, width=600)
mujoco.mj_forward(model, data)
renderer.update_scene(data)
img = renderer.render()
renderer.close()

out_png = Path(__file__).parent / "cartpole_render.png"
from PIL import Image
Image.fromarray(img).save(out_png)
print(f"\n✅ 渲染帧已保存: {out_png}")

print("\n💡 动手练习: 修改 CART_POLE_XML 中的 "
      "timestep / 摆杆长度 / gear，观察行为如何变化。")
