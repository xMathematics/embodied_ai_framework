#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 03_sensors_actuators.py — 传感器与执行器
 ================================================================================
 本节目标:
   1. 认识 MuJoCo 的传感器系统 (<sensor> 元素) 与三种核心类型
   2. 认识执行器的三种控制模式: 力矩(motor) / 位置(position) / 速度(velocity)
   3. 理解 ctrl 在每种模式下的含义

 传感器类型 (接入 <sensor> 标签, 值在 data.sensordata 中):
   - jointpos / jointvel  : 关节位置/速度
   - actuatorforce        : 执行器实际输出力
   - framepos / framevel  : 指定 body 的位置/速度
   - touch / contact      : 接触力

 执行器类型:
   - motor    : ctrl 是"力" (乘以 gear)
   - position : ctrl 是"目标位置"，内部用隐式 PD 伺服
   - velocity : ctrl 是"目标速度"，内部用隐式阻尼伺服

 运行方式:
   python 03_sensors_actuators.py
================================================================================
"""

from pathlib import Path

import mujoco
import numpy as np

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"

# ── 1. 加载倒立摆并为其注入传感器 ────────────────────────────────────────────
# 在 MJCF 字符串里直接加 <sensor> 元素 (这里演示传感器最常用的几种)
cart_pole_xml = (MODEL_DIR / "cartpole.xml").read_text(encoding="utf-8")
sensor_xml = """
<sensor>
  <jointpos name="j_pos" joint="slider"/>
  <jointvel name="j_vel" joint="slider"/>
  <actuatorfrc name="act_force" actuator="cart_motor"/>
  <framepos name="pole_tip_pos" objtype="site" objname="pole_tip"/>
</sensor>
"""
# 把传感器插入到 </mujoco> 之前；同时给摆杆加一个 site (用于测量摆尖位置)
cart_pole_xml = cart_pole_xml.replace(
    "<geom name=\"pole_mass\"",
    "<site name=\"pole_tip\" pos=\"0 0 0.5\" size=\"0.01\" />\n      "
    "<geom name=\"pole_mass\"",
)
cart_pole_xml = cart_pole_xml.replace("</mujoco>", sensor_xml + "</mujoco>")

model = mujoco.MjModel.from_xml_string(cart_pole_xml)
data = mujoco.MjData(model)

# 传感器数量与名称
print("✅ 传感器列表:")
for i in range(model.nsensor):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
    print(f"  [{i}] {name}")

# ── 2. 读取传感器数据 ────────────────────────────────────────────────────────
print("\n── 驱动 200 步后读取传感器 ──")
data.ctrl[0] = 0.5                      # 施加 5N (gear=10)
for _ in range(200):
    mujoco.mj_step(model, data)

for i in range(model.nsensor):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
    # 每个传感器在 sensordata 中的起始位置和维度
    adr = model.sensor_adr[i]
    dim = model.sensor_dim[i]
    print(f"  {name:<14} = {np.round(data.sensordata[adr:adr + dim], 3)}")

# 摆杆尖端 (site) 的世界坐标
tip_pos = data.site("pole_tip").xpos.copy()
print(f"  摆尖世界坐标    = {np.round(tip_pos, 3)}")

# ── 3. 三种执行器模式对比 ────────────────────────────────────────────────────
print("\n── 三种执行器模式对比 (同一个倒立摆) ──")
print("   每次重置后设 ctrl=0.1，观察小车 100 步内的位移含义")

def demo_actuator(actuator_tag: str, ctrl_value: float, label: str, n=100):
    """用指定的执行器类型构建模型并观察行为"""
    xml = (MODEL_DIR / "cartpole.xml").read_text(encoding="utf-8")
    # 替换原有 motor 执行器
    xml = xml.replace(
        '<motor name="cart_motor" joint="slider" gear="10" ctrlrange="-3 3"/>',
        actuator_tag,
    )
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    d.ctrl[0] = ctrl_value
    for _ in range(n):
        mujoco.mj_step(m, d)
    print(f"  {label:<38} ctrl={ctrl_value:<6} → 小车位移={d.qpos[0]:+.3f} m")

demo_actuator(
    '<motor name="cart_motor" joint="slider" gear="10" ctrlrange="-3 3"/>',
    0.1, "motor (力 = ctrl×gear = 1N)")
demo_actuator(
    '<position name="cart_motor" joint="slider" kp="50" ctrlrange="-1.2 1.2"/>',
    0.5, "position (目标位置 = 0.5m)")
demo_actuator(
    '<velocity name="cart_motor" joint="slider" kv="10" ctrlrange="-5 5"/>',
    0.5, "velocity (目标速度 = 0.5 m/s)")

print("\n💡 关键理解:")
print("   - motor    : 直接控制力/力矩 → 适合 RL 连续动作、经典控制理论")
print("   - position : 内部做 PD 伺服 → 机械臂抓取最常用 (本框架机械臂即用此)")
print("   - velocity : 内部做阻尼伺服 → 速度跟踪任务")
print("   - 切换模式只需改 <actuator> 一行，模型其他部分不变！")
