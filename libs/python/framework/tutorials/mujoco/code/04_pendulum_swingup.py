#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 04_pendulum_swingup.py — 经典控制实战: 力矩控制 + 能量整形起摆
 ================================================================================
 本节目标:
   1. 用 <motor> (力矩) 执行器控制一个单摆
   2. 实现经典的"能量整形 + PD"起摆控制 (Energy Shaping)
   3. 观察摆杆从下垂位置被"泵"到直立并稳定

 控制思想
 --------
 单摆只有 1 个自由度，力矩受限 (|u| ≤ 10 Nm)，无法直接把摆"抬"到直立。
 经典解法: 能量整形 (Energy Shaping)。约定: θ=0 表示直立 (摆杆指向上方)。
   1. 系统机械能 E = ½Iω² + mgL·cosθ   (直立时 cosθ=1, 能量最高)
   2. 目标能量 E* = mgL (直立静止)
   3. 泵能控制: u = kp·(E*-E)·sign(ω)
      由 dE/dt = u·ω 可知: 能量不足 (E<E*) 时让力矩与角速度同号即可
      "顺着摆动方向推一把"逐步累积能量; 能量过冲 (E>E*) 时自动反向制动。
   4. 当 |θ|<阈值 且 |E-E*|<阈值 (摆真的到达直立附近且能量足够)
      → 切换为线性 PD 锁定: u = -kpp·θ - kdd·ω

 运行方式:
   python 04_pendulum_swingup.py
================================================================================
"""

from pathlib import Path

import mujoco
import numpy as np

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"

PENDULUM_XML = """
<mujoco model="pendulum">
  <option timestep="0.01" gravity="0 0 -9.81"/>

  <worldbody>
    <light name="light" pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="pivot" type="cylinder" size="0.03 0.01" pos="0 0 0" rgba="0.5 0.5 0.5 1"/>

    <!-- 单摆: 铰链在原点, 摆杆沿 +Z, 摆尖有配重 -->
    <body name="pole" pos="0 0 0">
      <joint name="hinge" type="hinge" axis="0 1 0"/>
      <geom name="pole_geom" type="capsule" size="0.025 0.25" pos="0 0 0.25" rgba="0.8 0.3 0.9 1"/>
      <geom name="pole_mass" type="sphere" size="0.06" pos="0 0 0.5" rgba="0.95 0.6 0.1 1"/>
    </body>
  </worldbody>

  <!-- 力矩执行器: ctrl 单位 = Nm -->
  <actuator>
    <motor name="motor" joint="hinge" gear="1" ctrlrange="-10 10"/>
  </actuator>
</mujoco>
"""

xml_path = MODEL_DIR / "pendulum.xml"
xml_path.write_text(PENDULUM_XML, encoding="utf-8")

model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)

# 物理常量: 绕铰链的转动惯量、摆总质量、质心到铰链的距离
#   body_inertia: body 绕自身坐标系原点 (即铰链点) 的对角惯量张量
#   绕 Y 轴 (铰链轴) 旋转 → 取 [1,1] 分量
I = float(model.body_inertia[1, 1])           # body 1 = pole 的绕轴转动惯量
m = float(model.body_subtreemass[1])          # pole 子树总质量
# 质心到铰链点的距离 L: body_ipos 是质心在 body 系下的偏移,
# 而铰链就在 body 原点, 所以 L = |body_ipos| (本模型中沿 +Z)
L = float(np.linalg.norm(model.body_ipos[1]))
g = -float(model.opt.gravity[2])              # 9.81

E_star = m * g * L                              # 直立静止能量 (θ=0)
print(f"转动惯量 I={I:.4f}, 质量 m={m:.3f} kg, 质心距 L={L:.3f} m")
print(f"目标能量 E* = {E_star:.4f} J")

# ── 控制参数 ────────────────────────────────────────────────────────────────
KP, KPP, KDD = 8.0, 15.0, 2.0     # 泵能增益 / PD 锁定增益
EPS_ANGLE = 0.4                    # 进入锁定模式的角度阈值 (弧度, ≈23°)
EPS_ENERGY = 0.25                  # 进入锁定模式的能量阈值 (相对 E*)

def compute_energy(d):
    """当前机械能 E = ½Iω² + mgL·cosθ  (θ=0 直立, cosθ=1 最高)"""
    theta = d.qpos[0]
    omega = d.qvel[0]
    return 0.5 * I * omega**2 + m * g * L * np.cos(theta)

def control(d):
    """返回控制力矩 (能量整形 → PD 切换)
    两个切换条件 (角度 + 能量) 缺一不可:
      - 角度足够接近直立
      - 能量已接近目标 (否则说明只是"路过"直立, 还没到平衡点)
    """
    theta, omega = d.qpos[0], d.qvel[0]
    # 把角度归一化到 [-π, π]: 直立时 θ≈0
    theta_n = np.angle(np.exp(1j * theta))
    E = compute_energy(d)

    if abs(theta_n) < EPS_ANGLE and abs(E - E_star) < EPS_ENERGY * E_star:
        # ── 锁定模式: 线性 PD 围绕直立点 (θ=0) ──
        return -KPP * theta_n - KDD * omega, "PD锁定"
    else:
        # ── 泵能模式: u = k·(E*-E)·sign(ω)
        #     能量不足 (E<E*) 时顺着摆动方向推一把 (dE/dt = u·ω > 0),
        #     能量过冲 (E>E*) 时反向制动 —— 这就是"能量整形"的本质。
        u = KP * (E_star - E) * np.sign(omega)
        return np.clip(u, -10, 10), "泵能"

# ── 仿真主循环 (10 秒) ──────────────────────────────────────────────────────
# 初始状态: 摆杆下垂 (θ=π), 让控制算法从零开始把摆"泵"到直立
data.qpos[0] = np.pi
mujoco.mj_forward(model, data)
print("\n── 开始起摆 (10 秒, 初始下垂) ──")
TOTAL_TIME = 10.0
n_steps = int(TOTAL_TIME / model.opt.timestep)

# 记录角度历史 (用于绘图/视频)
theta_hist = []
for step in range(n_steps):
    u, mode = control(data)
    data.ctrl[0] = u
    mujoco.mj_step(model, data)
    theta_hist.append(data.qpos[0])
    if step % 200 == 0:
        print(f"  t={data.time:5.2f}s  θ={np.degrees(data.qpos[0]):7.1f}°  "
              f"ω={data.qvel[0]:+.2f} rad/s  [{mode}]")

# ── 结果分析 ────────────────────────────────────────────────────────────────
theta_final = np.degrees(theta_hist[-1]) % 360
# 直立 = 0° (θ=0 即直立)
upright = abs(np.degrees(np.angle(np.exp(1j * data.qpos[0]))))
print(f"\n最终角度 = {theta_final:.1f}°, 距直立偏差 = {upright:.1f}°")
print("✅ 起摆成功！" if upright < 10 else "⚠️ 未完全直立，可调大 KP 或延长仿真时间")

# 输出用于 Matplotlib 的绘图示例 (演示 numpy/matplotlib 分析能力)
try:
    import matplotlib
    matplotlib.use("Agg")                     # 无界面后端
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 3))
    t = np.arange(len(theta_hist)) * model.opt.timestep
    ax.plot(t, np.degrees(np.unwrap(theta_hist)) % 360)
    ax.axhline(0, color="r", ls="--", label="upright (0 deg)")
    ax.set_xlabel("time (s)"); ax.set_ylabel("angle (deg)")
    ax.legend(); ax.grid(alpha=0.3)
    out = Path(__file__).parent / "swingup_angle.png"
    fig.tight_layout(); fig.savefig(out)
    print(f"📈 角度曲线已保存: {out}")
except ImportError:
    print("未安装 matplotlib，跳过绘图")

print("\n💡 进阶: 把这个『泵能+锁定』策略交给强化学习 (PPO) 学习，"
      "就是 08 节倒立摆训练的原型。")
