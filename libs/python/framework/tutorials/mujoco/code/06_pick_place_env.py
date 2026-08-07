#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 06_pick_place_env.py — 项目实战 (一): 抓取环境 PickPlaceEnv
 ================================================================================
 这是"从 MuJoCo API 到机器人任务"的关键一课。我们把前三节学到的所有
 知识点组装成一个完整的抓取环境:
   - MJCF 场景 (自定义 5-DoF 机械臂 + 方块 + 目标点)
   - 位置执行器 (position actuator, 关节伺服)
   - 阻尼最小二乘逆运动学 (DLS-IK), 用雅可比矩阵把"末端目标"转成"关节角"
   - 两指平行夹爪开合控制
   - Gymnasium 风格接口 (reset / step / render)

 环境接口 (方便接入强化学习或脚本控制):
   reset() -> obs
   step(action) -> obs, reward, terminated, truncated, info
   obs:  {qpos, ee_pos, block_pos, target_pos, gripper_width}
   action: 前 4 维 = 关节目标角, 后 2 维 = 手指目标开度

 逆运动学 (IK) 说明
 ------------------
 机械臂有 4 个关节: shoulder_yaw / shoulder_pitch / elbow_pitch / wrist_pitch。
 我们的目标: 让腕部 (gripper_base) 到达指定 3D 位置。
 用雅可比矩阵 J (3×4) 做阻尼最小二乘 (DLS):
     Δq = Jᵀ (J Jᵀ + λI)⁻¹ e        (e = 目标 - 当前位置)
 迭代若干次直至收敛。wrist_pitch 由解析法给出 (使夹爪竖直向下)。

 运行方式: 本文件是库, 由 07_scripted_pick_place.py 调用;
   也可 python 06_pick_place_env.py 自测 IK 是否可达指定点。
================================================================================
"""

from pathlib import Path

import mujoco
import numpy as np

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"

# 关节配置: 名称 → (qpos 索引, qvel 索引, actuator id)
ARM_JOINTS = ["shoulder_yaw", "shoulder_pitch", "elbow_pitch", "wrist_pitch"]
FINGER_JOINTS = ["left_finger", "right_finger"]

FINGER_OPEN = 0.0      # 手指张开 (qpos=0 → 手指在外侧 ±0.04)
FINGER_CLOSE = 0.048   # 手指闭合 (内表面 ±0.008, 强压方块面 ±0.02)

GRASP_OFFSET = 0.040   # 抓取时腕部相对方块中心的高度 (指尖落在方块中平面)


class PickPlaceEnv:
    """
    抓取环境: 把方块从初始位置抓起并放到目标位置。
    支持: 脚本控制 (PD + IK) 和强化学习 (gym 风格 step)。
    """
    def __init__(self, model_path=None, seed=None, visualize=False):
        model_path = model_path or str(MODEL_DIR / "pick_place_scene.xml")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.n_substeps = 5                      # 每个 step 推进的物理步
        self.max_steps = 600                     # 单 episode 最大步数
        self._step = 0
        self._attached_offset = None             # 虚拟抓取偏移 (None=未连接)

        # 关节元数据
        self.arm_jnt_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in ARM_JOINTS]
        self.arm_qadr = [self.model.jnt_qposadr[i] for i in self.arm_jnt_ids]
        self.arm_act_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n + "_act")
            for n in ARM_JOINTS]
        self.finger_act_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n + "_act")
            for n in FINGER_JOINTS]

        # 关键 body / site id
        self.body_ee = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper_base")
        self.body_block = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "block")

        if seed is not None:
            np.random.seed(seed)

    # ───────────────────────── 环境接口 ──────────────────────────────────────
    def reset(self, block_pos=None):
        """重置: 清空状态, 可选随机化方块初始位置"""
        mujoco.mj_resetData(self.model, self.data)
        self._attached_offset = None
        if block_pos is not None:
            self._set_body_pos("block", block_pos)
        # 默认姿态: 机械臂朝天举起, 夹爪张开
        self._set_joint("shoulder_pitch", -1.1)
        self._set_joint("elbow_pitch", -1.5)
        self._set_joint("wrist_pitch", -1.1)
        self._set_fingers(FINGER_OPEN)
        mujoco.mj_forward(self.model, self.data)
        self._step = 0
        return self._get_obs()

    def step(self, action):
        """
        执行一步。
        action (6,): [4 个关节目标角, 2 个手指开度]
        """
        action = np.asarray(action, dtype=np.float64)
        for i, act_id in enumerate(self.arm_act_ids):
            self.data.ctrl[act_id] = float(action[i])
        for i, act_id in enumerate(self.finger_act_ids):
            self.data.ctrl[act_id] = float(action[4 + i])
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        self._step += 1
        obs = self._get_obs()
        reward = self._reward(obs)
        terminated = self.is_success(obs)
        truncated = self._step >= self.max_steps
        return obs, reward, terminated, truncated, {}

    def render(self, camera=None, height=480, width=640):
        """离屏渲染一帧 RGB 图像 (H, W, 3); camera=None 用默认视角"""
        if not hasattr(self, "_renderer"):
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
        if camera is not None:
            self._renderer.update_scene(self.data, camera=camera)
        else:
            self._renderer.update_scene(self.data)
        return self._renderer.render()

    def close(self):
        if hasattr(self, "_renderer"):
            self._renderer.close()
            del self._renderer

    # ───────────────────────── 观测 / 奖励 ───────────────────────────────────
    def _get_obs(self):
        return {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "ee_pos": self.data.body("gripper_base").xpos.copy(),
            "block_pos": self.data.body("block").xpos.copy(),
            "target_pos": self.data.body("target").xpos.copy(),
            "gripper_width": self._finger_width(),
        }

    def _reward(self, obs):
        """密集奖励: 靠近方块 + 抓稳 + 放到目标"""
        block = obs["block_pos"]
        target = obs["target_pos"]
        ee = obs["ee_pos"]
        dist_block = np.linalg.norm(ee - block)
        dist_target = np.linalg.norm(block - target)
        return -0.01 * dist_block - 0.5 * (dist_target < 0.08)

    def is_success(self, obs=None):
        """方块到达目标点附近 5cm 内且稳定"""
        obs = obs or self._get_obs()
        block = obs["block_pos"]
        target = obs["target_pos"]
        # 只用 x-y 距离判断 (桌面高度已知)
        dist = np.linalg.norm(block[:2] - target[:2])
        return bool(dist < 0.05 and block[2] > 0.3)

    # ───────────────────────── 底层控制原语 ──────────────────────────────────
    def _set_joint(self, name, value):
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        self.data.qpos[self.model.jnt_qposadr[jid]] = value

    def _set_body_pos(self, name, pos):
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        jadr = self.model.body_jntadr[bid]
        # body 有 freejoint → qpos 中前 3 维是位置
        self.data.qpos[jadr:jadr + 3] = pos

    def _finger_width(self):
        """夹爪开度 = 左右手指滑动位移之和"""
        return float(self.data.qpos[4] + self.data.qpos[5])

    def _set_fingers(self, aperture):
        """设置手指目标开度 (两个手指一起开合, 0=张开, ~0.05=闭合)"""
        for act_id in self.finger_act_ids:
            self.data.ctrl[act_id] = float(np.clip(aperture, 0, 0.05))

    # ───────────────────────── 逆运动学 (DLS) ────────────────────────────────
    def ik_wrist(self, target_pos, n_iter=100, damping=1e-3, tol=2e-4,
                 step_max=0.1):
        """
        阻尼最小二乘逆运动学: 让腕部 (gripper_base) 到达 target_pos。
        用前 3 个关节 (shoulder_yaw/pitch, elbow_pitch) 的雅可比迭代求解，
        wrist_pitch 每轮由解析法设定 (让夹爪竖直朝下)。

        原理:
          已知当前雅可比 J (3×3) 与位置误差 e, 求关节增量 Δq:
            min ‖J·Δq - e‖² + λ‖Δq‖²   →   Δq = Jᵀ (J Jᵀ + λI)⁻¹ e
          λ 是阻尼系数, 既避免奇异点 (JJᵀ 不可逆) 又限制步长。

        Args:
            target_pos: (3,) 期望的腕部世界坐标
        Returns:
            (qpos 目标数组 (4,), 是否收敛)
        """
        target = np.asarray(target_pos, dtype=np.float64)
        # 前 3 个关节在 qvel 中的索引
        dofs = [self.model.jnt_dofadr[i] for i in self.arm_jnt_ids[:3]]

        for _ in range(n_iter):
            # 每轮重新计算雅可比 (位形变了)
            jacp = np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(self.model, self.data, jacp, None, self.body_ee)
            J = jacp[:, dofs]

            cur = self.data.body("gripper_base").xpos.copy()
            err = target - cur
            if np.linalg.norm(err) < tol:
                break

            # DLS 解: Δq = Jᵀ (J Jᵀ + λI)⁻¹ e
            JJt = J @ J.T
            dq = J.T @ np.linalg.solve(JJt + damping * np.eye(3), err)
            dq = np.clip(dq, -step_max, step_max)      # 限幅防震荡

            for i in range(3):
                self.data.qpos[self.arm_qadr[i]] += dq[i]

            # 解析设定 wrist_pitch: 让夹爪竖直朝下。
            # 推导: 腕部局部 X 轴在世界系的方向由累计俯仰角 φ=θ1+θ2+θ3 决定:
            #   R_y(φ)·X̂ = (cosφ, 0, -sinφ)
            # 要求朝下 (-Z) → cosφ=0, sinφ=1 → φ=π/2 → θ3 = π/2 - θ1 - θ2
            th1 = self.data.qpos[self.arm_qadr[1]]
            th2 = self.data.qpos[self.arm_qadr[2]]
            self.data.qpos[self.arm_qadr[3]] = np.pi / 2 - (th1 + th2)

            mujoco.mj_forward(self.model, self.data)

        err = np.linalg.norm(target - self.data.body("gripper_base").xpos)
        return self.data.qpos.copy(), err < tol

    def move_ee_to(self, target_pos, fingers=FINGER_OPEN, n_iter=100, tol=2e-4):
        """
        高层命令: 让夹爪移动到指定位置并设定开合度。
        内部: DLS-IK 求出关节目标 → 位置伺服 (ctrl=目标角) → 等待收敛。
        若处于"虚拟抓取"状态 (attach_block), 每步同步方块跟随夹爪。
        """
        self.ik_wrist(target_pos, n_iter=n_iter)     # 一次完整 IK 求解
        # 把 IK 结果作为 position actuator 的目标
        for i, act_id in enumerate(self.arm_act_ids):
            self.data.ctrl[act_id] = float(self.data.qpos[self.arm_qadr[i]])
        self._set_fingers(fingers)
        # 伺服到收敛
        for _ in range(300):
            mujoco.mj_step(self.model, self.data)
            self._sync_block()                       # 虚拟抓取时同步方块
            if np.linalg.norm(target_pos - self.data.body("gripper_base").xpos) < tol:
                break

    # ───────────────────────── 虚拟抓取 (运动学) ─────────────────────────────
    # 说明: 真实摩擦夹持在 MuJoCo 里对薄手指+桌面方块容易失效 (接触建模很微妙)。
    #       为让"项目实战"可靠运行, 采用运动学虚拟抓取: 抓取后让方块每步
    #       精确跟随夹爪 (像被"焊住"), 抬升/搬运/放置完全可控。
    #       这也正是很多 MuJoCo 官方演示的做法 (mj_attachBody 的功能)。
    def attach_block(self):
        """记录夹爪与方块的相对偏移, 开始虚拟抓取"""
        self._attached_offset = (
            self.data.body("block").xpos - self.data.body("gripper_base").xpos
        )
        return self._attached_offset.copy()

    def release_block(self, drop=True):
        """结束虚拟抓取 (可选让方块自然下落)"""
        self._attached_offset = None
        if drop:
            # 解除后让方块在重力下自由落体 (会落到桌面/目标处)
            for _ in range(self.n_substeps * 10):
                mujoco.mj_step(self.model, self.data)

    def _sync_block(self):
        """若处于虚拟抓取, 把方块位置同步到夹爪 + 偏移"""
        if self._attached_offset is None:
            return
        # 方块 freejoint 的 qpos 前 3 维是位置
        bid = self.body_block
        jadr = self.model.body_jntadr[bid]
        self.data.qpos[jadr:jadr + 3] = (
            self.data.body("gripper_base").xpos + self._attached_offset
        )
        mujoco.mj_forward(self.model, self.data)


if __name__ == "__main__":
    # 自测: 让夹爪到达方块正上方, 验证 IK 可用
    env = PickPlaceEnv(seed=0)
    obs = env.reset()
    print("初始观测: 夹爪 =", np.round(obs["ee_pos"], 3),
          " 方块 =", np.round(obs["block_pos"], 3),
          " 目标 =", np.round(obs["target_pos"], 3))

    # 计算到达方块上方 0.06m 处的 IK
    target = obs["block_pos"].copy()
    target[2] += 0.06
    q, converged = env.ik_wrist(target, n_iter=40)
    print(f"IK 求解: 目标 = {np.round(target, 3)}, 收敛 = {converged}")
    print(f"  实际到达 = {np.round(env.data.body('gripper_base').xpos, 3)}")

    # 渲染确认
    img = env.render()
    from PIL import Image
    out = Path(__file__).parent / "ik_test.png"
    Image.fromarray(img).save(out)
    print(f"  渲染帧已保存: {out}")
    print("\n✅ 环境初始化与 IK 自测完成")
