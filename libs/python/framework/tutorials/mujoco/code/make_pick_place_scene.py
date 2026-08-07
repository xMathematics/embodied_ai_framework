#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 make_pick_place_scene.py — 程序化生成 Pick-Place 抓取场景
 ================================================================================
 把"机械臂 + 被夹取的方块 + 目标落点"组装成一个完整的 MJCF 场景。
 演示 MJCF 的 <freejoint> (自由关节, 让方块拥有 6-DoF 运动学) 与程序化建模。

 输出:
   - 直接 import:  build_scene_xml() 返回场景 XML 字符串
   - 命令行运行:   生成 models/pick_place_scene.xml 静态文件

 场景布局 (z 轴向上):
   - 桌面顶面        : z = 0.40
   - 方块 (2cm 立方) : 中心 z = 0.42, 初始位置 (0.25, 0, 0.42)
   - 目标落点标记    : 桌面上的红色圆柱

 运行方式:
   python make_pick_place_scene.py
================================================================================
"""

from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


def build_scene_xml(
    block_pos: tuple = (0.25, 0.0, 0.42),
    target_pos: tuple = (-0.10, -0.25, 0.401),
    block_size: float = 0.02,
) -> str:
    """
    生成抓取场景的 MJCF 字符串。

    Args:
        block_pos:   方块初始位置 (桌面顶面 0.4 + 半高)
        target_pos:  目标落点位置 (薄圆柱放在桌面上)
        block_size:  方块半边长
    Returns:
        MJCF 场景字符串
    """
    arm_xml = (MODEL_DIR / "robot_arm.xml").read_text(encoding="utf-8")

    # 固定相机 (用于渲染视频): 从斜上方看桌面
    camera = """
    <camera name="side" pos="0.1 -1.1 0.75" xyaxes="1 0 0 0 0.35 0.94"
            fovy="55"/>
    <camera name="top" pos="0 0 1.3" xyaxes="1 0 0 0 -1 0" fovy="50"/>
    """

    block = f"""
    <!-- 被抓取的方块: 简单立方体, freejoint 赋予其 6 自由度 -->
    <body name="block" pos="{block_pos[0]} {block_pos[1]} {block_pos[2]}">
      <freejoint/>
      <geom name="block_geom" type="box" size="0.02 0.02 0.02"
            rgba="0.2 0.8 0.3 1" mass="0.05" friction="1.5 0.2 0.1"/>
    </body>

    <!-- 目标落点标记: 只做可视化，不影响物理 -->
    <body name="target" pos="{target_pos[0]} {target_pos[1]} {target_pos[2]}">
      <geom name="target_geom" type="cylinder" size="0.035 0.002"
            rgba="0.95 0.3 0.1 0.9" mass="0" contype="0" conaffinity="0"/>
    </body>
    """

    # 把相机、方块和目标插入到世界根节点的末尾 (</worldbody> 之前)
    scene_xml = arm_xml.replace("</worldbody>", camera + block + "  </worldbody>")
    return scene_xml


if __name__ == "__main__":
    # 命令行运行: 生成静态场景文件
    scene = build_scene_xml()
    out = MODEL_DIR / "pick_place_scene.xml"
    out.write_text(scene, encoding="utf-8")

    # 校验场景可正常加载
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(out))
    print(f"✅ 场景已生成: {out}")
    print(f"   关节数 = {model.njnt} (含夹爪), "
          f"body 数 = {model.nbody}, geom 数 = {model.ngeom}")
    print(f"   自由度 nq = {model.nq} (4 旋转 + 2 滑动 + 6 方块自由关节)")
