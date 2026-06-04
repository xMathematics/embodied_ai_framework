"""
================================================================================
 算法层 (Algorithm Layer)
 ──────────────────────────────────────────────────────────────────────────────
 五层架构中的第四层，提供各种机器人学习算法的实现。

 核心模块:
   - policy.py:        策略网络基类和神经网络组件
   - imitation/:       模仿学习算法实现
       - bc.py:              行为克隆 (Behavior Cloning)
       - act.py:             Action Chunking Transformer
       - diffusion_policy.py: 扩散策略 (Diffusion Policy)
   - reinforcement/:   强化学习算法实现
       - ppo.py:             PPO (Proximal Policy Optimization)
       - sac.py:             SAC (Soft Actor-Critic)
   - world_model.py:   世界模型与基于模型的强化学习
   - sim2real.py:      Sim-to-Real 迁移模块

 设计原则:
   所有算法实现遵循统一的 Policy 和 Trainer 接口，
   可以无缝切换不同算法进行比较和集成。
================================================================================
"""
