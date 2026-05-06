"""Go1 residual-RL environment with MPC-based nominal controller.

Data flow per step:
    1. NominalController → ctrl_nominal (position targets via MPC + Cartesian swing + LLC)
    2. τ_nominal = kp_act * (ctrl_nominal - q)   (equivalent nominal torques)
    3. RL action δ ∈ [-1, 1]^12 → δτ = action_scale * δ
    4. τ_total  = clip(τ_nominal + δτ, ±torque_limit)
    5. ctrl_final = q + τ_total / kp_act          (implicit torque control)
    6. MuJoCo forward step

Go1 uses position actuators (kp=100 Nm/rad).  The implicit torque formula
ensures force = kp*(ctrl - q) = τ_total regardless of RL residual.
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from gymnasium import spaces

from quadrl.control.nominal_controller import NominalController
from quadrl.envs.base_env import BaseQuadrupedEnv
from quadrl.envs.reward import RewardFunction
from quadrl.utils.config import Config

# ── Constants ──────────────────────────────────────────────────────────────────
NUM_JOINTS = 12

_FOOT_SITE_NAMES = ("FR", "FL", "RR", "RL")

# Observation layout (total = 60):
#   joint positions    (12)
#   joint velocities   (12)
#   base linear vel    ( 3)
#   base angular vel   ( 3)
#   gravity in base    ( 3)
#   phase sin/cos      ( 8)  — 2 × 4 legs from gait clock
#   τ_nominal          (12)  — nominal controller equivalent joint torques
#   base quat (w,x,y,z)( 4)
#   cmd_vel (vx,vy,ωz) ( 3)
OBS_DIM = 60

# Velocity command bounds for training randomisation
_CMD_VX_RANGE   = (0.2, 1.5)
_CMD_VY_RANGE   = (-0.4, 0.4)
_CMD_VYAW_RANGE = (-0.5, 0.5)


class Go1Env(BaseQuadrupedEnv):
    """Unitree Go1 residual-RL environment with MPC reference controller.

    Action space (12-dim):
        Residual joint torque offsets, normalised to [-1, 1].

    Observation space (60-dim):
        See OBS_DIM breakdown above.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self._cfg = cfg
        self._sim_steps = round(cfg.env.control_dt / cfg.env.sim_dt)

        model_path = Path(cfg.env.model_path)
        self._model = mujoco.MjModel.from_xml_path(str(model_path))
        self._data  = mujoco.MjData(self._model)

        self._foot_site_ids = [
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, name)
            for name in _FOOT_SITE_NAMES
        ]

        # Nominal controller — footstep planner + MPC + Cartesian swing + LLC
        self._nominal = NominalController(cfg, self._model, self._data)

        self._reward_fn    = RewardFunction()
        self._action_scale = cfg.control.action_scale
        self._torque_limit = cfg.control.torque_limit
        self._kp_actuator  = cfg.control.kp_actuator
        self._max_steps    = cfg.env.max_episode_steps
        self._step_count   = 0
        self._tau_nominal  = np.zeros(NUM_JOINTS)   # for observation
        self._cmd_vel      = np.array([0.5, 0.0, 0.0])

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(NUM_JOINTS,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )

        self.render_mode = cfg.env.render_mode
        self._renderer: mujoco.Renderer | None = None
        self._viewer = None

    # ── gymnasium.Env interface ────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self._model, self._data, 0)   # "home" standing pose @ 0.27m
        mujoco.mj_forward(self._model, self._data)
        self._nominal.reset()
        self._step_count  = 0
        self._tau_nominal = np.zeros(NUM_JOINTS)

        if options and "cmd_vel" in options:
            self._cmd_vel = np.asarray(options["cmd_vel"], dtype=np.float64)
        else:
            self._cmd_vel = np.array([
                self.np_random.uniform(*_CMD_VX_RANGE),
                self.np_random.uniform(*_CMD_VY_RANGE),
                self.np_random.uniform(*_CMD_VYAW_RANGE),
            ])

        return self._get_obs().astype(np.float32), self._get_info()

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        # 1. Nominal controller → position targets (MPC + Cartesian swing + LLC)
        ctrl_nominal = self._nominal.step(self._cmd_vel)

        # 2. Convert nominal position targets to equivalent joint torques
        q = self._data.qpos[7:19]
        tau_nominal = self._kp_actuator * (ctrl_nominal - q)
        self._tau_nominal = tau_nominal.copy()

        # 3. Add RL residual in torque space, then re-apply implicit torque control
        tau_residual = self._action_scale * np.clip(action, -1.0, 1.0)
        tau_total = np.clip(
            tau_nominal + tau_residual,
            -self._torque_limit,
            self._torque_limit,
        )
        self._data.ctrl[:NUM_JOINTS] = q + tau_total / self._kp_actuator

        # 4. Simulate at MuJoCo frequency
        for _ in range(self._sim_steps):
            mujoco.mj_step(self._model, self._data)

        self._step_count += 1
        obs        = self._get_obs().astype(np.float32)
        reward     = self._compute_reward(tau_total)
        terminated = self._is_terminated()
        truncated  = self._step_count >= self._max_steps

        return obs, reward, terminated, truncated, self._get_info()

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self._model, height=480, width=640)
            self._renderer.update_scene(self._data)
            return self._renderer.render()
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self._model, self._data)
            self._viewer.sync()
        return None

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    def set_command(self, vx: float, vy: float = 0.0, yaw_rate: float = 0.0) -> None:
        """Set the velocity command for the robot (m/s, rad/s)."""
        self._cmd_vel = np.array([vx, vy, yaw_rate])

    # ── BaseQuadrupedEnv interface ─────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        q    = self._data.qpos[7:19]                              # (12) joint positions
        dq   = self._data.qvel[6:18]                              # (12) joint velocities
        lv   = self._data.qvel[0:3]                               # ( 3) base linear vel
        av   = self._data.qvel[3:6]                               # ( 3) base angular vel
        grav = self._gravity_in_base()                            # ( 3) gravity in base
        phi  = self._nominal._scheduler.clock.phase_obs()         # ( 8) sin/cos phases
        quat = self._data.qpos[3:7]                               # ( 4) base quaternion
        return np.concatenate([q, dq, lv, av, grav, phi, self._tau_nominal, quat, self._cmd_vel])

    def _is_terminated(self) -> bool:
        return float(self._data.qpos[2]) < 0.15

    def _get_info(self) -> dict:
        return {
            "step":        self._step_count,
            "base_height": float(self._data.qpos[2]),
            "forward_vel": float(self._data.qvel[0]),
            "gait_phase":  float(self._nominal._scheduler.clock.phase),
            "cmd_vel":     self._cmd_vel.copy(),
        }

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _foot_tangential_velocities(self) -> np.ndarray:
        """Horizontal foot speed per leg — used as stance-slip proxy."""
        stance = self._nominal._scheduler.stance_mask()
        foot_vels = np.zeros(4)
        n_v  = self._model.nv
        jacp = np.zeros((3, n_v))

        for i, sid in enumerate(self._foot_site_ids):
            if not stance[i]:
                continue
            jacp[:] = 0.0
            mujoco.mj_jacSite(self._model, self._data, jacp, None, sid)
            v_foot = jacp @ self._data.qvel
            foot_vels[i] = float(np.linalg.norm(v_foot[:2]))

        return foot_vels

    def _compute_reward(self, tau_cmd: np.ndarray) -> float:
        lv        = self._data.qvel[0:3]
        av        = self._data.qvel[3:6]
        foot_vels = self._foot_tangential_velocities()
        return self._reward_fn(lv, av, tau_cmd, foot_vels, self._data.qpos[2], self._cmd_vel)

    def _gravity_in_base(self) -> np.ndarray:
        quat = self._data.qpos[3:7]
        rot  = np.zeros(9)
        mujoco.mju_quat2Mat(rot, quat)
        R = rot.reshape(3, 3)
        return R.T @ np.array([0.0, 0.0, -1.0])
