from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from quadrl.control.mpc.mpc_controller import MPCController, RobotState
from quadrl.envs.base_env import BaseQuadrupedEnv
from quadrl.envs.reward import RewardFunction
from quadrl.gait.patterns.trot import TrotPattern
from quadrl.gait.scheduler import GaitScheduler
from quadrl.utils.config import Config

# ── Constants ──────────────────────────────────────────────────────────────────
NUM_JOINTS = 12   # 3 DoF × 4 legs

# Observation layout (total = 57):
#   joint positions    (12)
#   joint velocities   (12)
#   base linear vel    ( 3)
#   base angular vel   ( 3)
#   gravity in base    ( 3)
#   phase sin/cos      ( 8)   ← 2 × 4 legs
#   τ_mpc  reference   (12)   ← MPC reference torques (replaces gait q_ref)
#   base quat (w,x,y,z)( 4)
OBS_DIM = 57


class Go1Env(BaseQuadrupedEnv):
    """Unitree Go1 residual-RL environment with MPC reference controller.

    Responsibility (SRP):
        Manage the MuJoCo simulation step, assemble observations,
        evaluate termination, and delegate to MPC and gait scheduler.

    Data flow per step:
        1. GaitScheduler → contact schedule (stance mask over MPC horizon)
        2. MPCController → optimal GRFs → reference joint torques τ_mpc
        3. RL action δ ∈ [-1, 1]^12 → scaled residual torque
        4. τ_cmd = τ_mpc + action_scale * δ   (clipped to torque_limit)
        5. MuJoCo forward step

    Action space (12-dim):
        Residual joint torque offsets, normalised to [-1, 1].
        Scaled to Nm by cfg.control.action_scale before being applied.

    Observation space (57-dim):
        See OBS_DIM breakdown above.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self._cfg = cfg
        self._sim_steps = round(cfg.env.control_dt / cfg.env.sim_dt)

        # MuJoCo model
        model_path = Path(cfg.env.model_path)
        self._model = mujoco.MjModel.from_xml_path(str(model_path))
        self._data  = mujoco.MjData(self._model)

        # Gait scheduler — provides contact schedule for MPC
        pattern = self._build_pattern(cfg.gait.pattern)
        self._scheduler = GaitScheduler(
            pattern=pattern,
            frequency=cfg.gait.frequency,
            dt=cfg.env.control_dt,
            step_height=cfg.gait.step_height,
        )

        # MPC controller — provides reference torques τ_mpc
        self._mpc = MPCController(cfg.mpc, self._model, self._data)

        # Reward function
        self._reward_fn = RewardFunction()

        self._action_scale  = cfg.control.action_scale
        self._torque_limit  = cfg.control.torque_limit
        self._max_steps     = cfg.env.max_episode_steps
        self._step_count    = 0
        self._tau_mpc       = np.zeros(NUM_JOINTS)   # last MPC reference torques

        # Gymnasium spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(NUM_JOINTS,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )

        self.render_mode = cfg.env.render_mode

    # ── gymnasium.Env interface ────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self._model, self._data)
        self._scheduler.reset()
        self._step_count = 0
        self._tau_mpc    = np.zeros(NUM_JOINTS)
        return self._get_obs().astype(np.float32), self._get_info()

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        # 1. Advance gait clock → get contact schedule for MPC horizon
        _, phase_obs = self._scheduler.step()
        contact_schedule = self._build_contact_schedule()

        # 2. MPC → reference torques based on current state
        state = self._read_robot_state()
        desired_vel = np.array([self._cfg.mpc.target_height * 0.0,  # placeholder
                                0.0, 0.0])                           # TODO: command vel
        self._tau_mpc = self._mpc.compute(state, contact_schedule, desired_vel)

        # 3. Compose final torques: MPC reference + scaled RL residual
        tau_residual = self._action_scale * np.clip(action, -1.0, 1.0)
        tau_cmd = np.clip(
            self._tau_mpc + tau_residual,
            -self._torque_limit,
            self._torque_limit,
        )
        self._data.ctrl[:NUM_JOINTS] = tau_cmd

        # 4. Simulate at MuJoCo frequency
        for _ in range(self._sim_steps):
            mujoco.mj_step(self._model, self._data)

        self._step_count += 1
        obs        = self._get_obs().astype(np.float32)
        reward     = self._compute_reward(tau_cmd)
        terminated = self._is_terminated()
        truncated  = self._step_count >= self._max_steps

        return obs, reward, terminated, truncated, self._get_info()

    def render(self) -> np.ndarray | None:
        return None   # passive viewer: use mujoco.viewer externally

    def close(self) -> None:
        pass

    # ── BaseQuadrupedEnv interface ─────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        q    = self._data.qpos[7:19]          # (12) joint positions
        dq   = self._data.qvel[6:18]          # (12) joint velocities
        lv   = self._data.qvel[0:3]           # ( 3) base linear velocity
        av   = self._data.qvel[3:6]           # ( 3) base angular velocity
        grav = self._gravity_in_base()        # ( 3) gravity in base frame
        phi  = self._scheduler.clock.phase_obs()   # ( 8) sin/cos phases
        quat = self._data.qpos[3:7]           # ( 4) base orientation quaternion
        return np.concatenate([q, dq, lv, av, grav, phi, self._tau_mpc, quat])

    def _is_terminated(self) -> bool:
        return float(self._data.qpos[2]) < 0.15   # body collapsed

    def _get_info(self) -> dict:
        return {
            "step":        self._step_count,
            "base_height": float(self._data.qpos[2]),
            "forward_vel": float(self._data.qvel[0]),
            "gait_phase":  float(self._scheduler.clock.phase),
        }

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _read_robot_state(self) -> RobotState:
        """Extract body state from MuJoCo data for the MPC."""
        quat  = self._data.qpos[3:7]    # w, x, y, z
        euler = self._quat_to_euler(quat)
        return RobotState(
            euler    = euler,
            position = self._data.qpos[0:3].copy(),
            ang_vel  = self._data.qvel[3:6].copy(),
            lin_vel  = self._data.qvel[0:3].copy(),
        )

    def _build_contact_schedule(self) -> np.ndarray:
        """Predict contact schedule over MPC horizon using gait clock.

        Returns: (N, 4) bool array.
        """
        N          = self._cfg.mpc.horizon
        stance_ratio = self._scheduler.pattern.stance_ratio
        freq       = self._cfg.gait.frequency
        dt_mpc     = self._cfg.mpc.dt_mpc
        offsets    = self._scheduler.pattern.leg_offsets
        phase0     = self._scheduler.clock.phase

        schedule = np.zeros((N, 4), dtype=bool)
        for k in range(N):
            phase_k = (phase0 + freq * dt_mpc * k) % 1.0
            for i in range(4):
                leg_phase = (phase_k + offsets[i]) % 1.0
                schedule[k, i] = leg_phase < stance_ratio
        return schedule

    def _compute_reward(self, tau_cmd: np.ndarray) -> float:
        lv        = self._data.qvel[0:3]
        av        = self._data.qvel[3:6]
        foot_vels = np.zeros(4)   # TODO: populate from contact data
        return self._reward_fn(lv, av, tau_cmd, foot_vels, self._data.qpos[2])

    def _gravity_in_base(self) -> np.ndarray:
        quat = self._data.qpos[3:7]
        rot  = np.zeros(9)
        mujoco.mju_quat2Mat(rot, quat)
        R = rot.reshape(3, 3)
        return R.T @ np.array([0.0, 0.0, -1.0])

    @staticmethod
    def _quat_to_euler(quat: np.ndarray) -> np.ndarray:
        """MuJoCo quaternion (w,x,y,z) → ZYX Euler angles [roll, pitch, yaw]."""
        w, x, y, z = quat
        roll  = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        pitch = np.arcsin(np.clip(2*(w*y - z*x), -1.0, 1.0))
        yaw   = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        return np.array([roll, pitch, yaw])

    @staticmethod
    def _build_pattern(name: str):
        patterns = {"trot": TrotPattern}
        if name not in patterns:
            raise ValueError(f"Unknown gait pattern '{name}'. Available: {list(patterns)}")
        return patterns[name]()
