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

# Leg index convention (matches MuJoCo actuator order): [FR, FL, RR, RL]
_FOOT_SITE_NAMES = ("FR", "FL", "RR", "RL")

# Observation layout (total = 60):
#   joint positions    (12)
#   joint velocities   (12)
#   base linear vel    ( 3)
#   base angular vel   ( 3)
#   gravity in base    ( 3)
#   phase sin/cos      ( 8)  — 2 × 4 legs
#   τ_mpc  reference   (12)  — MPC reference torques
#   base quat (w,x,y,z)( 4)
#   cmd_vel (vx,vy,ωz) ( 3)  — velocity command to the policy
OBS_DIM = 60

# Velocity command bounds for training randomisation
_CMD_VX_RANGE   = (0.2, 1.5)   # m/s forward
_CMD_VY_RANGE   = (-0.4, 0.4)  # m/s lateral
_CMD_VYAW_RANGE = (-0.5, 0.5)  # rad/s yaw rate

class Go1Env(BaseQuadrupedEnv):
    """Unitree Go1 residual-RL environment with MPC reference controller.

    Data flow per step:
        1. GaitScheduler → contact schedule (stance mask over MPC horizon)
        2. MPCController → optimal GRFs → reference joint torques τ_mpc
        3. RL action δ ∈ [-1, 1]^12 → scaled residual torque
        4. τ_cmd = τ_mpc + action_scale * δ   (clipped to torque_limit)
        5. MuJoCo forward step

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

        # MuJoCo model
        model_path = Path(cfg.env.model_path)
        self._model = mujoco.MjModel.from_xml_path(str(model_path))
        self._data  = mujoco.MjData(self._model)

        # Foot site IDs (used for Jacobian-based slip calculation)
        self._foot_site_ids = [
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, name)
            for name in _FOOT_SITE_NAMES
        ]

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
        self._tau_mpc       = np.zeros(NUM_JOINTS)
        self._cmd_vel       = np.array([0.5, 0.0, 0.0])  # [vx, vy, yaw_rate]

        # Gymnasium spaces
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
        mujoco.mj_resetData(self._model, self._data)
        self._scheduler.reset()
        self._step_count = 0
        self._tau_mpc    = np.zeros(NUM_JOINTS)

        if options and "cmd_vel" in options:
            self._cmd_vel = np.asarray(options["cmd_vel"], dtype=np.float64)
        else:
            # Randomise velocity command each episode for training generalisation
            self._cmd_vel = np.array([
                self.np_random.uniform(*_CMD_VX_RANGE),
                self.np_random.uniform(*_CMD_VY_RANGE),
                self.np_random.uniform(*_CMD_VYAW_RANGE),
            ])

        return self._get_obs().astype(np.float32), self._get_info()

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        # 1. Advance gait clock → get contact schedule for MPC horizon
        self._scheduler.step()
        contact_schedule = self._build_contact_schedule()

        # 2. MPC → reference torques based on current state + velocity command
        state = self._read_robot_state()
        desired_lin_vel = np.array([self._cmd_vel[0], self._cmd_vel[1], 0.0])
        self._tau_mpc = self._mpc.compute(
            state, contact_schedule, desired_lin_vel,
            desired_yaw_rate=float(self._cmd_vel[2]),
        )

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
        q    = self._data.qpos[7:19]               # (12) joint positions
        dq   = self._data.qvel[6:18]               # (12) joint velocities
        lv   = self._data.qvel[0:3]                # ( 3) base linear velocity
        av   = self._data.qvel[3:6]                # ( 3) base angular velocity
        grav = self._gravity_in_base()             # ( 3) gravity in base frame
        phi  = self._scheduler.clock.phase_obs()   # ( 8) sin/cos phases
        quat = self._data.qpos[3:7]                # ( 4) base orientation quaternion
        return np.concatenate([q, dq, lv, av, grav, phi, self._tau_mpc, quat, self._cmd_vel])

    def _is_terminated(self) -> bool:
        return float(self._data.qpos[2]) < 0.15   # body collapsed

    def _get_info(self) -> dict:
        return {
            "step":        self._step_count,
            "base_height": float(self._data.qpos[2]),
            "forward_vel": float(self._data.qvel[0]),
            "gait_phase":  float(self._scheduler.clock.phase),
            "cmd_vel":     self._cmd_vel.copy(),
        }

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _read_robot_state(self) -> RobotState:
        quat  = self._data.qpos[3:7]
        euler = self._quat_to_euler(quat)
        return RobotState(
            euler    = euler,
            position = self._data.qpos[0:3].copy(),
            ang_vel  = self._data.qvel[3:6].copy(),
            lin_vel  = self._data.qvel[0:3].copy(),
        )

    def _build_contact_schedule(self) -> np.ndarray:
        """Predict contact schedule over MPC horizon, returns (N, 4) bool array."""
        N            = self._cfg.mpc.horizon
        stance_ratio = self._scheduler.pattern.stance_ratio
        freq         = self._cfg.gait.frequency
        dt_mpc       = self._cfg.mpc.dt_mpc
        offsets      = self._scheduler.pattern.leg_offsets
        phase0       = self._scheduler.clock.phase

        schedule = np.zeros((N, 4), dtype=bool)
        for k in range(N):
            phase_k = (phase0 + freq * dt_mpc * k) % 1.0
            for i in range(4):
                leg_phase = (phase_k + offsets[i]) % 1.0
                schedule[k, i] = leg_phase < stance_ratio
        return schedule

    def _foot_tangential_velocities(self) -> np.ndarray:
        """Tangential (horizontal) speed of each foot, used as stance slip proxy.

        Returns: (4,) array, non-zero only for stance feet.
        """
        stance = self._scheduler.stance_mask()   # (4,) bool
        foot_vels = np.zeros(4)
        n_v = self._model.nv
        jacp = np.zeros((3, n_v))

        for i, sid in enumerate(self._foot_site_ids):
            if not stance[i]:
                continue
            jacp[:] = 0.0
            mujoco.mj_jacSite(self._model, self._data, jacp, None, sid)
            v_foot = jacp @ self._data.qvel   # (3,) foot velocity in world frame
            foot_vels[i] = float(np.linalg.norm(v_foot[:2]))   # horizontal speed
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
