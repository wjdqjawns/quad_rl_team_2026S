"""Nominal MPC controller for Go1 quadruped locomotion.

Corresponds to the "Nominal Controller" block in Fig. 2 of:
    Kim et al., "A Modular Residual Learning Framework to Enhance
    Model-Based Approach for Robust Locomotion," IEEE RA-L 2025.

Data flow per step:
    1. GaitScheduler  → contact schedule (N, 4) + stance/swing mask
    2. FootstepPlanner → planned foot positions (stance) + Cartesian swing targets
    3. MPCController  → optimal GRFs (4, 3) using planned foot positions
    4. LowLevelController → MuJoCo ctrl (12,) [position targets]

Stance legs use implicit torque control:  ctrl = q + τ_mpc / kp_actuator
Swing  legs use Cartesian PD:             ctrl = q + J^T*(Kp*e_p + Kd*e_v) / kp

The sin²(πs) swing trajectory ensures zero foot velocity at liftoff and
touchdown, preventing the impact impulse of the naive sin(πs) profile.

Residual RL interface (for future use):
    residual footstep  : δp_f per leg → added to foot_targets inside step()
    residual dynamics  : f_res        → passed as foot_positions offset to MPC
"""
from __future__ import annotations

import mujoco
import numpy as np

from quadrl.control.footstep_planner import FootstepPlanner
from quadrl.control.low_level_controller import LowLevelController
from quadrl.control.mpc.mpc_controller import MPCController, RobotState
from quadrl.gait.patterns.trot import TrotPattern
from quadrl.gait.scheduler import GaitScheduler
from quadrl.utils.config import Config


class NominalController:
    """MPC-based nominal locomotion controller.

    Args:
        cfg:   Full configuration (uses env, gait, mpc, control sections).
        model: MuJoCo model (read-only after init).
        data:  MuJoCo data (caller must advance the simulation externally).
    """

    def __init__(
        self,
        cfg: Config,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> None:
        self._cfg   = cfg
        self._model = model
        self._data  = data

        pattern = TrotPattern()

        self._scheduler = GaitScheduler(
            pattern=pattern,
            frequency=cfg.gait.frequency,
            dt=cfg.env.control_dt,
            step_height=cfg.gait.step_height,
        )

        self._mpc = MPCController(cfg.mpc, model, data)

        self._footstep = FootstepPlanner(
            model=model,
            data=data,
            stance_ratio=pattern.stance_ratio,
            gait_freq=cfg.gait.frequency,
            step_height=cfg.gait.step_height,
        )

        self._low_level = LowLevelController(
            model=model,
            data=data,
            kp_actuator=cfg.control.kp_actuator,
            torque_limit=cfg.control.torque_limit,
            kp_cart=cfg.control.kp_cart,
            kd_cart=cfg.control.kd_cart,
        )

        self._pattern = pattern

    # ── Public interface ────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset gait phase and liftoff state. Call after mujoco.mj_resetData()."""
        self._scheduler.reset()
        # Re-create footstep planner state so liftoff positions are fresh
        self._footstep._prev_stance = None
        self._footstep._liftoff_pos = np.zeros((4, 3))

    def step(self, cmd_vel: np.ndarray) -> np.ndarray:
        """Compute one control step.

        Args:
            cmd_vel: Velocity command [vx (m/s), vy (m/s), yaw_rate (rad/s)].

        Returns:
            ctrl: (12,) position targets for MuJoCo actuators.
        """
        # 1. Advance gait clock → stance/swing mask
        self._scheduler.step()
        stance_mask = self._scheduler.stance_mask()

        # 2. Read robot state
        state = self._read_state()

        # 3. Update liftoff positions (detect stance→swing transitions)
        self._footstep.update(stance_mask)

        # 4. Footstep planner → planned foot positions for MPC
        foot_targets = self._footstep.compute(
            com_pos=state.position,
            euler=state.euler,
            cmd_vel=cmd_vel,
            stance_mask=stance_mask,
        )

        # 5. Contact schedule over MPC horizon
        contact_schedule = self._build_contact_schedule()

        # 6. MPC → optimal GRFs using planned foot positions
        desired_vel = np.array([cmd_vel[0], cmd_vel[1], 0.0])
        grfs = self._mpc.compute_grfs(
            state=state,
            contact_schedule=contact_schedule,
            desired_velocity=desired_vel,
            desired_yaw_rate=float(cmd_vel[2]),
            foot_positions=foot_targets,
        )

        # 7. Cartesian swing targets for non-stance legs
        swing_pos, swing_vel = self._compute_swing_targets(
            stance_mask, state.position, state.euler, cmd_vel
        )

        # 8. Low-level controller → ctrl
        return self._low_level.compute(grfs, stance_mask, swing_pos, swing_vel)

    # ── Internal helpers ────────────────────────────────────────────────────────

    def _compute_swing_targets(
        self,
        stance_mask: np.ndarray,
        com_pos: np.ndarray,
        euler: np.ndarray,
        cmd_vel: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute desired foot state for all swing legs.

        Returns:
            swing_pos: (4, 3) desired foot positions (zeros for stance legs).
            swing_vel: (4, 3) desired foot velocities (zeros for stance legs).
        """
        swing_pos = np.zeros((4, 3))
        swing_vel = np.zeros((4, 3))

        phase   = self._scheduler.clock.phase
        offsets = self._pattern.leg_offsets
        sr      = self._pattern.stance_ratio

        for i in range(4):
            if stance_mask[i]:
                continue
            leg_phase = (phase + offsets[i]) % 1.0
            s = (leg_phase - sr) / (1.0 - sr)   # normalised swing progress ∈ [0,1]
            s = np.clip(s, 0.0, 1.0)

            p_des, v_des = self._footstep.swing_trajectory(
                leg_idx=i,
                swing_progress=s,
                com_pos=com_pos,
                euler=euler,
                cmd_vel=cmd_vel,
            )
            swing_pos[i] = p_des
            swing_vel[i] = v_des

        return swing_pos, swing_vel

    def _read_state(self) -> RobotState:
        quat = self._data.qpos[3:7]
        w, x, y, z = quat
        roll  = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
        yaw   = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return RobotState(
            euler    = np.array([roll, pitch, yaw]),
            position = self._data.qpos[0:3].copy(),
            ang_vel  = self._data.qvel[3:6].copy(),
            lin_vel  = self._data.qvel[0:3].copy(),
        )

    def _build_contact_schedule(self) -> np.ndarray:
        cfg          = self._cfg
        N            = cfg.mpc.horizon
        stance_ratio = self._pattern.stance_ratio
        freq         = cfg.gait.frequency
        dt_mpc       = cfg.mpc.dt_mpc
        offsets      = self._pattern.leg_offsets
        phase0       = self._scheduler.clock.phase

        schedule = np.zeros((N, 4), dtype=bool)
        for k in range(N):
            phase_k = (phase0 + freq * dt_mpc * k) % 1.0
            for i in range(4):
                leg_phase = (phase_k + offsets[i]) % 1.0
                schedule[k, i] = leg_phase < stance_ratio
        return schedule
