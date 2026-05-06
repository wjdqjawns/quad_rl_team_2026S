"""MPC Controller — main interface.

Responsibility (SRP):
    Given the current robot state and gait scheduler output, solve the
    SRBD convex MPC QP and return reference joint torques.
    Does NOT handle simulation, RL policy, or gait scheduling.

Residual RL interface:
    τ_cmd = τ_mpc + action_scale * τ_residual
    where τ_residual ∈ [-1, 1]^12 is the RL policy's output.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from quadrl.control.mpc.dynamics import (
    STATE_DIM,
    CTRL_DIM,
    NUM_LEGS,
    compute_A,
    compute_B,
    gravity_term,
)
from quadrl.control.mpc.qp_problem import build_qp, solve_qp
from quadrl.utils.config import MpcConfig


@dataclass
class RobotState:
    """Snapshot of body state required by the MPC (world frame)."""
    euler:      np.ndarray   # (3,) roll, pitch, yaw
    position:   np.ndarray   # (3,) CoM position
    ang_vel:    np.ndarray   # (3,) angular velocity
    lin_vel:    np.ndarray   # (3,) linear velocity

    def to_vec(self) -> np.ndarray:
        return np.concatenate([self.euler, self.position, self.ang_vel, self.lin_vel])


class MPCController:
    """Convex MPC based on Single Rigid Body Dynamics.

    Computes optimal ground reaction forces (GRFs) over a receding horizon,
    then converts them to joint torques via foot Jacobians.

    Args:
        cfg:   MPC configuration (horizon, weights, friction, …).
        model: MuJoCo MjModel (for mass, inertia, Jacobian queries).
        data:  MuJoCo MjData (updated externally before each call).
    """

    # MuJoCo site names — order defines leg index convention [FR, FL, RR, RL]
    _FOOT_SITE_NAMES = ["FR", "FL", "RR", "RL"]

    def __init__(self, cfg: MpcConfig, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self._cfg   = cfg
        self._model = model
        self._data  = data
        self._N     = cfg.horizon

        # Robot physical parameters (from MuJoCo model)
        self._mass = float(model.body_subtreemass[1])   # total mass (root body)
        self._A    = compute_A(cfg.dt_mpc)

        # Cost matrices
        self._Q = np.diag(cfg.state_weights)    # (STATE_DIM,)
        self._R = np.diag(cfg.ctrl_weights)     # (CTRL_DIM,)

        # Foot site IDs for Jacobian queries
        self._foot_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
            for name in self._FOOT_SITE_NAMES
        ]

    # ── Public interface ───────────────────────────────────────────────────────

    def compute_grfs(
        self,
        state: RobotState,
        contact_schedule: np.ndarray,              # (N, 4) bool stance mask
        desired_velocity: np.ndarray,              # (3,) [vx, vy, vz]
        desired_yaw_rate: float = 0.0,             # rad/s
        foot_positions: np.ndarray | None = None,  # (4, 3) optional planned feet
    ) -> np.ndarray:
        """Solve MPC QP and return first-step ground reaction forces.

        Args:
            state:             Current robot body state.
            contact_schedule:  Stance contact plan over horizon, shape (N, 4).
            desired_velocity:  Target body velocity [vx, vy, vz].
            desired_yaw_rate:  Target yaw rate (rad/s).
            foot_positions:    Optional (4, 3) planned foot positions in world
                               frame from a footstep planner. Falls back to
                               current MuJoCo foot positions when None.

        Returns:
            grfs: (4, 3) optimal GRFs in world frame (first MPC step).
        """
        foot_pos = foot_positions if foot_positions is not None else self._foot_positions()
        com_pos  = state.position
        I_w      = self._inertia_world(state.euler)

        Bs = [
            compute_B(I_w, foot_pos, com_pos, self._mass, self._cfg.dt_mpc, contact_schedule[k])
            for k in range(self._N)
        ]
        gs = [gravity_term(self._cfg.dt_mpc)] * self._N

        x_ref = self._build_reference(state, desired_velocity, desired_yaw_rate)

        qp = build_qp(self._A, Bs, gs, state.to_vec(), x_ref,
                      self._Q, self._R, contact_schedule,
                      self._cfg.friction_mu, self._cfg.f_max,
                      mass=self._mass)
        U  = solve_qp(qp, mass=self._mass)

        return U[:CTRL_DIM].reshape(NUM_LEGS, 3)

    def compute(
        self,
        state: RobotState,
        contact_schedule: np.ndarray,       # (N, 4) bool stance mask over horizon
        desired_velocity: np.ndarray,       # (3,) desired body linear velocity [vx, vy, vz]
        desired_yaw_rate: float = 0.0,      # desired body yaw rate (rad/s)
    ) -> np.ndarray:
        """Solve MPC QP and return reference joint torques.

        Args:
            state:             Current robot body state.
            contact_schedule:  Stance contact plan over the horizon, shape (N, 4).
            desired_velocity:  Target body velocity [vx, vy, vz].
            desired_yaw_rate:  Target yaw rate (rad/s).

        Returns:
            tau_mpc: Joint torques from MPC, shape (12,).
        """
        grfs = self.compute_grfs(state, contact_schedule, desired_velocity, desired_yaw_rate)
        return self._grfs_to_torques(grfs)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _foot_positions(self) -> np.ndarray:
        """World-frame foot positions from MuJoCo site data, shape (4, 3)."""
        return np.array([self._data.site_xpos[sid] for sid in self._foot_ids])

    def _inertia_world(self, euler: np.ndarray) -> np.ndarray:
        """Composite robot inertia in world frame from the full mass matrix.

        Uses the top-left 3×3 rotational block of the floating-base mass matrix
        (= full robot inertia, not just the trunk) for accurate SRBD dynamics.
        """
        from quadrl.control.mpc.dynamics import euler_to_rotation
        M = np.zeros((self._model.nv, self._model.nv))
        mujoco.mj_fullM(self._model, M, self._data.qM)
        # Rows/cols 3:6 are the rotational DOFs of the floating base (world frame)
        return M[3:6, 3:6]

    def _build_reference(
        self,
        state: RobotState,
        desired_velocity: np.ndarray,
        desired_yaw_rate: float = 0.0,
    ) -> np.ndarray:
        """Constant-velocity reference trajectory over horizon, shape (N, STATE_DIM)."""
        cfg    = self._cfg
        x_ref  = np.zeros((self._N, STATE_DIM))
        dt_mpc = cfg.dt_mpc

        for k in range(self._N):
            t = (k + 1) * dt_mpc
            # Orientation: zero roll/pitch; yaw integrates from commanded yaw rate
            x_ref[k, 0:3] = [0.0, 0.0, state.euler[2] + desired_yaw_rate * t]
            # Position: propagate from current at desired velocity
            x_ref[k, 3:6] = state.position + desired_velocity * t
            x_ref[k, 5]   = cfg.target_height
            # Angular velocity: commanded yaw rate only
            x_ref[k, 6:9] = [0.0, 0.0, desired_yaw_rate]
            # Linear velocity: desired
            x_ref[k, 9:12] = desired_velocity

        return x_ref

    def _grfs_to_torques(self, grfs: np.ndarray) -> np.ndarray:
        """Convert GRFs to joint torques via foot Jacobians: τ = -J^T F.

        Args:
            grfs: (4, 3) GRFs in world frame.

        Returns:
            tau: (12,) joint torques.
        """
        tau = np.zeros(12)
        n_v = self._model.nv

        for i, sid in enumerate(self._foot_ids):
            jacp  = np.zeros((3, n_v))
            jacr  = np.zeros((3, n_v))
            mujoco.mj_jacSite(self._model, self._data, jacp, jacr, sid)

            # Joint columns only (skip 6 floating-base DoFs)
            J_joints = jacp[:, 6:]   # (3, 12)
            tau += J_joints.T @ (-grfs[i])   # reaction: τ = -J^T F

        return np.clip(tau, -self._cfg.torque_limit, self._cfg.torque_limit)
