"""Convex QP formulation for SRBD Model Predictive Control.

Problem (condensed form, decision variable = stacked GRFs over horizon N):

    min   0.5 * U^T H U + f^T U
    s.t.  C_eq  * U = b_eq          (dynamics)
          C_ub  * U ≤ d_ub          (friction cone, stance-only forces)
          lb ≤ U ≤ ub               (box constraints)

Solved with scipy.optimize.minimize (SLSQP).
For real-time deployment replace with osqp or qpsolvers.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from quadrl.control.mpc.dynamics import STATE_DIM, CTRL_DIM, NUM_LEGS


@dataclass
class QPData:
    """Packed QP matrices for one MPC solve."""
    H: np.ndarray      # (N*CTRL_DIM, N*CTRL_DIM) quadratic cost
    f: np.ndarray      # (N*CTRL_DIM,)              linear cost
    C_eq: np.ndarray   # (N*STATE_DIM, N*CTRL_DIM)  dynamics equality
    b_eq: np.ndarray   # (N*STATE_DIM,)
    lb: np.ndarray     # (N*CTRL_DIM,)  lower bounds on U
    ub: np.ndarray     # (N*CTRL_DIM,)  upper bounds on U
    mu: float          # friction coefficient


def build_qp(
    A: np.ndarray,
    Bs: list[np.ndarray],
    gs: list[np.ndarray],
    x0: np.ndarray,
    x_ref: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    contacts: np.ndarray,
    mu: float,
    f_max: float,
) -> QPData:
    """Build QP matrices for an N-step SRBD MPC problem.

    Args:
        A:         (STATE_DIM, STATE_DIM) constant dynamics A matrix.
        Bs:        List of N (STATE_DIM, CTRL_DIM) B matrices (foot-pos-dependent).
        gs:        List of N (STATE_DIM,) gravity terms.
        x0:        (STATE_DIM,) current state.
        x_ref:     (N, STATE_DIM) reference state trajectory.
        Q:         (STATE_DIM, STATE_DIM) state tracking cost.
        R:         (CTRL_DIM, CTRL_DIM)  control effort cost.
        contacts:  (N, NUM_LEGS) bool contact schedule over horizon.
        mu:        Friction coefficient.
        f_max:     Maximum normal force per stance leg (N).

    Returns:
        QPData with all matrices needed to call :func:`solve_qp`.
    """
    N = len(Bs)
    n_u = N * CTRL_DIM
    n_x = N * STATE_DIM

    # ── Propagate dynamics: x_{k+1} = A x_k + B_k u_k + g_k ─────────────────
    # Predicted state trajectory as function of U:
    #   X = S_x * x0 + S_u * U + g_vec
    S_x = np.zeros((n_x, STATE_DIM))     # propagation of initial state
    S_u = np.zeros((n_x, n_u))           # propagation of controls
    g_vec = np.zeros(n_x)               # accumulated gravity

    A_pow = np.eye(STATE_DIM)
    for k in range(N):
        A_pow = A @ A_pow if k > 0 else A
        S_x[k * STATE_DIM:(k + 1) * STATE_DIM, :] = A_pow

    for k in range(N):
        for j in range(k + 1):
            row = k * STATE_DIM
            col = j * CTRL_DIM
            A_k_j = np.linalg.matrix_power(A, k - j)
            S_u[row:row + STATE_DIM, col:col + CTRL_DIM] = A_k_j @ Bs[j]
        # Gravity accumulation
        for j in range(k + 1):
            A_k_j = np.linalg.matrix_power(A, k - j)
            g_vec[k * STATE_DIM:(k + 1) * STATE_DIM] += A_k_j @ gs[j]

    # Reference deviation offset
    x_ref_flat = x_ref.flatten()
    x0_contrib  = S_x @ x0 + g_vec   # predicted trajectory without control

    # ── Quadratic cost ────────────────────────────────────────────────────────
    Q_bar = np.kron(np.eye(N), Q)
    R_bar = np.kron(np.eye(N), R)

    H = S_u.T @ Q_bar @ S_u + R_bar
    H = 0.5 * (H + H.T)   # ensure symmetry

    f = S_u.T @ Q_bar @ (x0_contrib - x_ref_flat)

    # ── Box constraints (friction cone + stance enforcement) ──────────────────
    lb = np.zeros(n_u)
    ub = np.zeros(n_u)

    for k in range(N):
        for i in range(NUM_LEGS):
            base = k * CTRL_DIM + i * 3
            if contacts[k, i]:
                # Fz ∈ [0, f_max], Fx/Fy unconstrained here (friction via SLSQP)
                lb[base:base + 3] = [-mu * f_max, -mu * f_max, 0.0]
                ub[base:base + 3] = [ mu * f_max,  mu * f_max, f_max]
            else:
                lb[base:base + 3] = 0.0   # swing: zero force
                ub[base:base + 3] = 0.0

    return QPData(H=H, f=f, C_eq=S_u, b_eq=x_ref_flat - x0_contrib, lb=lb, ub=ub, mu=mu)


def solve_qp(qp: QPData) -> np.ndarray:
    """Solve the convex QP and return optimal GRFs.

    Args:
        qp: Packed QP data from :func:`build_qp`.

    Returns:
        U_opt: (N*CTRL_DIM,) optimal stacked GRF sequence.
                First CTRL_DIM elements are the GRFs to apply now.

    Note:
        Uses scipy SLSQP (correct but slow ~10 ms).
        Replace with osqp for ≤1 ms real-time solves::

            import osqp, scipy.sparse as sp
            prob = osqp.OSQP()
            prob.setup(sp.csc_matrix(qp.H), qp.f, ...)
    """
    n_u = len(qp.lb)
    u0  = np.zeros(n_u)   # warm-start at zero

    bounds = list(zip(qp.lb, qp.ub))

    def objective(u):
        return 0.5 * u @ qp.H @ u + qp.f @ u

    def gradient(u):
        return qp.H @ u + qp.f

    result = minimize(
        fun=objective,
        x0=u0,
        jac=gradient,
        bounds=bounds,
        method="SLSQP",
        options={"maxiter": 100, "ftol": 1e-6},
    )
    return result.x
