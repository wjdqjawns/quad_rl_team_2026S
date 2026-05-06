"""Convex QP formulation for SRBD Model Predictive Control.

Problem (condensed form, decision variable = stacked GRFs over horizon N):

    min   0.5 * U^T H U + f^T U
    s.t.  lb ≤ U ≤ ub               (box constraints — Fz≥0, swing=0)

Solved with OSQP (warm-starting, ~0.5 ms per solve).
Falls back to scipy SLSQP if OSQP is not installed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

try:
    import osqp as _osqp_mod
    _OSQP_AVAILABLE = True
except ImportError:
    _OSQP_AVAILABLE = False

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
    mass: float = 12.0,
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
    # Minimum Fz per stance leg prevents the solver from zeroing out one leg
    # of a diagonal pair (which creates asymmetric roll torques).
    lb = np.zeros(n_u)
    ub = np.zeros(n_u)

    for k in range(N):
        n_stance_k = int(np.sum(contacts[k]))
        fz_min_k   = (mass * 9.81 / max(n_stance_k, 1)) * 0.05  # 5% of equilibrium per leg

        for i in range(NUM_LEGS):
            base = k * CTRL_DIM + i * 3
            if contacts[k, i]:
                lb[base:base + 3] = [-mu * f_max, -mu * f_max, fz_min_k]
                ub[base:base + 3] = [ mu * f_max,  mu * f_max, f_max]
            else:
                lb[base:base + 3] = 0.0   # swing: zero force
                ub[base:base + 3] = 0.0

    return QPData(H=H, f=f, C_eq=S_u, b_eq=x_ref_flat - x0_contrib, lb=lb, ub=ub, mu=mu)


def solve_qp(qp: QPData, mass: float = 12.0) -> np.ndarray:
    """Solve the box-constrained QP using OSQP (fast) or SLSQP (fallback).

    Args:
        qp:   Packed QP data from :func:`build_qp`.
        mass: Robot mass (kg) — used for SLSQP fallback warm-start.

    Returns:
        U_opt: (N*CTRL_DIM,) optimal stacked GRF sequence.
    """
    if _OSQP_AVAILABLE:
        return _solve_osqp(qp)
    return _solve_slsqp(qp, mass)


def _solve_osqp(qp: QPData) -> np.ndarray:
    """Solve with OSQP.  Box constraints lb ≤ u ≤ ub encoded as I*u in [lb, ub]."""
    n_u = len(qp.lb)
    prob = _osqp_mod.OSQP()
    prob.setup(
        P=sp.csc_matrix(qp.H),
        q=qp.f,
        A=sp.eye(n_u, format="csc"),
        l=qp.lb,
        u=qp.ub,
        warm_starting=True,
        verbose=False,
        eps_abs=1e-5,
        eps_rel=1e-5,
        max_iter=4000,
        polish=True,
        polish_refine_iter=3,
    )
    res = prob.solve()
    if res.info.status_val in (1, 2):   # solved or solved_inaccurate
        return res.x
    # Fallback: clipped unconstrained optimum
    try:
        return np.clip(-np.linalg.solve(qp.H, qp.f), qp.lb, qp.ub)
    except np.linalg.LinAlgError:
        return np.clip(np.zeros(n_u), qp.lb, qp.ub)


def _solve_slsqp(qp: QPData, mass: float) -> np.ndarray:
    from scipy.optimize import minimize

    try:
        u0 = np.clip(-np.linalg.solve(qp.H, qp.f), qp.lb, qp.ub)
    except np.linalg.LinAlgError:
        u0 = np.zeros(len(qp.lb))
        n_stance = max(1, int(np.sum(qp.ub[2::3] > 0)))
        u0[2::3] = np.where(qp.ub[2::3] > 0, mass * 9.81 / n_stance, 0.0)

    result = minimize(
        fun=lambda u: 0.5 * u @ qp.H @ u + qp.f @ u,
        x0=u0,
        jac=lambda u: qp.H @ u + qp.f,
        bounds=list(zip(qp.lb, qp.ub)),
        method="SLSQP",
        options={"maxiter": 200, "ftol": 1e-8},
    )
    return result.x
