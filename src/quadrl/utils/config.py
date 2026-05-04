from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# ── Per-section dataclasses ────────────────────────────────────────────────────
@dataclass
class EnvConfig:
    model_path: str         = "asset/unitree_go1/go1.xml"
    sim_dt: float           = 0.002    # MuJoCo step (500 Hz)
    control_dt: float       = 0.02     # policy / controller step (50 Hz)
    max_episode_steps: int  = 1000
    render_mode: str | None = None     # None | "human" | "rgb_array"

@dataclass
class GaitConfig:
    pattern: str        = "trot"
    frequency: float    = 2.0          # Hz
    stance_ratio: float = 0.6          # fraction of cycle in stance
    step_height: float  = 0.08         # m

@dataclass
class ControlConfig:
    action_scale: float = 5.0          # residual torque scale (Nm)
    torque_limit: float = 33.5         # Nm per joint (hardware limit)

@dataclass
class MpcConfig:
    horizon: int              = 10     # prediction steps N
    dt_mpc: float             = 0.02   # MPC step (= control_dt)
    target_height: float      = 0.30   # desired CoM height (m)
    friction_mu: float        = 0.6    # friction cone coefficient
    f_max: float              = 200.0  # max normal GRF per leg (N)
    torque_limit: float       = 33.5   # Nm per joint (for GRF→τ clipping)
    # State cost weights: [roll, pitch, yaw, px, py, pz, ωx, ωy, ωz, vx, vy, vz]
    state_weights: list[float] = field(
        default_factory=lambda: [50., 50., 0., 0., 0., 100., 0., 0., 10., 10., 10., 0.]
    )
    # Control cost weights: [Fx, Fy, Fz] × 4 legs
    ctrl_weights: list[float] = field(
        default_factory=lambda: [1e-4, 1e-4, 1e-4] * 4
    )

@dataclass
class NetworkConfig:
    hidden_sizes: list[int] = field(default_factory=lambda: [256, 256])
    activation: str         = "elu"    # elu | tanh | relu

@dataclass
class TrainingConfig:
    algorithm: str      = "ppo"
    total_timesteps: int = 10_000_000
    n_envs: int         = 1
    n_steps: int        = 2048         # rollout steps per update
    n_epochs: int       = 10
    batch_size: int     = 64
    lr: float           = 3.0e-4
    gamma: float        = 0.99
    gae_lambda: float   = 0.95
    clip_eps: float     = 0.2
    value_coef: float   = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    checkpoint_freq: int = 100_000     # save every N env steps

@dataclass
class LoggingConfig:
    level: str       = "INFO"
    export_dir: str  = "export/temp/log"
    tensorboard: bool = True
    csv: bool         = True
    run_name: str | None = None

@dataclass
class Config:
    env:      EnvConfig      = field(default_factory=EnvConfig)
    gait:     GaitConfig     = field(default_factory=GaitConfig)
    control:  ControlConfig  = field(default_factory=ControlConfig)
    mpc:      MpcConfig      = field(default_factory=MpcConfig)
    network:  NetworkConfig  = field(default_factory=NetworkConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging:  LoggingConfig  = field(default_factory=LoggingConfig)


# ── Internal helpers ───────────────────────────────────────────────────────────
def _merge_into(obj: Any, data: dict) -> None:
    """Recursively overwrite dataclass fields from a plain dict."""
    for key, value in data.items():
        if not hasattr(obj, key):
            continue
        current = getattr(obj, key)
        if isinstance(current, (int, float, str, bool, list, type(None))):
            setattr(obj, key, value)
        elif isinstance(current, dict):
            setattr(obj, key, value)
        else:
            # nested dataclass
            _merge_into(current, value)


# ── Public API ─────────────────────────────────────────────────────────────────
def load_config(path: str | Path) -> Config:
    """Load a YAML file and merge it onto default Config values.

    Missing keys fall back to dataclass defaults, so a minimal YAML
    (e.g., only overriding ``training.lr``) is perfectly valid.

    Args:
        path: Path to the YAML config file.

    Returns:
        A fully-populated :class:`Config` instance.
    """
    cfg = Config()
    with open(path, "r", encoding="utf-8") as f:
        data: dict = yaml.safe_load(f) or {}

    for section_key, section_data in data.items():
        section_obj = getattr(cfg, section_key, None)
        if section_obj is not None and isinstance(section_data, dict):
            _merge_into(section_obj, section_data)

    return cfg