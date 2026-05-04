"""SB3-based training orchestrator for residual RL.

Uses Stable Baselines3 PPO with a custom MLP policy.
The Go1Env is a standard gymnasium.Env — no SB3-specific changes needed in the env.

Usage:
    trainer = Trainer(cfg)
    trainer.train()
"""
from __future__ import annotations

from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from quadrl.envs.go1_env import Go1Env
from quadrl.utils.config import Config
from quadrl.utils.logger import get_logger

_log = get_logger("training.trainer")


class Trainer:
    """Orchestrates SB3 PPO training on Go1Env.

    Responsibility (SRP):
        Wire SB3, environment, and callbacks together.
        All RL math is delegated to SB3.
    """

    def __init__(self, cfg: Config) -> None:
        self._cfg     = cfg
        self._out_dir = Path(cfg.logging.export_dir)
        self._ckpt_dir = self._out_dir / "checkpoints"
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)

        # ── Environment ───────────────────────────────────────────────────────
        # Use SubprocVecEnv for n_envs > 1, DummyVecEnv for debugging
        n_envs = cfg.training.n_envs
        VecEnvCls = SubprocVecEnv if n_envs > 1 else DummyVecEnv

        self._env = make_vec_env(
            lambda: Go1Env(cfg),
            n_envs=n_envs,
            vec_env_cls=VecEnvCls,
        )

        # Separate evaluation env (single, not vectorised)
        self._eval_env = DummyVecEnv([lambda: Go1Env(cfg)])

        # ── SB3 PPO ───────────────────────────────────────────────────────────
        t = cfg.training
        self._model = PPO(
            policy="MlpPolicy",
            env=self._env,
            n_steps=t.n_steps,
            batch_size=t.batch_size,
            n_epochs=t.n_epochs,
            learning_rate=t.lr,
            gamma=t.gamma,
            gae_lambda=t.gae_lambda,
            clip_range=t.clip_eps,
            vf_coef=t.value_coef,
            ent_coef=t.entropy_coef,
            max_grad_norm=t.max_grad_norm,
            tensorboard_log=str(self._out_dir) if cfg.logging.tensorboard else None,
            policy_kwargs=dict(
                net_arch=dict(
                    pi=cfg.network.hidden_sizes,
                    vf=cfg.network.hidden_sizes,
                ),
                activation_fn=_get_activation(cfg.network.activation),
            ),
            verbose=1,
        )

        # ── Callbacks ─────────────────────────────────────────────────────────
        self._callbacks = [
            CheckpointCallback(
                save_freq=max(t.checkpoint_freq // n_envs, 1),
                save_path=str(self._ckpt_dir),
                name_prefix="ppo_go1",
            ),
            EvalCallback(
                self._eval_env,
                best_model_save_path=str(self._ckpt_dir / "best"),
                log_path=str(self._out_dir / "eval"),
                eval_freq=max(t.checkpoint_freq // n_envs, 1),
                n_eval_episodes=5,
                deterministic=True,
            ),
        ]

    # ── Public interface ───────────────────────────────────────────────────────
    def train(self) -> None:
        """Run PPO training to completion."""
        run_name = self._cfg.logging.run_name or "ppo_go1_residual"
        _log.info("Training started — %s", run_name)

        self._model.learn(
            total_timesteps=self._cfg.training.total_timesteps,
            callback=self._callbacks,
            tb_log_name=run_name,
            reset_num_timesteps=True,
        )

        # Save final model
        final_path = self._ckpt_dir / "final_model"
        self._model.save(str(final_path))
        _log.info("Training complete. Model saved to %s", final_path)

    def load_and_continue(self, checkpoint_path: str | Path) -> None:
        """Resume training from a checkpoint."""
        self._model = PPO.load(
            str(checkpoint_path),
            env=self._env,
            tensorboard_log=str(self._out_dir),
        )
        _log.info("Resumed from checkpoint: %s", checkpoint_path)
        self.train()


# ── Helpers ────────────────────────────────────────────────────────────────────
def _get_activation(name: str):
    import torch.nn as nn
    acts = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}
    if name not in acts:
        raise ValueError(f"Unknown activation '{name}'. Choose from {list(acts)}")
    return acts[name]