import random
import numpy as np
import torch

from envs.vec_env import make_vec_env
from models.actor_critic import ActorCritic
from ppo.rollout_buffer import RolloutBuffer
from ppo.updater import PPOUpdater


class Trainer:
    def __init__(self, cfg):

        self.cfg = cfg

        # =========================================================
        # Reproducibility
        # =========================================================
        random.seed(cfg.seed)

        np.random.seed(cfg.seed)

        torch.manual_seed(cfg.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)

        # =========================================================
        # Device
        # =========================================================
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # =========================================================
        # Environment
        # =========================================================
        self.envs = make_vec_env(
            cfg.env_id,
            cfg.n_envs,
            seed=cfg.seed,
        )

        # =========================================================
        # Model
        # =========================================================
        self.model = ActorCritic(
            feature_dim=cfg.feature_dim,
            action_dim=cfg.action_dim,
        ).to(self.device)

        # =========================================================
        # Optimizer
        # =========================================================
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.lr,
        )

        # =========================================================
        # Rollout Buffer
        # =========================================================
        self.buffer = RolloutBuffer(
            n_steps=cfg.n_steps,
            n_envs=cfg.n_envs,
            device=self.device, # type: ignore
        )

        # =========================================================
        # PPO Updater
        # =========================================================
        self.updater = PPOUpdater(
            model=self.model,
            optimizer=self.optimizer,
            clip_eps=cfg.clip_eps,
            value_coef=cfg.value_coef,
            entropy_coef=cfg.entropy_coef,
            max_grad_norm=cfg.max_grad_norm,
        )

    def train(self):

        obs, _ = self.envs.reset()

        obs = torch.tensor(
            obs,
            dtype=torch.uint8,
            device=self.device,
        )

        total_steps = 0

        while total_steps < self.cfg.total_steps:

            self.buffer.reset()

            # =====================================================
            # Rollout Collection
            # =====================================================
            for _ in range(self.cfg.n_steps):

                with torch.no_grad():

                    (
                        action,
                        log_prob,
                        v_ext,
                        v_int,
                        _,
                    ) = self.model.get_action(obs)

                action_np = action.cpu().numpy()

                (
                    next_obs,
                    reward,
                    terminated,
                    truncated,
                    _,
                ) = self.envs.step(action_np)

                done = np.logical_or(
                    terminated,
                    truncated,
                )

                # ================================================
                # Convert to tensors
                # ================================================
                next_obs = torch.tensor(
                    next_obs,
                    dtype=torch.uint8,
                    device=self.device,
                )

                reward = torch.tensor(
                    reward,
                    dtype=torch.float32,
                    device=self.device,
                )

                done = torch.tensor(
                    done,
                    dtype=torch.bool,
                    device=self.device,
                )

                # ================================================
                # B1 PPO-only intrinsic reward
                # ================================================
                r_int = torch.zeros_like(reward)

                # ================================================
                # Store transition
                # ================================================
                self.buffer.add(
                    obs,
                    action,
                    log_prob,
                    reward,
                    r_int,
                    v_ext,
                    v_int,
                    done,
                )

                obs = next_obs

                total_steps += self.cfg.n_envs

            # =====================================================
            # Bootstrap final values
            # =====================================================
            with torch.no_grad():

                (
                    _,
                    last_v_ext,
                    last_v_int,
                    _,
                ) = self.model.forward(obs)

            # =====================================================
            # Compute GAE
            # =====================================================
            self.buffer.compute_returns_and_advantages(
                last_v_ext=last_v_ext,
                last_v_int=last_v_int,
                gamma_ext=self.cfg.gamma,
                gamma_int=self.cfg.gamma_int,
                lam=self.cfg.gae_lambda,
            )

            # =====================================================
            # PPO Update
            # =====================================================
            loss = self.updater.update(
                buffer=self.buffer,
                batch_size=self.cfg.batch_size,
                n_epochs=self.cfg.n_epochs,
            )

            # =====================================================
            # Logging
            # =====================================================
            print(
                f"Steps: {total_steps} | "
                f"Loss: {loss:.4f}"
            )