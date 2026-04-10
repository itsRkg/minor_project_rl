import torch
import numpy as np

from scde.envs.vec_env import make_vec_env
from scde.models.actor_critic import ActorCritic
from scde.ppo.rollout_buffer import RolloutBuffer
from scde.ppo.updater import PPOUpdater


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # -----------------------
        # ENV
        # -----------------------
        self.envs = make_vec_env(cfg.env_id, cfg.n_envs)

        # -----------------------
        # MODEL
        # -----------------------
        self.model = ActorCritic(
            feature_dim=cfg.feature_dim,
            action_dim=cfg.action_dim,
        ).to(self.device)

        # -----------------------
        # OPTIMIZER
        # -----------------------
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)

        # -----------------------
        # BUFFER
        # -----------------------
        self.buffer = RolloutBuffer(
            n_steps=cfg.n_steps,
            n_envs=cfg.n_envs,
            device=self.device, # pyright: ignore[reportArgumentType]
        )

        # -----------------------
        # PPO UPDATER
        # -----------------------
        self.updater = PPOUpdater(
            self.model,
            self.optimizer,
            clip_eps=cfg.clip_eps,
            value_coef=cfg.value_coef,
            entropy_coef=cfg.entropy_coef,
        )

    def train(self):
        obs, _ = self.envs.reset()
        obs = torch.tensor(obs, dtype=torch.uint8, device=self.device)

        total_steps = 0

        while total_steps < self.cfg.total_steps:

            self.buffer.reset()

            # -----------------------
            # ROLLOUT COLLECTION
            # -----------------------
            for _ in range(self.cfg.n_steps):

                with torch.no_grad():
                    action, log_prob, v_ext, v_int, _ = self.model.get_action(obs)

                # Convert action to numpy
                action_np = action.cpu().numpy()

                next_obs, reward, terminated, truncated, _ = self.envs.step(action_np)

                done = np.logical_or(terminated, truncated)

                # Convert to tensors
                next_obs = torch.tensor(next_obs, dtype=torch.uint8, device=self.device)
                reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
                done = torch.tensor(done, dtype=torch.float32, device=self.device)

                # B1 → intrinsic reward = 0
                r_int = torch.zeros_like(reward)

                # Store
                self.buffer.add(
                    obs,
                    action,
                    log_prob,
                    reward,   # r_ext
                    r_int,
                    v_ext,
                    v_int,
                    done,
                )

                obs = next_obs
                total_steps += self.cfg.n_envs

            # -----------------------
            # LAST VALUE (BOOTSTRAP)
            # -----------------------
            with torch.no_grad():
                _, last_v_ext, last_v_int, _ = self.model.forward(obs)

            # -----------------------
            # COMPUTE RETURNS + ADV
            # -----------------------
            self.buffer.compute_returns_and_advantages(
                last_v_ext,
                last_v_int,
                gamma=self.cfg.gamma,
                lam=self.cfg.gae_lambda,
            )

            # -----------------------
            # PPO UPDATE
            # -----------------------
            loss = self.updater.update(
                self.buffer,
                batch_size=self.cfg.batch_size,
                n_epochs=self.cfg.n_epochs,
            )

            # -----------------------
            # LOGGING
            # -----------------------
            print(f"Steps: {total_steps}, Loss: {loss:.4f}")