import torch
import numpy as np

from envs.vec_env import make_vec_env

from models.actor_critic import ActorCritic
from models.rnd import RNDModule

from ppo.rollout_buffer import RolloutBuffer
from ppo.updater import PPOUpdater


class Trainer:
    def __init__(self, cfg):

        self.cfg = cfg

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        # =====================================================
        # ENVIRONMENT
        # =====================================================

        self.envs = make_vec_env(
            cfg.env_id,
            cfg.n_envs,
        )

        # =====================================================
        # MODEL
        # =====================================================

        self.model = ActorCritic(
            feature_dim=cfg.feature_dim,
            action_dim=cfg.action_dim,
        ).to(self.device)

        # =====================================================
        # OPTIONAL RND MODULE
        # =====================================================

        self.rnd = None

        if cfg.use_rnd:

            self.rnd = RNDModule(
                feature_dim=cfg.feature_dim,
                rnd_dim=cfg.rnd_dim,
            ).to(self.device)

        # =====================================================
        # OPTIMIZER
        # =====================================================

        params = list(
            self.model.parameters()
        )

        # Only predictor is trainable
        if self.rnd is not None:

            params += list(
                self.rnd.predictor.parameters()
            )

        self.optimizer = torch.optim.Adam(
            params,
            lr=cfg.lr,
        )

        # =====================================================
        # BUFFER
        # =====================================================

        self.buffer = RolloutBuffer(
            n_steps=cfg.n_steps,
            n_envs=cfg.n_envs,
            device=self.device, # type: ignore
        )

        # =====================================================
        # PPO UPDATER
        # =====================================================

        self.updater = PPOUpdater(
            model=self.model,
            optimizer=self.optimizer,
            rnd=self.rnd,
            clip_eps=cfg.clip_eps,
            value_coef=cfg.value_coef,
            entropy_coef=cfg.entropy_coef,
            rnd_update_proportion=(
                cfg.rnd_update_proportion
            ),
        )

    def train(self):

        obs, _ = self.envs.reset()

        obs = torch.tensor(
            obs,
            dtype=torch.uint8,
            device=self.device,
        )

        # =====================================================
        # TRACKING VARIABLES
        # =====================================================

        total_steps = 0

        # Running episode trackers
        episode_returns = np.zeros(
            self.cfg.n_envs,
            dtype=np.float32,
        )

        episode_lengths = np.zeros(
            self.cfg.n_envs,
            dtype=np.int32,
        )

        # Historical metrics
        all_returns = []
        all_lengths = []
        all_successes = []

        # =====================================================
        # TRAINING LOOP
        # =====================================================

        while total_steps < self.cfg.total_steps:

            self.buffer.reset()

            # =================================================
            # ROLLOUT COLLECTION
            # =================================================

            for _ in range(self.cfg.n_steps):

                with torch.no_grad():

                    (
                        action,
                        log_prob,
                        v_ext,
                        v_int,
                        h,
                    ) = self.model.get_action(obs)

                    # =========================================
                    # OPTIONAL RND REWARD
                    # =========================================

                    if self.rnd is not None:

                        r_int = (
                            self.rnd
                            .intrinsic_reward(h)
                        )

                    else:

                        r_int = torch.zeros(
                            self.cfg.n_envs,
                            device=self.device,
                        )

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

                # =================================================
                # METRICS
                # =================================================

                episode_returns += reward
                episode_lengths += 1

                for i in range(self.cfg.n_envs):

                    if done[i]:

                        # =========================================
                        # STORE EPISODE METRICS
                        # =========================================

                        ep_return = episode_returns[i]
                        ep_length = episode_lengths[i]

                        all_returns.append(ep_return)
                        all_lengths.append(ep_length)

                        # =========================================
                        # SPARSE REWARD SUCCESS METRIC
                        # =========================================

                        success = (
                            1 if ep_return > 0 else 0
                        )

                        all_successes.append(success)

                        # =========================================
                        # RESET TRACKERS
                        # =========================================

                        episode_returns[i] = 0
                        episode_lengths[i] = 0

                # =================================================
                # CONVERT TO TENSORS
                # =================================================

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

                done_t = torch.tensor(
                    done,
                    dtype=torch.bool,
                    device=self.device,
                )

                # =================================================
                # STORE TRANSITION
                # =================================================

                self.buffer.add(
                    obs,
                    action,
                    log_prob,
                    reward,
                    r_int,
                    v_ext,
                    v_int,
                    h,
                    done_t,
                )

                obs = next_obs

                total_steps += self.cfg.n_envs

            # =====================================================
            # BOOTSTRAP VALUES
            # =====================================================

            with torch.no_grad():

                (
                    _,
                    last_v_ext,
                    last_v_int,
                    _,
                ) = self.model.forward(obs)

            # =====================================================
            # COMPUTE RETURNS + ADVANTAGES
            # =====================================================

            self.buffer.compute_returns_and_advantages(
                last_v_ext,
                last_v_int,
                gamma=self.cfg.gamma,
                gamma_int=self.cfg.gamma_int,
                lam=self.cfg.gae_lambda,
            )

            # =====================================================
            # PPO UPDATE
            # =====================================================

            loss = self.updater.update(
                self.buffer,
                batch_size=self.cfg.batch_size,
                n_epochs=self.cfg.n_epochs,
            )

            # =====================================================
            # LOGGING
            # =====================================================

            if len(all_returns) > 0:

                mean_return = np.mean(
                    all_returns[-10:]
                )

                mean_ep_len = np.mean(
                    all_lengths[-10:]
                )

                # Last 100 completed episodes
                success_rate = np.mean(
                    all_successes[-100:]
                )

                print(
                    f"Steps: {total_steps} | "
                    f"Loss: {loss:.3f} | "
                    f"Mean Return: {mean_return:.2f} | "
                    f"Mean Ep Len: {mean_ep_len:.1f} | "
                    f"Success Rate: {success_rate:.2f}"
                )

        print("Training completed.")