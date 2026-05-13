import torch
from torch.nn.utils import clip_grad_norm_


class PPOUpdater:
    def __init__(
        self,
        model,
        optimizer,
        clip_eps=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
    ):
        self.model = model
        self.optimizer = optimizer

        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

    def update(self, buffer, batch_size, n_epochs):
        total_loss = 0.0

        for _ in range(n_epochs):

            for batch in buffer.get_batches(batch_size):

                obs = batch["obs"]
                actions = batch["actions"]
                old_log_probs = batch["log_probs"]

                returns_ext = batch["returns_ext"]
                returns_int = batch["returns_int"]

                adv = batch["adv"]

                # =====================================================
                # Forward pass
                # =====================================================
                new_log_probs, entropy, v_ext, v_int = (
                    self.model.evaluate_actions(obs, actions)
                )

                # =====================================================
                # PPO Policy Loss
                # =====================================================
                ratio = torch.exp(new_log_probs - old_log_probs)

                unclipped = ratio * adv

                clipped = (
                    torch.clamp(
                        ratio,
                        1.0 - self.clip_eps,
                        1.0 + self.clip_eps,
                    )
                    * adv
                )

                policy_loss = -torch.min(unclipped, clipped).mean()

                # =====================================================
                # Dual Critic Value Loss
                # =====================================================
                value_loss_ext = (v_ext - returns_ext).pow(2).mean()

                value_loss_int = (v_int - returns_int).pow(2).mean()

                value_loss = value_loss_ext + value_loss_int

                # =====================================================
                # Entropy Bonus
                # =====================================================
                entropy_loss = -entropy.mean()

                # =====================================================
                # Total Loss
                # =====================================================
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                # =====================================================
                # Optimisation Step
                # =====================================================
                self.optimizer.zero_grad()

                loss.backward()

                # IMPORTANT FIX:
                # Gradient clipping stabilizes PPO training
                clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm,
                )

                self.optimizer.step()

                total_loss += loss.item()

        return total_loss