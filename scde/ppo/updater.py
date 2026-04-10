import torch


class PPOUpdater:
    def __init__(
        self,
        model,
        optimizer,
        clip_eps=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
    ):
        self.model = model
        self.optimizer = optimizer

        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

    def update(self, buffer, batch_size, n_epochs):
        total_loss = 0

        for _ in range(n_epochs):
            for batch in buffer.get_batches(batch_size):

                obs = batch["obs"]
                actions = batch["actions"]
                old_log_probs = batch["log_probs"]

                returns_ext = batch["returns_ext"]
                returns_int = batch["returns_int"]
                adv = batch["adv"]

                # Forward pass
                new_log_probs, entropy, v_ext, v_int = self.model.evaluate_actions(
                    obs, actions
                )

                # -----------------------
                # 1. POLICY LOSS
                # -----------------------
                ratio = torch.exp(new_log_probs - old_log_probs)

                unclipped = ratio * adv
                clipped = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv

                policy_loss = -torch.min(unclipped, clipped).mean()

                # -----------------------
                # 2. VALUE LOSS (DUAL)
                # -----------------------
                value_loss_ext = (v_ext - returns_ext).pow(2).mean()
                value_loss_int = (v_int - returns_int).pow(2).mean()

                value_loss = value_loss_ext + value_loss_int

                # -----------------------
                # 3. ENTROPY BONUS
                # -----------------------
                entropy_loss = -entropy.mean()

                # -----------------------
                # TOTAL LOSS
                # -----------------------
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                # -----------------------
                # OPTIM STEP
                # -----------------------
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

        return total_loss