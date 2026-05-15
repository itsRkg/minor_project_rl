import torch
import torch.nn as nn


class RNDModule(nn.Module):
    """
    Random Network Distillation module.

    Contains:
    - frozen target network
    - trainable predictor network

    Input:
        h : (B, feature_dim)

    Output:
        intrinsic reward = prediction error
    """

    def __init__(
        self,
        feature_dim=256,
        rnd_dim=512,
    ):
        super().__init__()

        # -----------------------
        # TARGET NETWORK (FROZEN)
        # -----------------------
        self.target = nn.Sequential(
            nn.Linear(feature_dim, rnd_dim),
            nn.ReLU(),

            nn.Linear(rnd_dim, rnd_dim),
        )

        # -----------------------
        # PREDICTOR NETWORK
        # -----------------------
        self.predictor = nn.Sequential(
            nn.Linear(feature_dim, rnd_dim),
            nn.ReLU(),

            nn.Linear(rnd_dim, rnd_dim),
            nn.ReLU(),

            nn.Linear(rnd_dim, rnd_dim),
        )

        # Freeze target network
        for param in self.target.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def intrinsic_reward(self, h):
        """
        Compute intrinsic reward.

        r_int = ||target(h) - predictor(h)||²

        Args:
            h: (B, feature_dim)

        Returns:
            intrinsic reward: (B,)
        """

        target_features = self.target(h)
        pred_features = self.predictor(h)

        reward = (
            (target_features - pred_features)
            .pow(2)
            .mean(dim=-1)
        )

        return reward

    def loss(
        self,
        h,
        update_proportion=0.25,
    ):
        """
        RND predictor loss.

        Uses random masked updates to prevent
        predictor converging too quickly.

        Args:
            h: (B, feature_dim)

        Returns:
            scalar loss
        """

        target_features = self.target(h).detach()

        pred_features = self.predictor(h)

        per_sample_loss = (
            (target_features - pred_features)
            .pow(2)
            .mean(dim=-1)
        )

        # Random mask
        mask = (
            torch.rand(
                len(per_sample_loss),
                device=h.device,
            )
            < update_proportion
        ).float()

        mask_sum = torch.clamp(
            mask.sum(),
            min=1.0,
        )

        loss = (
            (per_sample_loss * mask).sum()
            / mask_sum
        )

        return loss