import torch
import torch.nn as nn

class MiniCNN(nn.Module):
    def __init__(self, in_channels=3, feature_dim=256):
        super().__init__()

        self.net = nn.Sequential(
            # (B, 3, 64, 64) → (B, 32, 32, 32)
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),

            # (B, 32, 32, 32) → (B, 64, 16, 16)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),

            # (B, 64, 16, 16) → (B, 64, 8, 8)
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),

            # (B, 64, 8, 8) → (B, 64, 4, 4)
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),

            nn.Flatten(),  # (B, 1024)

            nn.Linear(64 * 4 * 4, feature_dim),
            nn.ELU(),
        )

    def forward(self, obs):
        """
        obs: (B, H, W, C) uint8
        """

        # Convert to float and normalize
        x = obs.float() / 255.0

        # Change to channel-first
        x = x.permute(0, 3, 1, 2)

        return self.net(x)

