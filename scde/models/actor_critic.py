import torch
import torch.nn as nn
from torch.distributions import Categorical

from models.encoder import MiniCNN

class ActorCritic(nn.Module):
    def __init__(self, feature_dim=256, action_dim=7):
        super().__init__()

        # Shared encoder
        self.encoder = MiniCNN(feature_dim=feature_dim)

        # Actor head → logits
        self.actor = nn.Linear(feature_dim, action_dim)

        #  TWO CRITIC HEADS (IMPORTANT)
        self.critic_ext = nn.Linear(feature_dim, 1)   # V_ext
        self.critic_int = nn.Linear(feature_dim, 1)   # V_int

    def forward(self, obs):
        """
        obs: (B, H, W, C)
        """

        h = self.encoder(obs)   # (B, 256)

        logits = self.actor(h)  # (B, action_dim)

        v_ext = self.critic_ext(h).squeeze(-1)  # (B,)
        v_int = self.critic_int(h).squeeze(-1)  # (B,)

        return logits, v_ext, v_int, h

    def get_action(self, obs):
        """
        Used during rollout
        """

        logits, v_ext, v_int, h = self.forward(obs)

        dist = Categorical(logits=logits)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob, v_ext, v_int, h

    def evaluate_actions(self, obs, actions):
        """
        Used during PPO update
        """

        logits, v_ext, v_int, h = self.forward(obs)

        dist = Categorical(logits=logits)

        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_prob, entropy, v_ext, v_int

