##----- test config loading -----##

# from utils.config import load_config

# cfg = load_config("configs/b5_proposed.yaml")

# print(cfg)
# print(cfg.use_rnd, cfg.use_clip, cfg.adaptive)

#------------------------------------------

#--- test vec env ---

# from envs.vec_env import make_vec_env

# envs = make_vec_env("MiniGrid-FourRooms-v0", n_envs=4)

# obs, _ = envs.reset()

# print("Obs shape:", obs.shape)

# import numpy as np
# actions = np.array([envs.single_action_space.sample() for _ in range(4)])

# obs, reward, terminated, truncated, info = envs.step(actions)

# print("Next obs:", obs.shape)
# print("Reward:", reward)
# print("Done:", terminated | truncated)


##----- test encoder -----##

# import torch
# from models.encoder import MiniCNN

# model = MiniCNN()

# dummy = torch.randint(0, 256, (4, 64, 64, 3), dtype=torch.uint8)

# out = model(dummy)

# print(out.shape)

##--- test actor-critic ---#
# import torch
# from models.actor_critic import ActorCritic

# model = ActorCritic()

# obs = torch.randint(0, 256, (4, 64, 64, 3), dtype=torch.uint8)

# action, log_prob, v_ext, v_int, h = model.get_action(obs)

# print("Action:", action.shape)
# print("Log prob:", log_prob.shape)
# print("V_ext:", v_ext.shape)
# print("V_int:", v_int.shape)

#------------------------
#    test rollout buffer and PPO updater together
#------------------------

from ppo.rollout_buffer import RolloutBuffer
import torch

buffer = RolloutBuffer(4, 2)

for _ in range(4):
    buffer.add(
        torch.zeros(2, 64, 64, 3, dtype=torch.uint8),
        torch.randint(0, 7, (2,)),
        torch.randn(2),
        torch.randn(2),
        torch.randn(2),
        torch.randn(2),
        torch.randn(2),
        torch.zeros(2)
    )

buffer.compute_returns_and_advantages(
    last_v_ext=torch.zeros(2),
    last_v_int=torch.zeros(2)
)

for batch in buffer.get_batches(4):
    print(batch["obs"].shape)


import torch
from models.actor_critic import ActorCritic
from ppo.updater import PPOUpdater

model = ActorCritic()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

updater = PPOUpdater(model, optimizer)

# fake batch
batch_size = 8

obs = torch.randint(0, 256, (batch_size, 64, 64, 3), dtype=torch.uint8)
actions = torch.randint(0, 7, (batch_size,))
log_probs = torch.randn(batch_size)

returns_ext = torch.randn(batch_size)
returns_int = torch.randn(batch_size)
adv = torch.randn(batch_size)

# simulate buffer
buffer = type("", (), {})()
buffer.get_batches = lambda _: [{ # pyright: ignore[reportAttributeAccessIssue]
    "obs": obs,
    "actions": actions,
    "log_probs": log_probs,
    "returns_ext": returns_ext,
    "returns_int": returns_int,
    "adv": adv
}]

loss = updater.update(buffer, batch_size=8, n_epochs=1)

print("Loss:", loss)





