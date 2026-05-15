# Source: jcwleo/random-network-distillation-pytorch (MIT)
# Patches: P1 (torch._six → torch.inf), P2 (global_grad_norm_ now actually clips)
from config import *
import numpy as np

import torch
from torch import inf  # P1: torch._six removed in PyTorch >=1.13

# if default_config['TrainMethod'] in ['PPO', 'ICM', 'RND']:
#     num_step = int(ppo_config['NumStep'])
# else:
#     num_step = int(default_config['NumStep'])

use_gae = default_config.getboolean('UseGAE')
lam = float(default_config['Lambda'])
train_method = default_config['TrainMethod']


def make_train_data(reward, done, value, gamma, num_step, num_worker):
    discounted_return = np.empty([num_worker, num_step])

    # Discounted Return
    if use_gae:
        gae = np.zeros_like([num_worker, ])
        for t in range(num_step - 1, -1, -1):
            delta = reward[:, t] + gamma * value[:, t + 1] * (1 - done[:, t]) - value[:, t]
            gae = delta + gamma * lam * (1 - done[:, t]) * gae

            discounted_return[:, t] = gae + value[:, t]

            # For Actor
        adv = discounted_return - value[:, :-1]

    else:
        running_add = value[:, -1]
        for t in range(num_step - 1, -1, -1):
            running_add = reward[:, t] + gamma * running_add * (1 - done[:, t])
            discounted_return[:, t] = running_add

        # For Actor
        adv = discounted_return - value[:, :-1]

    return discounted_return.reshape([-1]), adv.reshape([-1])


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


class RewardForwardFilter(object):
    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma

    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems


def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis]  # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]  # dito
    return e_x / div


def global_grad_norm_(parameters, max_norm, norm_type=2):
    """Clips gradient norm and returns the pre-clip total norm.

    P2 fix (jcwleo issue #17): the original function computed the norm but
    never rescaled gradients, so ClipGradNorm had no effect.  This wrapper
    delegates to torch.nn.utils.clip_grad_norm_ which clips in-place and
    returns the pre-clip norm.

    Args:
        parameters: iterable of Tensors with .grad populated
        max_norm:   clip threshold (pass self.clip_grad_norm from agent, e.g. 0.5)
        norm_type:  p-norm order (default L2)
    Returns:
        total norm before clipping (float tensor)
    """
    return torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type)
