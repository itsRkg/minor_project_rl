# Source: jcwleo/random-network-distillation-pytorch (MIT)
# Patches applied:
#   P6  – tensorboardX → torch.utils.tensorboard
#   P7  – argparse + --resume / --config / --total-frames / --chunk-frames
#          is_load_model no longer hardcoded True (would crash on first run)
#   P8  – full-state checkpoint (model + optimizer + RMS stats + RNG)
#          atomic write via tmp-then-rename; keep last-3 + milestones
# Additional: train_one_chunk() callable from notebooks; metrics.jsonl logging;
#             eval_checkpoint() with RecordVideo at chunk boundaries.

import os
import sys
import json
import time
import random
import shutil
import hashlib
import argparse
import csv
from collections import deque

import numpy as np
import torch
from torch.multiprocessing import Pipe
from torch.utils.tensorboard import SummaryWriter  # P6

# ── local imports ─────────────────────────────────────────────────────────────
# Insert rnd_baseline/ dir so imports work from any working directory
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from agents import RNDAgent
from envs import AtariEnvironment
from utils import RunningMeanStd, RewardForwardFilter, make_train_data, softmax
from config import config, default_config


# ─────────────────────────────────────────────────────────────────────────────
# P7 – Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='PPO+RND training — rnd_baseline')
    parser.add_argument('--resume', type=str, default='',
                        help='Path to a checkpoint .pt to resume from. '
                             'Pass "auto" to load latest.pt automatically. '
                             'Empty string = start fresh.')
    parser.add_argument('--config', type=str, default='DEFAULT',
                        help='Config section name in config.conf '
                             '(e.g. DEFAULT, MINIGRID)')
    parser.add_argument('--total-frames', type=int, default=50_000_000,
                        help='Total training frames across all chunks')
    parser.add_argument('--chunk-frames', type=int, default=2_000_000,
                        help='Frames per chunk before checkpointing and exiting')
    parser.add_argument('--run-name', type=str, default='run_001',
                        help='Run name; outputs go to runs/<run-name>/')
    parser.add_argument('--seed', type=int, default=42,
                        help='Global random seed')
    parser.add_argument('--log-interval', type=int, default=50,
                        help='Log metrics every N updates')
    return parser.parse_args(argv)


# ─────────────────────────────────────────────────────────────────────────────
# P8 – Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ckpt_path(ckpt_dir, global_step):
    return os.path.join(ckpt_dir, f'chunk_{global_step:011d}.pt')


def _config_hash():
    conf_path = os.path.join(_HERE, 'config.conf')
    return hashlib.sha1(open(conf_path, 'rb').read()).hexdigest()


def save_checkpoint(ctx):
    """Atomically save full training state (P8).

    Payload:
      global_step, global_update, model, rnd, optimizer,
      obs_rms, reward_rms, reward_filter, rng states,
      config_section, config_hash, best_extrinsic_so_far
    """
    ckpt_dir   = ctx.ckpt_dir
    tmp_path   = _ckpt_path(ckpt_dir, ctx.global_step) + '.tmp'
    final_path = _ckpt_path(ckpt_dir, ctx.global_step)

    payload = {
        'global_step':   ctx.global_step,
        'global_update': ctx.global_update,
        'model':         ctx.agent.model.state_dict(),
        'rnd': {
            'predictor': ctx.agent.rnd.predictor.state_dict(),
            'target':    ctx.agent.rnd.target.state_dict(),
        },
        'optimizer':     ctx.agent.optimizer.state_dict(),
        'obs_rms': {
            'mean':  ctx.obs_rms.mean,
            'var':   ctx.obs_rms.var,
            'count': ctx.obs_rms.count,
        },
        'reward_rms': {
            'mean':  ctx.reward_rms.mean,
            'var':   ctx.reward_rms.var,
            'count': ctx.reward_rms.count,
        },
        'reward_filter': {
            'rewems': ctx.reward_filter.rewems,
            'gamma':  ctx.reward_filter.gamma,
        },
        'rng': {
            'numpy':      np.random.get_state(),
            'python':     random.getstate(),
            'torch':      torch.get_rng_state(),
            'torch_cuda': (torch.cuda.get_rng_state()
                           if torch.cuda.is_available() else None),
        },
        'config_section':        ctx.args.config,
        'config_hash':           _config_hash(),
        'best_extrinsic_so_far': ctx.best_ext_so_far,
    }

    torch.save(payload, tmp_path)
    os.replace(tmp_path, final_path)   # atomic rename

    # keep latest.pt as a plain copy (avoids symlink issues on Windows)
    latest_path = os.path.join(ckpt_dir, 'latest.pt')
    shutil.copy2(final_path, latest_path)

    _prune_checkpoints(ckpt_dir, ctx.global_step)
    print(f'[ckpt] saved → {final_path}')


def load_checkpoint(resume_path, ctx):
    """Restore full training state from a checkpoint (P8)."""
    ckpt = torch.load(resume_path, map_location='cpu')

    ctx.agent.model.load_state_dict(ckpt['model'])
    ctx.agent.rnd.predictor.load_state_dict(ckpt['rnd']['predictor'])
    ctx.agent.rnd.target.load_state_dict(ckpt['rnd']['target'])
    ctx.agent.optimizer.load_state_dict(ckpt['optimizer'])

    ctx.obs_rms.mean  = ckpt['obs_rms']['mean']
    ctx.obs_rms.var   = ckpt['obs_rms']['var']
    ctx.obs_rms.count = ckpt['obs_rms']['count']

    ctx.reward_rms.mean  = ckpt['reward_rms']['mean']
    ctx.reward_rms.var   = ckpt['reward_rms']['var']
    ctx.reward_rms.count = ckpt['reward_rms']['count']

    ctx.reward_filter.rewems = ckpt['reward_filter']['rewems']

    rng = ckpt['rng']
    np.random.set_state(rng['numpy'])
    random.setstate(rng['python'])
    torch.set_rng_state(rng['torch'])
    if rng['torch_cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(rng['torch_cuda'])

    ctx.global_step      = ckpt['global_step']
    ctx.global_update    = ckpt['global_update']
    ctx.best_ext_so_far  = ckpt.get('best_extrinsic_so_far', -1e9)
    print(f'[ckpt] resumed from step {ctx.global_step:,}  ({resume_path})')


def _prune_checkpoints(ckpt_dir, current_step,
                       keep_last=3, milestone_every=10_000_000):
    """Delete old chunk files; keep last N + every M-frame milestone."""
    import glob
    chunks = sorted(glob.glob(os.path.join(ckpt_dir, 'chunk_*.pt')))
    milestones = set()
    for m in range(0, current_step + milestone_every, milestone_every):
        milestones.add(m)

    keep = set(chunks[-keep_last:])
    for p in chunks:
        step = int(os.path.basename(p)
                   .replace('chunk_', '').replace('.pt', ''))
        if step in milestones:
            keep.add(p)

    for p in chunks:
        if p not in keep:
            try:
                os.remove(p)
            except OSError:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# Training context  (holds all mutable state; passed between notebook cells)
# ─────────────────────────────────────────────────────────────────────────────

class TrainingContext:
    """Bag of all mutable training state.  Notebook Cell 2 creates this via
    setup_training(); Cell 3 passes it into train_one_chunk() each iteration.
    """
    def __init__(self):
        self.args            = None
        self.agent           = None
        self.obs_rms         = None
        self.reward_rms      = None
        self.reward_filter   = None
        self.works           = []
        self.parent_conns    = []
        self.states          = None
        self.output_size     = None
        self.global_step     = 0
        self.global_update   = 0
        self.best_ext_so_far = -1e9
        self.writer          = None
        self.log_fh          = None
        self.metrics_fh      = None
        self.ckpt_dir        = None
        self.video_dir       = None
        self.run_dir         = None
        self._hp             = {}
        # rolling buffers for metric calculation
        self.recent_ext_returns = deque(maxlen=100)
        self.recent_ep_lengths  = deque(maxlen=100)
        self.recent_int_rewards = deque(maxlen=1000)

    def log(self, msg):
        ts   = time.strftime('%Y-%m-%d %H:%M:%S')
        line = f'[{ts}] {msg}'
        print(line)
        if self.log_fh:
            self.log_fh.write(line + '\n')


# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────

def _set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_training(args):
    """Initialise everything for a training run.  Returns a TrainingContext.

    If args.resume points to a valid checkpoint, loads it and skips the
    obs-RMS warmup (warmup already baked into the checkpoint).
    If args.resume == 'auto', loads runs/<run_name>/checkpoints/latest.pt
    if it exists, otherwise starts fresh.
    """
    ctx      = TrainingContext()
    ctx.args = args

    # ── resolve config section ────────────────────────────────────────────
    cfg = config[args.config]

    env_id   = cfg['EnvID']
    env_type = cfg['EnvType']

    # ── seed ─────────────────────────────────────────────────────────────
    _set_seeds(args.seed)

    # ── probe env for shapes ──────────────────────────────────────────────
    import gym
    env_probe   = gym.make(env_id)
    input_size  = env_probe.observation_space.shape
    output_size = env_probe.action_space.n
    if 'Breakout' in env_id:
        output_size -= 1
    env_probe.close()
    ctx.output_size = output_size

    # ── run directories (P7) ──────────────────────────────────────────────
    run_dir   = os.path.join(_HERE, 'runs', args.run_name)
    ckpt_dir  = os.path.join(run_dir, 'checkpoints')
    log_dir   = os.path.join(run_dir, 'logs')
    video_dir = os.path.join(run_dir, 'videos')
    for d in (ckpt_dir, log_dir, video_dir):
        os.makedirs(d, exist_ok=True)
    ctx.run_dir   = run_dir
    ctx.ckpt_dir  = ckpt_dir
    ctx.video_dir = video_dir

    # ── TensorBoard + file logging (P6) ──────────────────────────────────
    tb_dir      = os.path.join(log_dir, 'tensorboard')
    ctx.writer  = SummaryWriter(log_dir=tb_dir)
    ctx.log_fh  = open(os.path.join(log_dir, 'train.log'),    'a', buffering=1)
    ctx.metrics_fh = open(os.path.join(log_dir, 'metrics.jsonl'), 'a', buffering=1)

    ctx.log(f'run={args.run_name}  config={args.config}  '
            f'total_frames={args.total_frames}  chunk={args.chunk_frames}  '
            f'seed={args.seed}  config_hash={_config_hash()}')

    # save pip freeze once per run
    pip_freeze_path = os.path.join(run_dir, 'pip_freeze.txt')
    if not os.path.exists(pip_freeze_path):
        import subprocess
        try:
            out = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
            with open(pip_freeze_path, 'wb') as f:
                f.write(out)
        except Exception:
            pass

    # ── hyperparams ───────────────────────────────────────────────────────
    use_cuda       = cfg.getboolean('UseGPU')
    use_gae        = cfg.getboolean('UseGAE')
    use_noisy_net  = cfg.getboolean('UseNoisyNet')
    lam            = float(cfg['Lambda'])
    num_worker     = int(cfg['NumEnv'])
    num_step       = int(cfg['NumStep'])
    ppo_eps        = float(cfg['PPOEps'])
    epoch          = int(cfg['Epoch'])
    mini_batch     = int(cfg['MiniBatch'])
    batch_size     = int(num_step * num_worker / mini_batch)
    learning_rate  = float(cfg['LearningRate'])
    entropy_coef   = float(cfg['Entropy'])
    gamma          = float(cfg['Gamma'])
    int_gamma      = float(cfg['IntGamma'])
    clip_grad_norm = float(cfg['ClipGradNorm'])
    ext_coef       = float(cfg['ExtCoef'])
    int_coef       = float(cfg['IntCoef'])
    sticky_action  = cfg.getboolean('StickyAction')
    action_prob    = float(cfg['ActionProb'])
    life_done      = cfg.getboolean('LifeDone')
    pre_obs_norm   = int(cfg['ObsNormStep'])

    # stash for train_one_chunk
    ctx._hp = dict(
        env_id=env_id, env_type=env_type, use_cuda=use_cuda,
        num_worker=num_worker, num_step=num_step, gamma=gamma,
        int_gamma=int_gamma, ext_coef=ext_coef, int_coef=int_coef,
        batch_size=batch_size, pre_obs_norm=pre_obs_norm,
        sticky_action=sticky_action, action_prob=action_prob,
        life_done=life_done,
    )

    # ── RMS objects ───────────────────────────────────────────────────────
    ctx.obs_rms       = RunningMeanStd(shape=(1, 1, 84, 84))
    ctx.reward_rms    = RunningMeanStd()
    ctx.reward_filter = RewardForwardFilter(int_gamma)

    # ── agent ─────────────────────────────────────────────────────────────
    ctx.agent = RNDAgent(
        input_size, output_size, num_worker, num_step, gamma,
        lam=lam, learning_rate=learning_rate, ent_coef=entropy_coef,
        clip_grad_norm=clip_grad_norm, epoch=epoch, batch_size=batch_size,
        ppo_eps=ppo_eps, use_cuda=use_cuda, use_gae=use_gae,
        use_noisy_net=use_noisy_net,
    )

    # ── resume or fresh start (P7 / P8) ──────────────────────────────────
    # P7: is_load_model was hardcoded True in jcwleo → crashes on first run.
    # Now: only load if --resume is supplied (or 'auto' finds latest.pt).
    resume_path = args.resume
    if resume_path == 'auto':
        latest = os.path.join(ckpt_dir, 'latest.pt')
        resume_path = latest if os.path.exists(latest) else ''

    if resume_path and os.path.exists(resume_path):
        load_checkpoint(resume_path, ctx)
        ctx.log(f'Resumed from step {ctx.global_step:,}')
    else:
        ctx.log(f'Starting fresh — obs-RMS warmup '
                f'({pre_obs_norm} rollouts × {num_step} steps × {num_worker} envs)…')
        _warmup_obs_rms(ctx, num_worker, num_step, output_size, pre_obs_norm,
                        env_id, sticky_action, action_prob, life_done)
        ctx.log('Obs-RMS warmup done.')

    # ── spawn env workers ─────────────────────────────────────────────────
    ctx.log(f'Spawning {num_worker} env workers for {env_id}…')
    works, parent_conns = [], []
    for idx in range(num_worker):
        parent_conn, child_conn = Pipe()
        work = AtariEnvironment(
            env_id, False, idx, child_conn,
            sticky_action=sticky_action, p=action_prob, life_done=life_done,
        )
        work.start()
        works.append(work)
        parent_conns.append(parent_conn)

    ctx.works        = works
    ctx.parent_conns = parent_conns
    ctx.states       = np.zeros([num_worker, 4, 84, 84])
    ctx.log('Workers started.')
    return ctx


def _warmup_obs_rms(ctx, num_worker, num_step, output_size,
                    pre_obs_norm, env_id, sticky_action, action_prob, life_done):
    """Seed obs_rms with pre_obs_norm rollouts of pure-random policy."""
    works_w, pconns_w = [], []
    for idx in range(num_worker):
        pc, cc = Pipe()
        work = AtariEnvironment(
            env_id, False, idx, cc,
            sticky_action=sticky_action, p=action_prob, life_done=life_done,
        )
        work.start()
        works_w.append(work)
        pconns_w.append(pc)

    buf = []
    for _ in range(num_step * pre_obs_norm):
        actions = np.random.randint(0, output_size, size=(num_worker,))
        for pc, act in zip(pconns_w, actions):
            pc.send(act)
        for pc in pconns_w:
            s, _, _, _, _ = pc.recv()
            buf.append(s[3, :, :].reshape([1, 84, 84]))
        if len(buf) % (num_step * num_worker) == 0:
            ctx.obs_rms.update(np.stack(buf))
            buf = []

    for w in works_w:
        w.terminate()


# ─────────────────────────────────────────────────────────────────────────────
# One-chunk training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_one_chunk(ctx):
    """Train for ctx.args.chunk_frames steps, checkpoint, then return ctx.

    Typical notebook Cell 3 usage:
        while ctx.global_step < args.total_frames:
            train_one_chunk(ctx)
    """
    args   = ctx.args
    agent  = ctx.agent
    hp     = ctx._hp
    writer = ctx.writer

    num_worker = hp['num_worker']
    num_step   = hp['num_step']
    gamma      = hp['gamma']
    int_gamma  = hp['int_gamma']
    ext_coef   = hp['ext_coef']
    int_coef   = hp['int_coef']
    env_id     = hp['env_id']

    chunk_start  = ctx.global_step
    chunk_target = chunk_start + args.chunk_frames
    t_start      = time.time()

    ctx.log(f'--- chunk start: step={chunk_start:,}  target={chunk_target:,} ---')

    while ctx.global_step < chunk_target:
        total_state, total_reward, total_done   = [], [], []
        total_next_state, total_action          = [], []
        total_int_reward, total_next_obs        = [], []
        total_ext_values, total_int_values      = [], []
        total_policy, total_policy_np           = [], []

        ctx.global_step   += (num_worker * num_step)
        ctx.global_update += 1

        # ── Step 1: n-step rollout ────────────────────────────────────────
        for _ in range(num_step):
            actions, value_ext, value_int, policy = agent.get_action(
                np.float32(ctx.states) / 255.)

            for pc, act in zip(ctx.parent_conns, actions):
                pc.send(act)

            next_states, rewards, dones = [], [], []
            real_dones, log_rewards, next_obs = [], [], []
            for pc in ctx.parent_conns:
                s, r, d, rd, lr = pc.recv()
                next_states.append(s)
                rewards.append(r)
                dones.append(d)
                real_dones.append(rd)
                log_rewards.append(lr)
                next_obs.append(s[3, :, :].reshape([1, 84, 84]))

            next_states = np.stack(next_states)
            rewards     = np.hstack(rewards)
            dones       = np.hstack(dones)
            real_dones  = np.hstack(real_dones)
            next_obs    = np.stack(next_obs)

            # intrinsic reward (normalised obs)
            obs_norm = ((next_obs - ctx.obs_rms.mean) /
                        np.sqrt(ctx.obs_rms.var)).clip(-5, 5)
            intrinsic_reward = agent.compute_intrinsic_reward(obs_norm)
            intrinsic_reward = np.hstack(intrinsic_reward)
            ctx.recent_int_rewards.extend(intrinsic_reward.tolist())

            total_next_obs.append(next_obs)
            total_int_reward.append(intrinsic_reward)
            total_state.append(ctx.states)
            total_reward.append(rewards)
            total_done.append(dones)
            total_action.append(actions)
            total_ext_values.append(value_ext)
            total_int_values.append(value_int)
            total_policy.append(policy)
            total_policy_np.append(policy.cpu().numpy())

            ctx.states = next_states

            # track episode returns
            for rd, lr in zip(real_dones, log_rewards):
                if rd:
                    ctx.recent_ext_returns.append(float(lr))

        # ── Step 2: bootstrap value ───────────────────────────────────────
        _, value_ext, value_int, _ = agent.get_action(
            np.float32(ctx.states) / 255.)
        total_ext_values.append(value_ext)
        total_int_values.append(value_int)

        # ── Step 3: reshape buffers ───────────────────────────────────────
        total_state      = np.stack(total_state).transpose(
                               [1, 0, 2, 3, 4]).reshape([-1, 4, 84, 84])
        total_reward     = np.stack(total_reward).transpose().clip(-1, 1)
        total_action     = np.stack(total_action).transpose().reshape([-1])
        total_done       = np.stack(total_done).transpose()
        total_next_obs   = np.stack(total_next_obs).transpose(
                               [1, 0, 2, 3, 4]).reshape([-1, 1, 84, 84])
        total_ext_values = np.stack(total_ext_values).transpose()
        total_int_values = np.stack(total_int_values).transpose()
        total_int_reward = np.stack(total_int_reward).transpose()

        # ── Step 4: normalise intrinsic reward ───────────────────────────
        total_reward_per_env = np.array([
            ctx.reward_filter.update(rps)
            for rps in total_int_reward.T
        ])
        mean_rpe = np.mean(total_reward_per_env)
        std_rpe  = np.std(total_reward_per_env)
        ctx.reward_rms.update_from_moments(mean_rpe, std_rpe ** 2,
                                           len(total_reward_per_env))
        total_int_reward /= np.sqrt(ctx.reward_rms.var)

        # ── Step 5: GAE / returns ─────────────────────────────────────────
        ext_target, ext_adv = make_train_data(
            total_reward, total_done, total_ext_values,
            gamma, num_step, num_worker)
        int_target, int_adv = make_train_data(
            total_int_reward, np.zeros_like(total_int_reward),
            total_int_values, int_gamma, num_step, num_worker)
        total_adv = int_adv * int_coef + ext_adv * ext_coef

        # ── Step 6: update obs RMS ────────────────────────────────────────
        ctx.obs_rms.update(total_next_obs)

        # ── Step 7: PPO + RND update ──────────────────────────────────────
        obs_norm_train = ((total_next_obs - ctx.obs_rms.mean) /
                          np.sqrt(ctx.obs_rms.var)).clip(-5, 5)
        metrics = agent.train_model(
            np.float32(total_state) / 255.,
            ext_target, int_target, total_action,
            total_adv, obs_norm_train, total_policy)

        # ── Step 8: log every N updates ──────────────────────────────────
        if ctx.global_update % args.log_interval == 0:
            elapsed  = time.time() - t_start
            fps      = (ctx.global_step - chunk_start) / max(elapsed, 1e-6)
            mean_ext = (float(np.mean(ctx.recent_ext_returns))
                        if ctx.recent_ext_returns else 0.0)
            mean_int = (float(np.mean(ctx.recent_int_rewards))
                        if ctx.recent_int_rewards else 0.0)
            if mean_ext > ctx.best_ext_so_far:
                ctx.best_ext_so_far = mean_ext

            record = {
                'global_step':     ctx.global_step,
                'global_update':   ctx.global_update,
                'wall_clock_s':    round(elapsed, 1),
                'fps':             round(fps, 1),
                'mean_ext_return': round(mean_ext, 4),
                'mean_int_reward': round(mean_int, 6),
                **{k: round(v, 6) for k, v in metrics.items()},
            }

            ctx.metrics_fh.write(json.dumps(record) + '\n')

            for k, v in record.items():
                if isinstance(v, (int, float)):
                    writer.add_scalar(f'train/{k}', v, ctx.global_step)

            ctx.log(
                f'step={ctx.global_step:>10,}  '
                f'fps={fps:>6.0f}  '
                f'ext_ret={mean_ext:>7.2f}  '
                f'int_rew={mean_int:>8.5f}  '
                f'ploss={metrics["policy_loss"]:>7.4f}  '
                f'ent={metrics["entropy"]:>6.4f}  '
                f'gnorm={metrics["grad_norm"]:>6.4f}'
            )

    # ── end of chunk: full-state checkpoint + eval ────────────────────────
    save_checkpoint(ctx)
    eval_checkpoint(ctx)
    ctx.log(f'--- chunk done: step={ctx.global_step:,} ---')
    return ctx


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation  (records MP4 video, appends to eval_results.csv)
# ─────────────────────────────────────────────────────────────────────────────

def eval_checkpoint(ctx, n_episodes=3):
    """Run n_episodes of greedy eval with manual frame capture (no RecordVideo).

    RecordVideo in gym==0.21 silently drops frames when ffmpeg encoding fails.
    Instead we call env.render(mode='rgb_array') explicitly and write with cv2.
    """
    import gym
    import cv2
    from PIL import Image

    hp       = ctx._hp
    agent    = ctx.agent
    env_id   = hp['env_id']

    step_tag  = f'{ctx.global_step:011d}'
    vid_dir   = os.path.join(ctx.video_dir, f'eval_step_{step_tag}')
    os.makedirs(vid_dir, exist_ok=True)

    try:
        env = gym.make(env_id)
        returns, ep_lengths = [], []

        for ep in range(n_episodes):
            raw   = env.reset()
            state = np.zeros([1, 4, 84, 84])
            done  = False
            rall  = 0
            steps = 0
            rgb_frames = []
            while not done:
                action, _, _, _ = agent.get_action(np.float32(state) / 255.)
                obs, reward, done, _ = env.step(int(action[0]))
                # obs IS (210,160,3) RGB for NoFrameskip-v4 — no render() needed
                rgb_frames.append(obs)
                # build grayscale state for policy input
                gray = np.array(Image.fromarray(obs).convert('L')).astype('float32')
                gray = cv2.resize(gray, (84, 84))
                state[0, :3] = state[0, 1:]
                state[0,  3] = gray
                rall  += reward
                steps += 1
                if steps > 4500:
                    break
            returns.append(rall)
            ep_lengths.append(steps)
            # write per-episode MP4 with cv2 (no ffmpeg subprocess needed)
            if rgb_frames:
                vid_path = os.path.join(vid_dir,
                                        f'eval_{step_tag}_ep{ep:02d}.mp4')
                h, w = rgb_frames[0].shape[:2]   # Atari: (210, 160, 3)
                writer = cv2.VideoWriter(
                    vid_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
                for f in rgb_frames:
                    writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
                writer.release()
                ctx.log(f'[eval] video → {vid_path}  ({len(rgb_frames)} frames)')

        env.close()

        mean_ret = float(np.mean(returns))
        std_ret  = float(np.std(returns))
        mean_len = float(np.mean(ep_lengths))

        if mean_ret > ctx.best_ext_so_far:
            ctx.best_ext_so_far = mean_ret
            best_src = _ckpt_path(ctx.ckpt_dir, ctx.global_step)
            best_dst = os.path.join(ctx.ckpt_dir, 'best.pt')
            if os.path.exists(best_src):
                shutil.copy2(best_src, best_dst)

        ctx.log(f'[eval] step={ctx.global_step:,}  '
                f'mean_return={mean_ret:.2f}±{std_ret:.2f}  '
                f'mean_ep_len={mean_len:.0f}  '
                f'video → {vid_dir}')

        csv_path   = os.path.join(ctx.run_dir, 'eval_results.csv')
        new_file   = not os.path.exists(csv_path)
        with open(csv_path, 'a', newline='') as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(['global_step', 'mean_return', 'std_return',
                             'mean_ep_length', 'n_episodes'])
            w.writerow([ctx.global_step, round(mean_ret, 4),
                         round(std_ret, 4), round(mean_len, 1), n_episodes])

    except Exception as e:
        ctx.log(f'[eval] WARNING: eval failed — {e}')


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    ctx  = setup_training(args)
    while ctx.global_step < args.total_frames:
        train_one_chunk(ctx)
    ctx.log('Training complete.')
    ctx.writer.close()


if __name__ == '__main__':
    main()
