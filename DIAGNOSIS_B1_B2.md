# SCDE — Diagnosis of B1 (PPO-only) and B2 (PPO + RND fixed)

> This document is written so it can be pasted directly into ChatGPT for help applying the fixes.
> The reference paper is **Burda et al., 2018 — "Exploration by Random Network Distillation"**
> (arXiv:1810.12894).  CleanRL's `ppo_rnd_envpool.py` and OpenAI's reference repo
> `openai/random-network-distillation` are the canonical reference implementations.

---

## 0. Observed Symptoms

From `scde/experiments/B2.ipynb` (full 2M-step run):

```
Steps:    2048 | Mean Return: 0.09 | Mean Ep Len: 91.1  | Success Rate: 0.06
...
Steps: 1000000 | Mean Return: 0.00 | Mean Ep Len: 101.0 | Success Rate: 0.02
...
Steps: 2000896 | Mean Return: 0.05 | Mean Ep Len: 96.7  | Success Rate: 0.02
```

- Success rate hovers around 2–6 % across the whole 2M run — essentially
  random play. `MiniGrid-FourRooms-v0` has `max_steps = 100`, so
  `Mean Ep Len ≈ 101` means **every episode is timing out**.
- The RND intrinsic reward at init is `0.0016`, **identical across all 16
  envs**. This is the smoking gun — RND has no usable signal.
- B1 (no RND) is exactly the same situation but without curiosity, so it's
  also unsurprisingly flat.

Conclusion: the agent is doing random exploration for 2M steps and the
intrinsic reward is contributing essentially nothing.

---

## 1. What the RND paper actually does (the reference baseline)

These are the bits that matter for FourRooms-style sparse-reward exploration,
taken from Burda et al. 2018 §2.4–§2.6 and Appendix A:

1. **Target & predictor are CNNs over RAW (normalised) pixel observations**,
   not over the policy encoder's features. The intrinsic reward is
   `‖f(s) − f̂(s)‖²` where `f, f̂` are **independent** networks separate
   from the policy.
2. **Observation normalisation**: each pixel dim is whitened with a running
   mean/std and clipped to `[-5, 5]`. Stats are seeded by running a random
   policy for ~1 % of training before learning starts.
3. **Intrinsic-reward normalisation**: `r_int` is divided by a running std
   of the *discounted intrinsic return* (not of the per-step reward).
   Without this, the reward magnitude is meaningless across runs.
4. **Two value heads** (extrinsic + intrinsic) with **different discount
   factors**: `γ_ext = 0.999`, `γ_int = 0.99`. ✅ (you already do this)
5. **Non-episodic intrinsic stream**: GAE for `r_int` does not zero out at
   `done`. ✅ (you already do this)
6. **Advantage combination**: `A = c_ext · A_ext + c_int · A_int`
   with `c_ext = 2.0`, `c_int = 1.0`. Both advantages are computed
   separately, summed weighted, **and the COMBINED advantage is then
   normalised**.
7. **Predictor is trained on a random 25 % subset** of each minibatch (the
   `update_proportion` trick). ✅ (you already do this)
8. **PPO config in the paper**: 128 parallel envs, `n_steps = 128`,
   `n_epochs = 4`, `clip = 0.1`, `lr = 1e-4`, Adam.

---

## 2. Root-cause bug list (ranked by severity)

The numbering matches a fix file later in this doc, so you can map fixes
back to the symptom they target.

### CRITICAL (these alone explain "0 reward after 2M steps")

**[C1] RND operates on the shared policy encoder's features `h`, not on raw pixels.**
- File: `scde/ppo/trainer.py`, line 162–166 — `r_int = self.rnd.intrinsic_reward(h)`.
- File: `scde/models/rnd.py` — target/predictor are 2-layer / 3-layer MLPs
  on `feature_dim=256`.
- Consequence: at init, the encoder produces nearly-identical 256-dim
  features for similar inputs across the 16 parallel envs. Both target and
  predictor are small MLPs initialised similarly, so `‖f(h)−f̂(h)‖²`
  starts at ~`0.0016` and stays there. There is **no novelty signal**.
- Worse: the encoder is updated by PPO every iteration, so the
  distribution of `h` drifts in a way that has nothing to do with
  novelty — the predictor chases a moving target produced by gradient
  flow it doesn't control. This conflates "encoder representation
  change" with "state novelty".
- The plan doc you previously generated does describe this design and
  even tries to justify it ("any gradient through actor/critic also
  updates encoder"). It is **not** what Burda et al. did, and on
  MiniGrid features it produces no usable curiosity signal.

**[C2] No observation normalisation.**
- The MiniCNN takes raw `uint8 / 255` pixels. Burda et al. requires running
  mean/std normalisation clipped to `[-5, 5]`, calibrated before training.
- For an RND-on-pixels design this is mandatory; otherwise the target net
  output variance collapses.

**[C3] No intrinsic-reward normalisation (`RunningNorm` is missing).**
- The plan doc specifies `scde/intrinsic/reward_normaliser.py` with
  EMA-based normalisation. **This file does not exist in the repo.**
- Effect: `r_int = 0.0016` is added raw to `r_ext` (which is `1.0` only on
  goal-reach, else `0`). The intrinsic signal is ~600× smaller than the
  goal reward and effectively invisible.

**[C4] No `α` (intrinsic coefficient) is applied anywhere.**
- File: `scde/ppo/rollout_buffer.py`, line 164: `adv_total = adv_ext + adv_int`.
- `cfg.alpha = 1.0` is in the config but never read. There is no
  `c_ext * A_ext + c_int * A_int`. In the paper these coefficients are
  `2.0` and `1.0` and they materially affect learning.

**[C5] Action space includes 4 useless actions for FourRooms.**
- MiniGrid action set: 0=left, 1=right, 2=forward, 3=pickup, 4=drop,
  5=toggle, 6=done. In FourRooms only {0,1,2} do anything.
- Config has `action_dim: 7`. A random policy thus spends 4/7 ≈ 57 %
  of every step on no-ops. This alone roughly **halves** the effective
  exploration speed.
- Fix: either restrict the action space to 3, or wrap the env to map
  the policy's 3 outputs onto {0,1,2}.

### HIGH

**[H1] `RGBImgObsWrapper` gives the FULL grid then you downsample to 64×64.**
- File: `scde/envs/wrappers.py`, `make_env()`.
- For FourRooms the rendered image is ~152×152. Downsampling to 64×64
  squashes the agent and the green goal cell into ~3-pixel blobs. The
  CNN gets a poor view of *both* the goal and the agent.
- The standard MiniGrid setup for pixel-based PPO is one of:
  - `ImgObsWrapper` (compact 7×7×3 symbolic) — fastest, recommended.
  - `RGBImgPartialObsWrapper` (agent-centric pixels) — best for
    curiosity-driven exploration because new rooms = visually new states.

**[H2] Stale `h` is used for RND loss.**
- File: `scde/ppo/trainer.py` stores `h` (computed under `no_grad`)
  in the rollout buffer.
- File: `scde/ppo/updater.py` line 60 + 133 reuses that stored `h`.
- Result: the RND predictor is trained on features produced by the
  encoder weights from N updates ago, not the current encoder. With
  fast-changing encoder weights this is incoherent.
- If RND is moved to raw pixels (per [C1]) this issue disappears.

**[H3] FourRooms `max_steps = 100` is too short for random exploration.**
- The default env time-limit truncates almost every early episode before
  the agent can stumble onto the goal. With 7-action random play the
  expected hitting time is well over 100.
- Either increase `max_steps` (e.g. 200) via a `gym.wrappers.TimeLimit`
  override or accept it and let curiosity solve it (which requires C1–C4
  to be fixed first).

**[H4] No LR annealing, no entropy annealing.**
- Standard PPO baselines (CleanRL, SB3) anneal LR linearly from `3e-4`
  to `0`. Without it the policy keeps taking large steps after it has
  found something useful and oscillates back to random.

### MEDIUM

**[M1] Advantage normalisation is done on the *sum*, not per-stream.**
- File: `scde/ppo/rollout_buffer.py`, line 167:
  `adv_total = (adv_total - mean) / (std + 1e-8)`.
- The paper normalises by stream then weights. Less critical than C1–C4
  but worth fixing for clean ablations.

**[M2] Single optimiser, single LR for policy + RND predictor.**
- File: `scde/ppo/trainer.py`, `torch.optim.Adam(params, lr=cfg.lr)`.
- Reference impls give RND its own optimiser (often same LR but logged
  separately, sometimes 10×).

**[M3] B1 notebook calls `buffer.add(...)` with 8 args (missing `h`) and uses `gamma_ext=` kwarg.**
- File: `scde/experiments/B1.ipynb` — `buffer.add(obs, action, log_prob, reward, r_int, v_ext, v_int, done_t)` is only 8 args; the buffer's current signature requires 9 (with `h` 8th and `done` 9th). Also the call `compute_returns_and_advantages(... gamma_ext=cfg.gamma ...)` uses a kwarg that does not exist in the current buffer (`gamma=`).
- Either B1 was run before buffer was refactored (silent miscompare) or
  it errored mid-run. Re-run B1 through the same `Trainer` class B2 uses
  to get an apples-to-apples baseline.

**[M4] RND target/predictor activations are ReLU, plan said ELU.**
- File: `scde/models/rnd.py`. Plan §2.5 justifies ELU specifically for
  the random target net (so that no neurons die at random init).
- Minor — but easy to fix.

### LOW

**[L1] `entropy_coef = 0.01` is high for FourRooms.**  Burda uses `0.001`.
Causes the policy to stay near-uniform for longer, which **looks** like
"agent isn't learning" even when value loss is going down.

**[L2] `gamma_ext = 0.999` with `max_steps=100`.**  The effective horizon
(1/(1−γ) = 1000) exceeds the env horizon by 10×. Use `0.99` for
FourRooms; reserve `0.999` for Montezuma-style envs.

---

## 3. Why exp_via_rnd_rl / Burda et al. get good results

The Burda paper + CleanRL `ppo_rnd_envpool.py` succeed on much harder envs
(Montezuma's Revenge) because they get all of:

- RND on **raw pixels** with a **separate CNN** (independent of the
  policy encoder).
- **Observation normalisation** seeded with random rollouts.
- **Intrinsic-return normalisation** so `c_int = 1.0` and `c_ext = 2.0`
  are meaningful constants.
- **Separate advantage normalisation** per stream.
- **Sticky-action / FrameStack** wrappers (not needed for MiniGrid).
- **128 parallel envs** (you use 16 — fine for MiniGrid, just slower).

In your code, 1 through 4 of those are all missing or broken. Fixing
them in order is the path forward.

---

## 4. Fix plan — minimum viable version

Below is the sequence I recommend. **Do not skip C1–C4.** Without those,
B2 will continue to look like B1 forever.

### Step 0 — Sanity baseline (1 hour of compute)
Re-run B1 (PPO-only) through the **same `Trainer` class** B2 uses (i.e.
fix M3 first so the B1 notebook is also driven by `Trainer`).  Use:
- `action_dim = 3` (apply C5 first)
- `entropy_coef = 0.001` (L1)
- `gamma = 0.99` (L2)
- `RGBImgPartialObsWrapper` instead of `RGBImgObsWrapper` (H1)
- `total_steps = 500k` (fast)

Expected: with these PPO-only fixes on FourRooms you should already see
some non-zero success rate (~0.1–0.3 by 500k steps) just from making the
problem learnable. This proves your PPO loop is correct.

### Step 1 — Restrict action space (C5)
Add a thin wrapper:

```python
class MiniGridActionWrapper(gym.ActionWrapper):
    """Restrict action set to {left, right, forward}."""
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(3)
    def action(self, a):
        return int(a)   # 0,1,2 already map to MiniGrid left/right/forward
```

Apply it inside `make_env()` *before* the observation wrappers, then set
`action_dim: 3` in `base.yaml`.

### Step 2 — Replace observation wrapper (H1)

```python
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
env = RGBImgPartialObsWrapper(env, tile_size=8)   # 56×56×3 RGB partial view
env = ImgObsWrapper(env)                          # drop the dict
env = ResizeWrapper(env, size=(64, 64))           # OR drop ResizeWrapper, MiniCNN already handles 56×56
```

If you keep the resize, the existing 64×64 CNN works unchanged.

### Step 3 — Re-implement RND on raw pixels (C1) — biggest fix

Replace `scde/models/rnd.py` so that target and predictor are **their own
CNNs** taking the same `(B, 64, 64, 3) uint8` observations as the policy
encoder. Sketch:

```python
class RNDCNN(nn.Module):
    """Independent CNN used by both target and predictor."""
    def __init__(self, in_channels=3, out_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4), nn.LeakyReLU(),
            nn.Conv2d(32, 64, 4, stride=2),          nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, stride=1),          nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(64*5*5, 512), nn.ReLU(),
            nn.Linear(512, out_dim),
        )
    def forward(self, obs_norm):
        # obs_norm: (B,3,64,64) float, already whitened
        return self.net(obs_norm)

class RNDModule(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()
        self.target    = RNDCNN(out_dim=out_dim)
        self.predictor = RNDCNN(out_dim=out_dim)
        for p in self.target.parameters():
            p.requires_grad_(False)
```

This is the same architecture as Burda Appendix A (smaller variant) and
what CleanRL ships with `ppo_rnd_envpool.py`.

The buffer should now store **observations** for the RND loss, not `h`.
You can drop the `h` field from the buffer entirely.

### Step 4 — Observation normaliser (C2)

A small running-stats class on the pixel obs (over a 50-rollout warmup):

```python
class RunningMeanStd:
    def __init__(self, shape, eps=1e-4):
        self.mean = torch.zeros(shape)
        self.var  = torch.ones(shape)
        self.count = eps
    def update(self, x):     # x: (B, *shape)
        bm = x.mean(0); bv = x.var(0, unbiased=False); bc = x.shape[0]
        delta = bm - self.mean
        tot = self.count + bc
        self.mean = self.mean + delta * bc / tot
        m_a = self.var * self.count
        m_b = bv * bc
        self.var = (m_a + m_b + delta**2 * self.count * bc / tot) / tot
        self.count = tot
    def normalize(self, x):
        return torch.clamp((x - self.mean) / torch.sqrt(self.var + 1e-8), -5, 5)
```

Warm up by stepping with a uniform random policy for ~16k steps before
training. Use the same normaliser for **both** the RND target and
predictor (but **not** the policy CNN, as in the paper).

### Step 5 — Intrinsic-reward normaliser (C3)

```python
class RewardForwardFilter:
    def __init__(self, gamma): self.gamma = gamma; self.rewems = None
    def update(self, r):
        self.rewems = r if self.rewems is None else self.rewems * self.gamma + r
        return self.rewems
```

Then maintain a `RunningMeanStd` over the *discounted* intrinsic return
and divide `r_int` by `sqrt(var + 1e-8)`. This is the exact recipe in
Burda's reference repo (`mpi_util.py` + `policies/cnn_policy.py`).

### Step 6 — Advantage combination (C4 + M1)

In `RolloutBuffer.compute_returns_and_advantages`:

```python
# normalise each stream independently
adv_ext = (adv_ext - adv_ext.mean()) / (adv_ext.std() + 1e-8)
adv_int = (adv_int - adv_int.mean()) / (adv_int.std() + 1e-8)
adv_total = cfg.ext_coef * adv_ext + cfg.int_coef * adv_int
```

Add `ext_coef: 2.0` and `int_coef: 1.0` to `base.yaml`.

### Step 7 — Re-run B2

Expected outcome (rough order of magnitude — compare against published
RND on MiniGrid):
- B1 fixed: success rate > 0.3 by 1M steps on FourRooms.
- B2 fixed: success rate > 0.6 by 1M steps. The whole point of RND on
  FourRooms is that the curious agent finds the goal much sooner than
  the random-explore baseline.

If after applying all of C1–C5 + H1 + H3 you still see ~0 success rate,
the bug is in the rollout / GAE / update loop and not in the
RND module itself. At that point, swap to CleanRL's
`ppo_rnd_envpool.py` as a reference and diff your loop against it
step by step.

---

## 5. What to give ChatGPT — a single prompt

If you want one self-contained block to hand to ChatGPT, paste this:

> I have an RL codebase that implements PPO + RND on MiniGrid-FourRooms-v0.
> After 2M training steps the success rate is 2–3 % and the intrinsic
> reward stays at ~0.0016 across all envs from the very first step.
> My RND module currently runs on the policy encoder's 256-dim feature
> output `h`, not on raw pixel observations. I do not normalise
> observations, I do not normalise intrinsic rewards, I do not weight
> `c_ext` and `c_int` separately, and I am using the default 7-action
> MiniGrid action space.
>
> Please help me refactor my code so that:
> 1. RND target and predictor are **independent CNNs over raw pixel
>    observations** (`(B,3,64,64) uint8` normalised by a running
>    mean/std, clipped to [-5,5]).
> 2. Intrinsic reward is divided by a running std of the **discounted
>    intrinsic return** (Burda 2018 recipe).
> 3. Extrinsic and intrinsic advantages are normalised separately, then
>    combined as `A = 2.0 * A_ext + 1.0 * A_int`.
> 4. The action space is restricted to {left, right, forward} via a
>    `gym.ActionWrapper`.
> 5. `RGBImgPartialObsWrapper` is used instead of `RGBImgObsWrapper`
>    so the agent gets an agent-centric view.
> 6. The RND loss uses **fresh** observations sampled from the rollout
>    buffer, not stale features `h` stored at rollout time.
>
> My current files are: `scde/envs/wrappers.py`, `scde/envs/vec_env.py`,
> `scde/models/encoder.py`, `scde/models/actor_critic.py`,
> `scde/models/rnd.py`, `scde/ppo/rollout_buffer.py`,
> `scde/ppo/updater.py`, `scde/ppo/trainer.py`, plus YAML configs in
> `scde/configs/`. I will paste each one in turn. Please give me the
> minimal diff for each file — do **not** rewrite from scratch.

Then paste each file one at a time and let it produce diffs.

---

## 6. Quick checklist before re-running B2

- [ ]  `action_dim = 3` (and matching `MiniGridActionWrapper`)
- [ ]  `RGBImgPartialObsWrapper`  (replaces `RGBImgObsWrapper`)
- [ ]  `RNDModule` is a CNN over `(B,3,64,64)` raw obs, with independent
       target & predictor — **not** on `h`
- [ ]  Running mean/std observation normaliser, warmed up on a random
       policy for 16k steps
- [ ]  Running intrinsic-return normaliser (`RewardForwardFilter` +
       `RunningMeanStd`)
- [ ]  `adv_ext` and `adv_int` normalised **separately**, combined with
       `ext_coef=2.0, int_coef=1.0`
- [ ]  `gamma_ext = 0.99`, `gamma_int = 0.99`, `entropy_coef = 0.001`,
       linear LR anneal `3e-4 → 0`
- [ ]  `total_steps = 500k` for a fast smoke test; only then go to 2M
- [ ]  B1 notebook is migrated to use `Trainer(cfg)` (same path as B2),
       not its own copy-pasted training loop

---

## 7. Sources / references

- Burda, Y. et al. 2018. *Exploration by Random Network Distillation.*
  arXiv:1810.12894 — §2 (algorithm), §A (hyperparameters).
- OpenAI reference repo: `openai/random-network-distillation`
  (`policies/cnn_policy.py`, `ppo_agent.py`).
- CleanRL `ppo_rnd_envpool.py` — clean single-file PPO+RND, follows the
  paper line-by-line.
- MiniGrid docs:
  - Wrappers: <https://minigrid.farama.org/api/wrapper/>
  - FourRooms env (max_steps=100):
    <https://minigrid.farama.org/environments/minigrid/FourRoomsEnv/>
- Plan doc you generated earlier: `SCDE_Complete_Project_Plan.docx`
  (the modules cited here as "missing" — `intrinsic/reward_normaliser.py`,
  `intrinsic/semantic_memory.py`, `intrinsic/adaptive_weights.py` — are
  specified there but not yet created in the repo).
