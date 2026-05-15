# REBUILD PLAN — PPO + RND ground-truth implementation for SCDE

> Drafted 2026-05-14. Companion to `DIAGNOSIS_B1_B2.md` (which catalogues the
> bugs in the current `scde/` codebase). This plan assumes you have read that
> diagnosis and have decided to lift a known-good implementation rather than
> continue patching `scde/` piecewise.

---

## 0. Decisions already locked in

These were confirmed in the planning conversation on 2026-05-14:

1. **Primary base repo:** `jcwleo/random-network-distillation-pytorch`
   (PyTorch, MIT licence, Windows-runnable after one small patch).
2. **Cross-reference repo:** `vwxyzjn/cleanrl` — file
   `cleanrl/ppo_rnd_envpool.py` (single file, 539 lines, MIT) used only to
   resolve ambiguities. **Will not run on Windows** because `envpool` ships
   no Windows wheels.
3. **Spec / behavioural ground-truth:** `openai/random-network-distillation`
   (TF1 + MPI, archived). Used only as a paper-fidelity reference for
   constants and algorithmic decisions — **no code lifted from it**.
4. **Benchmark scope:** Atari (Montezuma, Venture, Gravitar at minimum) +
   MiniGrid-FourRooms (to keep your existing ablation track B1–B5 alive).
5. **Layout:** existing `scde/` stays untouched (CLAUDE.md rule #1).
   New work lives in a sibling module — proposed name `rnd_baseline/`.
6. **Plan granularity:** 5–7 sessions, each with a verifiable
   pass/fail checkpoint.

---

## 1. Why a rebuild instead of more patches

From `DIAGNOSIS_B1_B2.md`, the current `scde/` has at minimum **5 critical
bugs (C1–C5)** that each independently can explain near-zero return on
FourRooms. Fixing them one by one risks the *synchronisation problem* you
flagged: changing one parameter or one model's input/output shape silently
breaks the next, and you end up debugging your own integration rather than
running experiments.

The rebuild flips the dependency: start from a codebase that is **already
known to converge** (jcwleo trains Montezuma to ~6,100 mean return, see
their issue #26 — short of Burda's ~10k but unambiguously a learning agent,
not random play), then add your SCDE contributions on top with the working
baseline as a regression guardrail.

---

## 2. Verified repo audit (don't re-research this)

### 2.1 jcwleo/random-network-distillation-pytorch — PRIMARY

| Aspect | Value |
| --- | --- |
| Top-level files | `agents.py`, `envs.py`, `model.py`, `train.py`, `eval.py`, `utils.py`, `config.py`, `config.conf`, `make_animation.py`, `train.sh`, `LICENSE` (MIT), `README.md` |
| Default env | `MontezumaRevengeNoFrameskip-v4`; Mario backend also supported via `gym_super_mario_bros` |
| Stack | `gym`, `opencv-python`, `torch`, `tensorboardX`, std `multiprocessing` |
| OS | Runs on Windows after **one patch**: replace `from torch._six import inf` in `utils.py` with `from math import inf` (`torch._six` removed in PyTorch ≥1.13) |
| Known bug | `utils.global_grad_norm_` computes the norm but never rescales gradients (open issue #17). Must be fixed if you rely on `ClipGradNorm`. |
| Config format | Python `configparser` INI (`config.conf`), parsed in `config.py` |
| Licence | MIT |

**Package-upgrade matrix (verified 2026-05-14 from the repo source, not assumed):**

The repo has **no `requirements.txt`** and **no version pins** anywhere. The README's
"Requirements" list is just unpinned bullets (`python3.6`, `gym`, `OpenCV Python`,
`PyTorch`, `tensorboardX`). Last meaningful issue activity Mar 2024. None of the 10
open issues mention `torch._six`, `gym 0.26`, or `numpy>=1.24` — so all compatibility
patches fall on us. Concrete plan:

| Dep | jcwleo's assumed version | Target version | Action |
| --- | --- | --- | --- |
| Python | 3.6 (README) | **3.10 or 3.11** | Code is plain f-strings + stdlib; nothing in 3.10/3.11 breaks it. Avoid 3.12 — `gym==0.21` wheels are flaky there. |
| PyTorch | unpinned, but `torch._six` import forces ≤1.12 | **`torch>=2.1` + CUDA 12.1** (RTX 4060 needs ≥cu121) | Patch P1 unblocks this. |
| torchvision | not used by jcwleo | match torch (`torchvision==0.16+`) | Needed only when CLIP arrives. |
| gym (classic) | inferred 0.21 (uses 4-tuple `step()` and single-value `reset()`) | **pin `gym==0.21.0`** | gym ≥ 0.26 broke the `step()`/`reset()` API and would require rewriting `envs.py`. Pinning is the cheap option. |
| ale-py | not directly imported | **latest** | Use `gym[atari,accept-rom-license]==0.21.0` + `AutoROM --accept-license` for ROMs. atari-py is deprecated; skip it. |
| numpy | unpinned | **any ≥ 1.24** | jcwleo doesn't use `np.bool`/`np.int`/`np.float` deprecated aliases (grep-verified). |
| opencv-python | unpinned | latest | Only `cv2.resize`/`cvtColor` used — version-insensitive. |
| tensorboardX | listed | **drop in favour of `torch.utils.tensorboard`** | See P6. |
| gym_super_mario_bros, nes_py | listed | **remove (P9)** | Mario is not a target env; `BinarySpaceToDiscreteSpaceEnv` is broken on modern `nes_py`. |
| MiniGrid | not in jcwleo | **`minigrid>=2.3`** | Add for Session 2. Pin against gym 0.21 compatibility — some MiniGrid versions require gymnasium. |

**CLIP integration (Session 5+):** `openai/CLIP` requires `torch>=1.7.1` + `torchvision`,
which our `torch>=2.1` already satisfies. Install: `pip install ftfy regex tqdm` then
`pip install git+https://github.com/openai/CLIP.git`. ViT-B/32 outputs 512-d features,
weights ~150 MB. **Atari 84×84 grayscale needs to be upsampled to 224×224 RGB before
feeding CLIP**, which is the main cost (~10–30 ms/batch on a 4060). CLIP itself works
on Windows; only `ftfy`/`regex` need wheels (they have them).

**Verdict:** the "outdated" label is fair — the repo is unmaintained — but the actual
upgrade work is **two real patches (P1, P7/P8) and three cosmetic ones (P6, P9, version
pins).** It is not a rewrite.

**Default Montezuma hyperparameters (from `config.conf [DEFAULT]`):**

```
EnvType            = atari
EnvID              = MontezumaRevengeNoFrameskip-v4
NumEnv             = 128
NumStep            = 128
LearningRate       = 1e-4
Gamma  (ext)       = 0.999
IntGamma           = 0.99
Lambda             = 0.95
StateStackSize     = 4
PreProcHeight=84   ProProcWidth=84  (typo in repo)
Entropy            = 0.001
Epoch              = 4
MiniBatch          = 4
PPOEps             = 0.1
ExtCoef            = 2.0
IntCoef            = 1.0
UpdateProportion   = 0.25
ObsNormStep        = 50              # rollouts of pure random policy used to seed obs RMS
StickyAction=True  ActionProb=0.25
LifeDone=False     UseGAE=True
ClipGradNorm       = 0.5
StableEps          = 1e-8
UseNorm=False  UseNoisyNet=False
MaxStepPerEpisode  = 4500
```

**Where the four critical pieces live in jcwleo:**

| Piece | File | Notes |
| --- | --- | --- |
| Running mean/std on raw pixels, clipped [-5,5] | `utils.RunningMeanStd` + warmup loop in `train.py` (~50 rollouts of a random policy before training) | `obs_rms = RunningMeanStd(shape=(1,1,84,84))` |
| Intrinsic-reward filter + RMS on discounted intrinsic return | `utils.RewardForwardFilter` + `RunningMeanStd`; used in `train.py` to divide `total_int_reward /= sqrt(reward_rms.var)` | This is the Burda recipe |
| Independent target + predictor CNN | `model.RNDModel` — Conv(1→32→64→64) + LeakyReLU + 3×Linear(512), `target.requires_grad=False` | 1-channel 84×84 input, separate from the 4-channel policy CNN |
| Two value heads + ext/int coef combo | `model.CnnActorCriticNetwork` has `critic_ext` and `critic_int`. `train.py` calls `make_train_data` twice (extrinsic episodic, intrinsic non-episodic) then `total_adv = int_adv*int_coef + ext_adv*ext_coef` | **Does NOT apply per-batch advantage mean/std normalisation** — see §2.2 |

### 2.2 vwxyzjn/cleanrl — `cleanrl/ppo_rnd_envpool.py` — CROSS-REFERENCE

Single file, 539 lines. Same Conv stack, same 50-step obs warmup, same
constants as jcwleo. Differs in one important place CleanRL gets right:

```python
# lines 485-486 of ppo_rnd_envpool.py
mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
```

**Decision for the rebuild:** lift jcwleo wholesale, then patch in CleanRL's
per-minibatch adv normalisation in the equivalent spot. Both repos are MIT
so this is fine.

Other line landmarks in `ppo_rnd_envpool.py`:

- `RunningMeanStd` import & init: lines 15, 303
- Obs-norm warmup: lines 322–334
- `RNDModel` class: lines 183–230
- `RewardForwardFilter`: lines 232–241
- Dual GAE loop: lines 410–434
- Adv combination: line 446

### 2.3 openai/random-network-distillation — PAPER-SPEC ONLY

TF1 + MPI, archived, no `LICENSE` file. Read `ppo_agent.py` and
`policies/cnn_policy_param_matched.py` only when CleanRL and jcwleo
disagree and you need the original. Do not import any TF code.

---

## 3. File-by-file lift map

Target module layout — drop this beside `scde/`, do not touch `scde/`:

```
minor_project_rl/
├── scde/                          # untouched; CLAUDE.md rule #1
└── rnd_baseline/                  # NEW
    ├── __init__.py
    ├── config.conf                # from jcwleo, MiniGrid section added
    ├── config.py                  # from jcwleo
    ├── envs.py                    # from jcwleo + MiniGrid wrappers
    ├── model.py                   # from jcwleo (RNDModel + CnnActorCritic)
    ├── agents.py                  # from jcwleo (RNDAgent class)
    ├── utils.py                   # from jcwleo + patches (P1, P2 below)
    ├── train.py                   # from jcwleo + per-mb adv norm patch
    ├── eval.py                    # from jcwleo
    ├── README_LIFTED.md           # licence attribution (MIT, jcwleo)
    └── experiments/
        ├── reproduce_montezuma.md
        ├── reproduce_fourrooms.md
        └── ablations/             # SCDE-specific (B1..B5 ported later)
```

### Patches we apply on top of the lifted code

| ID | File | Change | Reason |
| --- | --- | --- | --- |
| P1 | `utils.py` (line 4) | `from torch._six import inf` → `from torch import inf` | `torch._six` removed in PyTorch ≥1.13. Verified still in repo on 2026-05-14. |
| P2 | `utils.py` | Fix `global_grad_norm_` to actually rescale (multiply each param.grad by `clip_coef / (total_norm + 1e-6)` when `total_norm > clip_coef`), or just replace with `torch.nn.utils.clip_grad_norm_` | jcwleo issue #17 |
| P3 | `train.py` | After computing combined adv and before the inner PPO epoch loop, add CleanRL's per-minibatch adv normalisation | matches Burda + CleanRL recipe |
| P4 | `envs.py` | Add a MiniGrid backend (`AtariEnvironment` already exists as a class — mirror its interface as `MiniGridEnvironment`) | keep your B1–B5 experiments runnable |
| P5 | `config.conf` | Add a `[MINIGRID]` section with FourRooms hyperparams (smaller NumEnv=16, gamma=0.99, max_steps via TimeLimit wrapper) | per `DIAGNOSIS_B1_B2.md` L2 + H3 |
| P6 | `train.py`, `eval.py` | Swap `from tensorboardX import SummaryWriter` → `from torch.utils.tensorboard import SummaryWriter` (drop-in, same API) | `tensorboardX` is unmaintained; native torch TB is part of every PyTorch install |
| P7 | `train.py` | Add `argparse` with `--resume <path>`, `--config <section>`, `--total-frames`, `--chunk-frames` flags. Flip hardcoded `is_load_model = True` to `False` (it currently crashes on first run because the checkpoint file doesn't exist) | Required for the 2M-chunk resume strategy on RTX 4060 8GB — see §3A |
| P8 | `train.py` | Replace the existing "save only `.model/.pred/.target`" with a **full-state checkpoint** (model weights, optimizer state, global_step, obs_rms mean/var/count, reward_rms mean/var/count, `RewardForwardFilter.rewems`, NumPy + PyTorch RNG states). Atomic write via temp file + rename. Keep last 3 + best-by-extrinsic. | jcwleo only saves weights — resume otherwise loses normalisation stats and re-warms from scratch, destroying training |
| P9 | `envs.py` (lines 8–10) | Delete `gym_super_mario_bros` and `nes_py` imports + `MarioEnvironment` class. `BinarySpaceToDiscreteSpaceEnv` was renamed to `JoypadSpace` in modern `nes_py` and breaks the import. | We are not training Mario. Removing the import unblocks Windows install. |

### What stays in `scde/` (for now)

Nothing is deleted. The existing `scde/experiments/B1.ipynb` and `B2.ipynb`
can keep running against the old code as a "before" baseline. Once
`rnd_baseline/` produces a working FourRooms baseline (Session 4
checkpoint), you decide whether to port your SCDE contributions
(`adaptive_weights`, `semantic_memory`, `reward_normaliser`) into
`rnd_baseline/` or replicate the working pieces from `rnd_baseline/` into
`scde/`. That decision is intentionally deferred to Session 5.

---

## 3A. Engineering infrastructure (checkpointing, logging, video, metrics, notebooks)

This section exists because the original jcwleo `train.py` is a hardcoded script
with no argparse, no resume, and only saves model weights. On an RTX 4060 8GB
we cannot train Montezuma 50M frames in one shot — we need resumable training
in chunks. We also need video, metric logs, and a notebook-driven workflow.

### 3A.1 Checkpoint design (the 2M-frame chunk strategy)

Concrete answer to §8 Q1: on a single RTX 4060 8GB laptop, plan for
**chunked training** of 2M frames per chunk (~30–90 minutes of wall-clock
depending on `NumEnv`), with a full-state checkpoint at every chunk
boundary. Resume must be **bit-for-bit equivalent** to a never-paused run —
otherwise you'll spend Session 3 chasing phantom regressions.

**Full-state checkpoint payload** (single `.pt` file per chunk):

```
{
    "global_step":        int,
    "global_update":      int,
    "model":              state_dict,         # CnnActorCritic
    "rnd":                state_dict,         # RNDModel (both target+predictor)
    "optimizer":          state_dict,         # ONE Adam for policy+predictor (jcwleo style)
    "lr_scheduler":       state_dict,         # if linear anneal added (see P-future)
    "obs_rms":  {"mean": np.array, "var": np.array, "count": float},
    "reward_rms": {"mean": float, "var": float, "count": float},
    "reward_forward_filter": {"rewems": np.array or None, "gamma": float},
    "rng": {"numpy": ..., "torch": ..., "python": ..., "torch_cuda": ...},
    "config_section": str,                    # e.g. "DEFAULT" or "MINIGRID"
    "config_hash": str,                       # sha1 of config.conf to detect drift
    "wall_clock_seconds_so_far": float,
    "best_extrinsic_so_far": float,
    "frames_since_last_checkpoint": int,
}
```

**Storage layout** (under `rnd_baseline/runs/<run_name>/`):

```
runs/
  montezuma_001/
    config.snapshot.conf            # exact config used (immutable for this run)
    checkpoints/
      chunk_00002000000.pt          # at frame 2M
      chunk_00004000000.pt          # at frame 4M
      chunk_00006000000.pt          # at frame 6M  (only last 3 kept)
      best.pt                       # symlink/copy → highest mean_ext_return
      latest.pt                     # symlink/copy → most recent chunk
    logs/
      train.log                     # stdout/stderr tee
      metrics.jsonl                 # one JSON record per logging interval
      tensorboard/                  # TB event files
    videos/
      eval_step_00002000000.mp4
      eval_step_00004000000.mp4
      ...
    eval_results.csv                # mean return / rooms-visited per chunk
```

**Atomicity:** every checkpoint write goes to `chunk_XXXX.pt.tmp` first,
then `os.replace(tmp, final)`. This prevents corrupted checkpoints if you
close the laptop lid mid-write.

**Retention:** keep last 3 chunks + `best.pt` + every 10M-frame "milestone"
chunk (00010000000, 00020000000, ...). Auto-delete older ones to manage
disk.

### 3A.2 Logging spec

Three parallel streams — all driven from one logger in `rnd_baseline/utils.py`:

1. **`train.log`** — human-readable, tee'd from stdout/stderr. Includes the
   git SHA, config-section name, and config-hash on line 1 so a future
   handover can reconstruct the run.
2. **`metrics.jsonl`** — one JSON object per `log_interval` (every 50
   updates). Schema:
   ```json
   {
     "global_step": 2048000,
     "global_update": 1000,
     "wall_clock_s": 2734.1,
     "fps": 749.5,
     "mean_ext_return":  3.0,
     "mean_int_reward":  0.241,
     "mean_episode_length": 412.0,
     "policy_loss": 0.013, "ext_value_loss": 0.02, "int_value_loss": 0.03,
     "rnd_loss": 0.41,  "entropy": 1.78, "approx_kl": 0.011, "clipfrac": 0.08,
     "explained_var_ext": 0.71, "explained_var_int": 0.58, "grad_norm": 0.42,
     "rooms_visited_unique": 7  // Montezuma only
   }
   ```
   JSONL is grep/jq/pandas-friendly — far better for post-hoc plots than
   parsing TB event files.
3. **TensorBoard** — same metrics, plus histograms of intrinsic-reward,
   advantage, action-distribution. Launch from a notebook with
   `%load_ext tensorboard` + `%tensorboard --logdir runs/`.

### 3A.3 Video recording

Two recording modes, both off the same `gym.wrappers.RecordVideo`:

1. **Eval-time video** — at every chunk boundary, instantiate **one**
   single-env eval rollout with `RecordVideo(env, video_folder=..., name_prefix="eval_step_{global_step}")`.
   Cap at 3 episodes. Goes to `videos/`. This is the "show me the agent
   playing" deliverable.
2. **Train-time spot-video** — optional. Record env-0 for every 5M frames
   so you can scrub through learning progress. Off by default (disk-heavy).

`gym==0.21` ships `gym.wrappers.RecordVideo` — confirmed compatible. For
MiniGrid, set `render_mode="rgb_array"` and use the same wrapper.

The repo already has `make_animation.py` for GIF export — we keep it for
quick previews but the MP4 path above is the canonical one.

### 3A.4 Metric evaluation — paper + field-standard

Drawn from Burda et al. 2018 §4 + standard PPO instrumentation (CleanRL,
SB3). Captured every logging tick; aggregated by `eval.py` between chunks.

**Paper-defined (Montezuma exploration metrics):**

- `mean_ext_return` over last 100 episodes (Burda Table 1 column).
- `rooms_visited_unique` — count of distinct rooms entered in the last
  N episodes. Burda reports up to 22/24 rooms on Montezuma. Implemented by
  reading the Atari RAM byte 0x83 (current room ID). Cite: Burda §4.2.
- `mean_int_reward` per step — should rise early then decay as predictor
  learns. A flat curve = the bug from `DIAGNOSIS_B1_B2.md`.

**For FourRooms:**

- `success_rate` over last 100 episodes (1 if reached goal else 0).
- `mean_episode_length`. Should fall below max_steps once the agent learns.
- `unique_cells_visited` per episode (proxy for exploration breadth).

**Field-standard PPO diagnostics (always on):**

- Per-update: policy loss, ext value loss, int value loss, RND loss,
  entropy, approx KL, clipfrac, grad norm.
- Explained variance for both value heads (Burda Appendix A reports this).
- FPS (frames per second) — sanity check that the loop isn't I/O-bound.

**Eval protocol (between chunks):**

`eval.py` (refactored from jcwleo's, see P7) runs 10 evaluation episodes
with stochastic policy, sticky-action OFF (deterministic eval is paper
convention), reports mean & std of return + rooms visited. Writes a single
row to `eval_results.csv`. This is the row your final comparison plots
read.

### 3A.5 Notebook structure (one per experiment)

Each experiment is one Jupyter notebook under
`rnd_baseline/experiments/notebooks/`. Notebook acts as an experiment
runner — it does NOT replace `train.py`. The notebook's job is to:

1. **Cell 1 — Config**: define run name, config section, total frames,
   chunk size, seed. Save a `config.snapshot.conf` to `runs/<run_name>/`.
2. **Cell 2 — Setup**: import, set seeds, build env, model, optimizer.
   If `runs/<run_name>/checkpoints/latest.pt` exists, load it. Print the
   resume status (e.g. "Resuming from step 4,000,000").
3. **Cell 3 — Train**: call `train_one_chunk(...)` from `rnd_baseline/train.py`
   until `global_step >= total_frames`. Each chunk writes a checkpoint
   and exits cleanly. The cell loop is `while global_step < total_frames:`.
   You can interrupt at chunk boundaries safely.
4. **Cell 4 — Eval**: run `eval.py` against the latest checkpoint, record
   video.
5. **Cell 5 — Plots**: `metrics.jsonl` → pandas → matplotlib. Plot
   ext-return, int-reward, rooms-visited, all losses.
6. **Cell 6 — Notes**: a free-text markdown cell for what you observed.

Naming: `S1_smoke_montezuma_1M.ipynb`, `S2_fourrooms_500k.ipynb`,
`S3_montezuma_full_50M.ipynb`, `S4_ablation_B1_fourrooms.ipynb`, etc.
Each notebook references its session number for traceability with
`HANDOVER.md`.

**Why notebook + `train.py` not pure script:** notebooks give you cell-
by-cell debuggability, in-cell TB embedding, and one canonical place per
experiment that holds the config + the result plots side by side. But
the heavy lifting must stay in `train.py` modules so notebooks don't
become impossible to diff in git.

### 3A.6 Reproducibility hygiene

- Set `numpy`, `python`, `torch`, `torch.cuda` seeds at run start.
  Record all four in the checkpoint (P8).
- `torch.backends.cudnn.deterministic = True` for ablation runs. Drop
  for the 50M training run to recover ~10% speed.
- Pin the upstream jcwleo SHA in `README_LIFTED.md` and `config.snapshot.conf`.
- Save `pip freeze` to `runs/<run_name>/pip_freeze.txt` on first chunk.

---

## 4. Synchronisation-risk mitigation

You correctly flagged that the bigger danger with mixing repos is silent
parameter/shape drift: e.g. CleanRL's 4-stack frames + 1-channel RND input,
but jcwleo's policy CNN takes the 4-channel stacked input — get one wrong
and the network shape mismatches surface as a single torch error, or worse,
silently broadcast and train to nothing.

**Rules to follow in every session:**

1. **Single source per file.** Each file in `rnd_baseline/` is copied
   verbatim from *exactly one* upstream repo. Mark the source in a
   one-line header comment. Only `train.py` is allowed to be a hybrid
   (jcwleo base + P3 from CleanRL) and that one hybrid spot is called out
   in this doc.
2. **Lock the upstream commit.** Pin the jcwleo SHA in `README_LIFTED.md`
   so future sessions know which version was lifted.
3. **Constants table.** Maintain a table in `rnd_baseline/README_LIFTED.md`
   of every hyperparameter and which repo's value we use. If a parameter
   exists in both repos and they agree, fine. If they disagree, the table
   records the choice and the reason.
4. **Shape pin file.** Add `rnd_baseline/shapes.md` listing input/output
   shapes for every model: policy CNN, RND target, RND predictor. Any PR
   that changes a shape has to update this file in the same change.
5. **Smoke test before any ablation.** Every session starts by running the
   Session-1 smoke test (Montezuma 1M steps, expect non-zero `mean_int_reward`
   and non-zero `mean_ext_reward`). If it fails, you didn't break the
   previous session's work — fix that first before adding anything new.

---

## 5. Session-by-session plan (5 sessions; +2 optional)

Each session has a **goal**, **steps**, and a **CHECKPOINT** (the pass/fail
signal you can paste into the next handover). Sessions are sized to fit a
single working session each. None of them edit `scde/`.

### Session 1 — Lift, patch, smoke-test on Montezuma

**Goal:** Get jcwleo running on this Windows + RTX 4060 8GB machine with
all 9 compatibility/infra patches applied, prove convergence on a 1M-frame
smoke test before any further work.

**Steps:**
1. `git clone https://github.com/jcwleo/random-network-distillation-pytorch`
   into `rnd_baseline/_upstream/` (sub-folder, not committed in final form).
   Pin the SHA in `rnd_baseline/README_LIFTED.md`.
2. Copy listed files into `rnd_baseline/` (see §3 layout).
3. Apply **all 9 patches** from §3: P1 (torch._six), P2 (grad-norm), P6
   (tensorboardX→torch.utils.tensorboard), P7 (argparse + --resume), P8
   (full-state checkpoint), P9 (drop Mario imports). P3, P4, P5 are
   Sessions 2–3 — skip for now.
4. Create `rnd_baseline/requirements.txt` matching the §2.1 upgrade matrix
   and install: `torch>=2.1`, `gym==0.21.0`, `gym[atari,accept-rom-license]==0.21.0`,
   `opencv-python`, `tensorboard`, `minigrid>=2.3`, `numpy`, `matplotlib`,
   `pandas`, `jupyter`. Install ROMs via `AutoROM --accept-license`.
5. Reduce `NumEnv=128 → 16` in `config.conf` for the smoke test only
   (Session 3 raises this to 32 with measured VRAM).
6. Create `experiments/notebooks/S1_smoke_montezuma_1M.ipynb` using the
   six-cell template from §3A.5. Set `total_frames=1_000_000`,
   `chunk_frames=500_000` (so we see two chunk-resume cycles inside the
   smoke test — this is what validates P8 before Session 3 stakes get
   higher).
7. Run all 6 cells. The notebook should auto-resume after Cell 3 is
   interrupted and re-run.

**CHECKPOINT 1 (must pass to start Session 2):**
- `metrics.jsonl` shows `mean_ext_return` rising above 0 within 1M frames.
- `mean_int_reward` is non-flat across envs (real intrinsic signal, not
  the `0.0016` constant from `scde/`).
- No NaN in any loss, no shape errors.
- **One MP4 eval video exists** in `videos/eval_step_00000500000.mp4`
  showing the agent attempting Montezuma. (Sanity check that video
  recording works — needed for all later sessions.)
- **Interrupting Cell 3 mid-run, then re-running it, resumes from the last
  chunk checkpoint with no metric discontinuity.** This validates P8.

**If failing:** do not proceed. Likely culprits — in order of frequency:
gym version mismatch (0.21 vs 0.26 API), ROM path, P1 not applied,
checkpoint not saving obs_rms (P8 incomplete).

---

### Session 2 — Add MiniGrid backend, smoke-test FourRooms

**Goal:** Make `rnd_baseline/` run FourRooms with the same algorithm that
works on Montezuma. This is your B2 redo with a known-good RND core.

**Steps:**
1. Apply P4 (MiniGrid backend): in `rnd_baseline/envs.py`, add a
   `MiniGridEnvironment` class mirroring the `AtariEnvironment` interface.
   Use `RGBImgPartialObsWrapper` + a 3-action wrapper (see
   `DIAGNOSIS_B1_B2.md` §C5).
2. Apply P5: add `[MINIGRID]` section to `config.conf`:
   - `NumEnv=16`, `NumStep=128`, `Gamma=0.99`, `IntGamma=0.99`,
     `Entropy=0.001`, `ExtCoef=2.0`, `IntCoef=1.0`,
     `UpdateProportion=0.25`, `MaxStepPerEpisode=200`,
     `StateStackSize=1` (no frame-stack for symbolic-ish input).
3. Branch on `EnvType` in `train.py` and `envs.py` to pick the right env.
4. Apply P3 (per-mb adv normalisation). This change applies to all
   sessions going forward.
5. Create `experiments/notebooks/S2_fourrooms_500k.ipynb` from the §3A.5
   template, with `config_section="MINIGRID"`, `total_frames=500_000`,
   `chunk_frames=250_000`.
6. Run all 6 cells. Confirm video recording works for MiniGrid env
   (`render_mode="rgb_array"`).

**CHECKPOINT 2:**
- `success_rate` ≥ 0.3 by 500k steps (per `DIAGNOSIS_B1_B2.md` Step 7
  expectations).
- `mean_int_reward` variance > 0 across envs (sanity — same regression
  guard as Session 1).
- `videos/eval_step_00000250000.mp4` shows agent moving through rooms.
- `eval_results.csv` has two rows (one per chunk).

---

### Session 3 — Full Atari reproduction sweep (chunked, resumable)

**Goal:** Reproduce jcwleo-grade numbers on the three Atari games used as
exploration benchmarks in Burda et al., on a single RTX 4060 8GB, training
in 2M-frame chunks so the laptop can be closed between chunks.

**Hardware-driven hyperparameter changes (from Burda/jcwleo defaults):**

- `NumEnv = 32` (not 128). 8GB VRAM does not fit 128 parallel envs + RNDModel
  + AC model + rollout buffer comfortably. 32 is the safe baseline; try 64
  once you have measured peak VRAM with `nvidia-smi`. Lower `NumEnv` → more
  wall-clock per frame but same total frames.
- `NumStep = 128` (unchanged).
- Effective batch per update = `NumEnv * NumStep = 4096` (down from 16384).
- Same `LearningRate=1e-4` — Burda's value, robust to batch size in practice.
- All other hyperparams unchanged from `config.conf [DEFAULT]`.

**Per-chunk procedure** (Cell 3 of `S3_montezuma_full_50M.ipynb`):

1. Read `runs/montezuma_001/checkpoints/latest.pt` if exists → resume.
   Else start fresh and run the 50-rollout obs-RMS warmup once.
2. Train for `chunk_frames = 2_000_000`. Log every 50 updates.
3. At chunk end: write full-state checkpoint (P8), run `eval.py`
   (10 episodes, record one MP4), append to `eval_results.csv`.
4. Exit cell cleanly. **You can shut the laptop here.** Re-running the cell
   resumes from the next chunk.

**Schedule (estimate, single 4060 8GB):**

- Montezuma 50M frames @ NumEnv=32 ≈ 25 chunks × ~45 min ≈ ~19 hours total.
  Spread across 3–5 days of laptop sessions; checkpoints make this safe.
- Venture / Gravitar: same recipe, can budget 20M each (~8 hours each) for
  indicative runs.

**CHECKPOINT 3:**
- Montezuma mean return ends the 50M-frame run at ~6,000 (jcwleo's reported
  final number, per their issue #26). Burda's ~10k is the stretch goal;
  matching jcwleo is the realistic one. Intermediate milestones are not
  documented upstream — record your own curve as the new reference.
- `rooms_visited_unique` curve trends upward across chunks. Burda reports
  up to 22 rooms; matching ≥15 is a strong outcome.
- Venture and Gravitar are non-trivially positive (>0 mean return,
  consistent with Burda Table 1). Exact targets unverified upstream; use
  Burda Table 1 as the target band, not jcwleo.
- A resume-from-checkpoint produces a metrics curve that is **continuous**
  with the pre-checkpoint segment (i.e. no obvious jump). This is the
  acid test for P8 correctness.

**If checkpoint discontinuity is detected:** stop the run, fix P8
(usually missing obs_rms or reward_rms state in the checkpoint), and
restart the chunk from the previous good checkpoint.

---

### Session 4 — Port SCDE ablations B1, B2 into `rnd_baseline/`

**Goal:** Re-implement your B1 (PPO-only) and B2 (PPO+RND-fixed) ablations
on top of the working baseline so they compare apples-to-apples.

**Steps:**
1. **B1 (PPO-only):** in `train.py`, add a flag `--ablation=ppo_only` that
   zeroes out `IntCoef` and skips RND forward/loss. Verifies your PPO loop
   alone learns FourRooms (Step 0 from the diagnosis doc).
2. **B2 (PPO+RND fixed):** the default `rnd_baseline/` run *is* B2. Just
   tag it.
3. Run both on FourRooms (500k steps) and on Montezuma (10M steps) for
   parity with paper-style ablations.
4. Save final metrics in `rnd_baseline/experiments/ablations/results_B1_B2.csv`.

**CHECKPOINT 4:**
- B2 outperforms B1 on FourRooms by a clear margin (B1 success-rate by
  500k ≤ B2 success-rate by 500k). This is the "RND helps" sanity check.

---

### Session 5 — Port SCDE-specific contributions (B3–B5)

**Goal:** Bring across the three modules called out as "specified but
missing" in `DIAGNOSIS_B1_B2.md` §7 — `reward_normaliser.py` (you actually
get this *for free* from jcwleo's `RewardForwardFilter`), plus
`semantic_memory.py` and `adaptive_weights.py`.

**Steps:**
1. Read `scde/configs/b3_clip_only.yaml`, `b4_rnd_clip_fixed.yaml`,
   `b5_proposed.yaml` to recover the intended algorithmic content.
2. For each, decide whether it is best implemented as:
   - a flag inside `train.py` (small change), or
   - a separate module under `rnd_baseline/scde_extensions/`.
3. Default to the latter so the lifted core stays unmodified.
4. Run B3/B4/B5 ablations on FourRooms first, Montezuma second.

**CHECKPOINT 5:**
- All five ablations (B1..B5) produce learning curves saved to
  `rnd_baseline/experiments/ablations/`. The B5 proposed method either
  beats B2 (success) or doesn't (a clean negative result you can defend).

---

### Session 6 (optional) — Hardening + paper figures

- Seed sweeps (3+ seeds per ablation), error-band plots.
- Wall-clock and sample-efficiency comparisons.
- Move `_upstream/` out of the repo; pin SHA in `README_LIFTED.md`.
- CI smoke test: 10k-step training run that just verifies no crash.

### Session 7 (optional) — Migrate or archive `scde/`

Only once `rnd_baseline/` is stable.

- Option A: archive `scde/` into `legacy/scde/` (needs your consent per
  CLAUDE.md rule #1). All experiments going forward use `rnd_baseline/`.
- Option B: keep both indefinitely; `scde/` is the historical record.

This decision is intentionally not made now.

---

## 6. Verification harness (use every session)

Three quick checks that catch ~all of the bug classes you hit in `scde/`:

1. **Intrinsic reward variance check.** Print `r_int.std()` across envs
   after the first 100 steps. If it's `< 1e-3` you have the `scde/` bug
   (RND not on raw pixels or no obs normalisation).
2. **Shape pin test.** A 20-line Pytest that builds the policy CNN, the
   RND target, the RND predictor, feeds a `(N, 4, 84, 84)` and
   `(N, 1, 84, 84)` batch respectively, and asserts output shapes.
3. **Single-env determinism.** Run 1k steps with `NumEnv=1`, seed 0, and
   diff the log against a saved reference log. Any drift = your code
   diverged from the lifted baseline.

---

## 7. What this plan deliberately defers

- Whether the final repo is `scde/` or `rnd_baseline/`. Locked in
  Session 7 only.
- Cross-game hyperparameter tuning (the paper uses Montezuma-tuned
  hyperparams for all Atari games; we follow suit).
- Multi-GPU. jcwleo is single-GPU + `multiprocessing` env workers.
- envpool / batched-env on Windows. CleanRL's vectorised env layer is
  Linux-only. Not adopted.

---

## 8. Open questions

1. ~~**Compute budget.**~~ **RESOLVED 2026-05-14.** Single RTX 4060 8GB.
   Strategy: train in **2M-frame chunks** with full-state checkpoints
   between chunks (see §3A.1). `NumEnv` reduced to 32 for Atari, 16 for
   MiniGrid, to fit 8GB VRAM. The 50M Montezuma run becomes ~25 chunks
   of ~45 min each — spread across multiple laptop sessions, lid can be
   closed between chunks.
2. **Atari ROMs.** Do you have `AutoROM` set up, or do we need to add an
   install step? (Windows install of `ale-py` is fine; ROMs are the
   licensing hurdle.) — Action: Session 1 Step 4 includes
   `AutoROM --accept-license`.
3. **gym vs gymnasium.** jcwleo uses classic `gym==0.21`. CleanRL has
   migrated to `gymnasium`. Your `scde/` is on classic `gym`. Sticking
   with `gym==0.21` keeps lifted code unchanged; migrating later is its
   own task. — Locked in: pin `gym==0.21.0`.
4. **B5 specification.** What exactly does B5 ("proposed") do
   algorithmically? Session 5 needs this nailed down — the existing
   `b5_proposed.yaml` config doesn't fully specify the method.

---

## 9. Handover hooks (for `HANDOVER.md` between sessions)

At the end of each session, capture:

- **Session number & checkpoint result** (passed/failed/partial).
- **Files added or modified** under `rnd_baseline/` since the last
  handover (use `git diff --stat` against the session-start tag).
- **TensorBoard run name** for the experiment.
- **One-line failure mode** if checkpoint failed.
- **Decision log entry** for any deviation from this plan.

This keeps each new session bootable from a single `HANDOVER.md` read
without re-deriving context.

---

## 10. Sources used to draft this plan (verified 2026-05-14)

- jcwleo repo: <https://github.com/jcwleo/random-network-distillation-pytorch> — file structure, `config.conf`, `model.py`, `utils.py`, `train.py`, open issues #17 / #26, MIT licence file.
- jcwleo `utils.py` (line 4): <https://github.com/jcwleo/random-network-distillation-pytorch/blob/master/utils.py> — confirmed `from torch._six import inf` still present, hence P1.
- jcwleo `envs.py` (lines 8–10): `gym_super_mario_bros` + `nes_py.wrappers.BinarySpaceToDiscreteSpaceEnv` — broken on modern `nes_py`, hence P9.
- jcwleo `train.py` / `eval.py`: hardcoded `is_load_model=True`, no argparse, saves only weights — hence P7+P8.
- jcwleo issue tracker: <https://github.com/jcwleo/random-network-distillation-pytorch/issues> — 10 open issues, none about Python/torch version drift; all algorithmic.
- CleanRL `ppo_rnd_envpool.py`: <https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_rnd_envpool.py> — single-file structure, line landmarks for the four critical pieces, per-mb adv normalisation.
- CleanRL RND docs: <https://docs.cleanrl.dev/rl-algorithms/ppo-rnd/> — referenced for algorithm but VRAM numbers not stated.
- OpenAI RND: <https://github.com/openai/random-network-distillation> — TF1 + MPI status; behaviour-only spec.
- OpenAI CLIP: <https://github.com/openai/CLIP> — requires `torch>=1.7.1` + `torchvision`; install via `pip install ftfy regex tqdm` + `pip install git+https://github.com/openai/CLIP.git`; ViT-B/32 outputs 512-d, ~150 MB weights, expects 224×224 RGB.
- Burda et al. 2018 — *Exploration by Random Network Distillation* (arXiv:1810.12894) — paper constants, §2.4–§2.6 + Appendix A, §4.2 for rooms-visited metric.
- "Pre-Trained Image Encoder for Generalizable Visual RL" (NeurIPS 2022): <https://proceedings.neurips.cc/paper_files/paper/2022/file/548a482d4496ce109cddfbeae5defa7d-Paper-Conference.pdf> — prior art for frozen CLIP encoder in RL.
- Your `DIAGNOSIS_B1_B2.md` — bug catalogue (C1–C5, H1–H4, M1–M4, L1–L2)
  that this rebuild is designed to make obsolete.
