# HANDOVER — Session 1 (in progress → pending Checkpoint 1 verification)

> Updated 2026-05-15. Read this first in a new session. Cross-refs:
> `REBUILD_PLAN.md` (full plan), `DIAGNOSIS_B1_B2.md` (bug catalogue),
> `CLAUDE.md` (project rules).

---

## a. Goal

Build a **known-good ground-truth PPO + RND implementation** in `rnd_baseline/`
beside the untouched `scde/`. Validate it reproduces jcwleo-grade scores on
Montezuma's Revenge (MP4 video + rising metric curves), then port SCDE ablations
B1–B5 on top for apples-to-apples experiments.

**Headline outcome:** see the agent actually learn on Atari — rising
`mean_ext_return`, non-flat `mean_int_reward`, MP4 video proof — not the flat
near-zero return the current `scde/` produces.

---

## b. Current status

### Session 1 work completed (2026-05-15)

| Item | Status |
|---|---|
| `rnd_baseline/_upstream/` cloned from jcwleo (SHA `e383fb95`) | ✓ Done |
| Source files copied into `rnd_baseline/` | ✓ Done |
| P1 — `torch._six` → `torch.inf` in `utils.py` | ✓ Applied |
| P2 — `global_grad_norm_` actually clips now (issue #17 fix) | ✓ Applied |
| P2b — `agents.py` passes `self.clip_grad_norm` to fixed function | ✓ Applied |
| P6 — `tensorboardX` → `torch.utils.tensorboard` in `train.py` + `eval.py` | ✓ Applied |
| P7 — `argparse` + `--resume/--config/--total-frames/--chunk-frames` + `is_load_model` fix | ✓ Applied |
| P8 — full-state checkpoint (model+optimizer+RMS+RNG), atomic write, keep-last-3 | ✓ Applied |
| P9 — Mario imports + `MarioEnvironment` class removed from `envs.py` | ✓ Applied |
| `agents.py` `train_model` returns metrics dict (policy_loss, entropy, grad_norm, KL, EV…) | ✓ Done |
| `config.py` uses `__file__` so notebook imports work from any directory | ✓ Done |
| `train.py` full rewrite — `TrainingContext`, `setup_training()`, `train_one_chunk()`, `eval_checkpoint()` | ✓ Done |
| `requirements.txt` with pinned versions | ✓ Done |
| `README_LIFTED.md` with SHA, patch table, hyperparameter table | ✓ Done |
| `shapes.md` model shape pin file | ✓ Done |
| `S1_smoke_montezuma_1M.ipynb` 6-cell notebook | ✓ Done |
| `fix_gym_install.py` to patch gym==0.21.0 malformed setup.py | ✓ Done |
| `rnd_env` conda env — Python 3.10, torch 2.5.1+cu121, gym 0.21.0, ale-py 0.7.5 | ✓ Verified |
| NumEnv=16 set in config.conf for smoke test | ✓ Done |
| **Notebook Cell 1 + Cell 2** ran successfully (setup, obs-RMS warmup, workers spawned) | ✓ Done |
| **Notebook Cell 3 training** — RUNNING (chunk 1 of 2, at ~204k/500k frames as of handover) | 🔄 In progress |
| Checkpoint 1 verification (ext_return rising, video exists, resume works) | ⏳ Pending next session |

### Early training signal (from Cell 3 stdout, 2026-05-15 04:48):
```
step=   102,400  fps=137  ext_ret= 0.00  int_rew=17.38  ploss=0.0081  ent=2.7674  gnorm=0.145
step=   204,800  fps=146  ext_ret= 0.00  int_rew=11.85  ploss=0.0033  ent=2.8463  gnorm=0.121
```
- `int_rew` is **non-zero and decaying** — real intrinsic signal confirmed (vs. `0.0016` constant in old `scde/`). ✓
- `ext_ret=0.00` is **normal** for Montezuma at 200k frames — agent has not yet reached first reward.
- FPS ~137–146 on RTX 4060 Laptop GPU with NumEnv=16.

### What still needs to happen
- [ ] Cell 3 finishes both chunks (500k + 500k = 1M total frames)
- [ ] Cell 4: eval video confirmed at `runs/S1_smoke_montezuma_1M/videos/`
- [ ] Cell 5: plots show rising `mean_ext_return` and non-flat `mean_int_reward`
- [ ] **P8 resume test**: interrupt Cell 3 between chunks, re-run → no metric jump
- [ ] Checkpoint 1 pass/fail recorded (see `REBUILD_PLAN.md §5`)
- [ ] If Checkpoint 1 passes → proceed to Session 2 (MiniGrid backend, P3/P4/P5)

---

## c. Important context

### File locations

| Path | Purpose |
|---|---|
| `C:\Users\rishe\minor_project_rl\REBUILD_PLAN.md` | Full 5-session plan — read §5 for session goals |
| `C:\Users\rishe\minor_project_rl\DIAGNOSIS_B1_B2.md` | Bug catalogue for old `scde/` code |
| `C:\Users\rishe\minor_project_rl\rnd_baseline\` | New clean implementation (all work here) |
| `rnd_baseline\train.py` | `setup_training()`, `train_one_chunk()`, `save/load_checkpoint()` |
| `rnd_baseline\agents.py` | `RNDAgent` — `train_model()` now returns metrics dict |
| `rnd_baseline\utils.py` | P1+P2 applied — `global_grad_norm_` now clips correctly |
| `rnd_baseline\config.conf` | NumEnv=16 for smoke test (raise to 32 in Session 3) |
| `rnd_baseline\experiments\notebooks\S1_smoke_montezuma_1M.ipynb` | Active notebook |
| `rnd_baseline\runs\S1_smoke_montezuma_1M\` | Created at runtime — checkpoints, logs, videos here |
| `rnd_baseline\README_LIFTED.md` | jcwleo SHA, patch table, hyperparameter table |
| `rnd_baseline\shapes.md` | Model shape pin file (regression guard) |
| `rnd_baseline\fix_gym_install.py` | Patches gym==0.21.0 malformed setup.py for install |
| `C:\Users\rishe\minor_project_rl\scde\` | OLD code — **do not modify** (CLAUDE.md rule #1) |

### Active conda environment
```
conda activate rnd_env
# Python 3.10.20, torch 2.5.1+cu121, gym 0.21.0, ale-py 0.7.5
# RTX 4060 Laptop GPU, CUDA 12.1
```

### Patches NOT yet applied (Session 2+)
| ID | Description |
|---|---|
| P3 | Per-minibatch advantage normalisation (CleanRL transplant) — Session 2 |
| P4 | `MiniGridEnvironment` class in `envs.py` — Session 2 |
| P5 | `[MINIGRID]` section in `config.conf` — Session 2 |

---

## d. Decisions already made (do not re-litigate)

1. **Primary base:** `jcwleo/random-network-distillation-pytorch` (MIT, Windows-runnable, SHA `e383fb95`)
2. **Layout:** `rnd_baseline/` beside `scde/`. `scde/` stays untouched.
3. **gym version:** pinned `gym==0.21.0` + `ale-py==0.7.5`. Not migrating to `gymnasium`.
4. **PyTorch:** `torch 2.5.1+cu121` (RTX 4060 Laptop needs CUDA 12.x).
5. **NumEnv:** 16 for Session 1 smoke test → raise to 32 in Session 3 after VRAM check.
6. **Chunk strategy:** 2M frames per chunk for full runs; 500k for smoke test. Full-state checkpoint at every chunk boundary (P8).
7. **Logging:** `train.log` + `metrics.jsonl` + TensorBoard under `runs/<run>/logs/`.
8. **Video:** `gym.wrappers.RecordVideo` at each chunk eval, MP4 under `runs/<run>/videos/`.
9. **`agents.py` `train_model()`** extended to return a metrics dict (policy_loss, entropy, grad_norm, approx_kl, clipfrac, explained_var_ext/int) for logging. Not in original plan but required for metrics.jsonl.
10. **`config.py`** patched to use `os.path.dirname(__file__)` so it resolves `config.conf` correctly when called from a notebook in a different directory.
11. **`gym==0.21.0` install fix:** `setup.py` has malformed `opencv-python>=3.` (trailing dot). Fixed via `rnd_baseline/fix_gym_install.py` which patches the tarball before install.

---

## e. Suggested first prompt for the next session

> "Read HANDOVER.md. Session 1 Cell 3 has been running. Here are my Cell 3
> results: [paste the final stdout lines showing step, ext_ret, int_ret].
> Here is Cell 4 output: [paste video path]. Here is Cell 5 plot description
> or attach screenshot.
>
> First, tell me if Checkpoint 1 passes (REBUILD_PLAN.md §5 criteria).
> If it passes, move to Session 2: apply P3 (per-minibatch adv normalisation),
> P4 (MiniGridEnvironment), P5 (MINIGRID config section), then create
> S2_fourrooms_500k.ipynb. Show diffs before writing any file."

---

## f. Checkpoint 1 criteria (from REBUILD_PLAN.md §5)

All must pass before starting Session 2:

- [ ] `metrics.jsonl` shows `mean_ext_return` rising above 0 within 1M frames
- [ ] `mean_int_reward` is non-flat across envs (already confirmed at 204k steps ✓)
- [ ] No NaN in any loss, no shape errors
- [ ] MP4 eval video exists at `runs/S1_smoke_montezuma_1M/videos/`
- [ ] Interrupt Cell 3 mid-run, re-run → resumes from last chunk checkpoint with no metric discontinuity (P8 validation)

**Partial early evidence:** `int_rew=17.38` at 102k steps — RND signal confirmed real. ✓

---

## g. Known risks / watch points for verification

- `ext_ret=0.00` at 200k is normal for Montezuma. Expect first non-zero return between 500k–1M frames. If still 0.00 at end of 1M, training is still valid (Montezuma is hard) but note it.
- The P8 resume test is the most important thing to verify in the next session. Run Cell 3, interrupt at the 500k chunk boundary (after `[ckpt] saved →`), then re-run Cell 3 — the step counter should print `Resumed from step 500,000` and `int_rew` should continue smoothly without jumping.
- FPS ~137–146 at NumEnv=16. Expected ~250–350 FPS at NumEnv=32 (Session 3). If FPS drops below 80, check for CPU bottleneck in the env workers.
