# rnd_baseline — Lifted from jcwleo

## Upstream source

| Field | Value |
|---|---|
| Repo | https://github.com/jcwleo/random-network-distillation-pytorch |
| Commit SHA pinned | `e383fb95177c50bfdcd81b43e37c443c8cde1d94` |
| Licence | MIT (see `_upstream/LICENSE`) |
| Cloned | 2026-05-15 |
| Last meaningful upstream activity | Mar 2024 |

Files in `_upstream/` are the verbatim upstream source at the pinned SHA.
Files in the parent directory (`rnd_baseline/`) are the patched working copies.

---

## Patches applied on top of jcwleo

| ID | File | Summary | Reason |
|---|---|---|---|
| P1 | `utils.py` line 5 | `from torch._six import inf` → `from torch import inf` | `torch._six` removed in PyTorch ≥1.13 |
| P2 | `utils.py` | `global_grad_norm_` now delegates to `torch.nn.utils.clip_grad_norm_` | jcwleo issue #17: norm was computed but gradients were never rescaled |
| P2b | `agents.py` | Pass `self.clip_grad_norm` to `global_grad_norm_` | Required by P2's new signature |
| P6 | `train.py`, `eval.py` | `tensorboardX` → `torch.utils.tensorboard` | tensorboardX unmaintained; native TB is in every PyTorch install |
| P7 | `train.py` | Add `argparse` (`--resume`, `--config`, `--total-frames`, `--chunk-frames`, `--run-name`, `--seed`). Fix `is_load_model = True` → derive from `--resume` | First-run crash fix; enables 2M-chunk strategy on RTX 4060 |
| P8 | `train.py` | Full-state checkpoint (model + optimiser + obs_rms + reward_rms + reward_filter + RNG). Atomic write. Keep last 3 + milestones | jcwleo saves weights only — resume loses normalisation stats |
| P9 | `envs.py` | Remove `gym_super_mario_bros` / `nes_py` imports + `MarioEnvironment` class | Not targeting Mario; `BinarySpaceToDiscreteSpaceEnv` broken on modern `nes_py` / Windows |

---

## Hyperparameter table (Montezuma DEFAULT)

Every hyperparameter listed here; source column shows which repo's value was used when they disagreed.

| Param | Value | Source | Notes |
|---|---|---|---|
| EnvID | `MontezumaRevengeNoFrameskip-v4` | jcwleo | |
| NumEnv | **16** (smoke), 32 (full) | ours | jcwleo default 128; reduced for RTX 4060 8GB |
| NumStep | 128 | jcwleo / Burda | |
| LearningRate | 1e-4 | jcwleo / Burda | |
| Gamma (ext) | 0.999 | jcwleo / Burda | |
| IntGamma | 0.99 | jcwleo / Burda | |
| Lambda | 0.95 | jcwleo | |
| Epoch | 4 | jcwleo / Burda | |
| MiniBatch | 4 | jcwleo | effective batch = NumEnv×NumStep/MiniBatch |
| PPOEps | 0.1 | jcwleo / Burda | |
| ExtCoef | 2.0 | jcwleo / Burda | |
| IntCoef | 1.0 | jcwleo / Burda | |
| UpdateProportion | 0.25 | jcwleo / Burda | fraction of RND predictor updates |
| ObsNormStep | 50 | jcwleo / Burda | random-policy rollouts for obs RMS warmup |
| StickyAction | True, p=0.25 | Burda §2.3 | |
| ClipGradNorm | 0.5 | jcwleo | now actually applied (P2) |
| Entropy | 0.001 | jcwleo | |
| MaxStepPerEpisode | 4500 | jcwleo | |

---

## Cross-reference repos (not lifted, read-only)

- **CleanRL** `cleanrl/ppo_rnd_envpool.py` — used for per-minibatch advantage normalisation (P3, Session 2). MIT.
- **OpenAI RND** `ppo_agent.py` — TF1+MPI, paper-spec only. No code lifted.
