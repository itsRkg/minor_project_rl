# rnd_baseline — Model shape pin file

Any change to a model's input or output shape MUST update this file in the same commit.
This file is the regression guard against silent shape drift (see REBUILD_PLAN.md §4 rule 4).

---

## Policy network — `CnnActorCriticNetwork`  (`model.py`)

| Layer | Input shape | Output shape | Notes |
|---|---|---|---|
| Input | `(N, 4, 84, 84)` | — | 4-frame stack of grayscale 84×84 Atari frames, pixel range [0,1] |
| Conv1 | `(N, 4, 84, 84)` | `(N, 32, 20, 20)` | 8×8 kernel, stride 4 |
| Conv2 | `(N, 32, 20, 20)` | `(N, 64, 9, 9)` | 4×4 kernel, stride 2 |
| Conv3 | `(N, 64, 9, 9)` | `(N, 64, 7, 7)` | 3×3 kernel, stride 1 |
| Flatten | `(N, 64, 7, 7)` | `(N, 3136)` | |
| FC shared | `(N, 3136)` | `(N, 448)` | |
| policy head | `(N, 448)` | `(N, n_actions)` | logits (pass through softmax for probs) |
| critic_ext | `(N, 448)` | `(N, 1)` | extrinsic value |
| critic_int | `(N, 448)` | `(N, 1)` | intrinsic value |

## RND target network — `RNDModel.target`  (`model.py`)

| Layer | Input shape | Output shape | Notes |
|---|---|---|---|
| Input | `(N, 1, 84, 84)` | — | **Single** grayscale frame (NOT 4-stack); obs-normalised, clipped ±5 |
| Conv1 | `(N, 1, 84, 84)` | `(N, 32, 20, 20)` | 8×8, stride 4, LeakyReLU |
| Conv2 | `(N, 32, 20, 20)` | `(N, 64, 9, 9)` | 4×4, stride 2, LeakyReLU |
| Conv3 | `(N, 64, 9, 9)` | `(N, 64, 7, 7)` | 3×3, stride 1, LeakyReLU |
| Flatten | `(N, 64, 7, 7)` | `(N, 3136)` | |
| FC1 | `(N, 3136)` | `(N, 512)` | |
| Output | `(N, 512)` | `(N, 512)` | **frozen** (requires_grad=False) |

## RND predictor network — `RNDModel.predictor`  (`model.py`)

Same conv backbone as target, then:

| Layer | Input | Output | Notes |
|---|---|---|---|
| FC1 | `(N, 3136)` | `(N, 512)` | |
| FC2 | `(N, 512)` | `(N, 512)` | |
| Output | `(N, 512)` | `(N, 512)` | trained to match target |

Intrinsic reward = `((target - predictor)^2).sum(1) / 2`, shape `(N,)`

---

## Key invariants

1. **Policy input ≠ RND input.** Policy: 4-channel stacked (history). RND: 1-channel latest frame (obs-normalised).
2. **RND target is frozen.** `rnd.target.requires_grad = False` at init; verify after any model surgery.
3. **Obs normalisation applies only to RND input**, not to the policy input (policy gets raw /255 pixels).
