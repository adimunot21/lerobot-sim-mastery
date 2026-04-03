# Chapter 7: Policy Comparison

## The Problem

You have two policy architectures, multiple training runs across different hardware, and a collection of success rates. How do you make a fair comparison? And more importantly: which policy should you actually use when your robot arm arrives?

## The Full Results Table

| Policy | Steps | Batch Size | GPU | Success Rate | Avg Reward | Eval Time/Ep |
|--------|-------|------------|-----|-------------|------------|-------------|
| ACT (local) | 80k | 4 | GTX 1650 | 44% | 148 | ~10s |
| ACT (Kaggle) | 40k | 16 | T4 | 58% | 187 | ~10s |
| ACT (A40) | 100k | 32 | A40 | 52% | 157 | ~10s |
| Diffusion (Kaggle) | 35k | 16 | T4 | 10% | 29 | ~170s |
| ACT Reference | 80k | 8 | A100 | ~90% | — | — |

## Finding 1: Batch Size Matters More Than Training Steps

ACT with bs16 at 40k steps (58%) significantly outperformed ACT with bs4 at 80k steps (44%). The model saw half the training iterations but each gradient update was 4x more reliable.

**Why:** Gradient noise. With batch_size=4, the gradient computed at each step is based on 4 random samples. It points roughly in the right direction but with high variance. The optimizer takes a noisy step, sometimes improving, sometimes worsening. Over many steps, progress is slow and inconsistent.

With batch_size=16, the gradient is based on 16 samples — the variance drops by 4x (variance scales as 1/batch_size). The optimizer takes more confident steps. The model converges faster and to a better minimum.

**The diminishing returns:** Going from bs16 to bs32 didn't help further (58% → 52%). Once the gradient is "stable enough," additional batch size doesn't contribute. The optimal batch size for this task appears to be around 16.

## Finding 2: ACT Has a Performance Ceiling Around 52-58%

Three separate ACT runs with different compute budgets all landed in the 44-58% range. The gap to the reference 90% persists regardless of:
- Training steps (40k vs 80k vs 100k)
- Batch size (4 vs 16 vs 32)
- GPU (GTX 1650 vs T4 vs A40)

This suggests the gap isn't about compute — it's about **hyperparameters**. The default ACT configuration in the current LeRobot version may not be optimally tuned for this task. Possible factors:
- Learning rate schedule (warmup length, decay rate)
- Chunk size and n_action_steps
- ResNet18 backbone vs other visual encoders
- CVAE latent dimension and KL weight

The reference model was likely trained with a carefully tuned configuration that may not match the current LeRobot defaults.

**Lesson for SO-101:** Don't assume that default hyperparameters are optimal. If your first run gives 50% success, try adjusting learning rate and chunk size before throwing more compute at it.

## Finding 3: Diffusion Policy Needs 5-10x More Training

At 35k steps, Diffusion achieved only 10% — barely above random. ACT at the same step count would already be at ~50%. This is inherent to the architecture:

- ACT learns a direct mapping (observation → actions). Simple, fast to converge.
- Diffusion learns to denoise at 100 different noise levels. Each level is effectively a different task. Convergence requires seeing many more examples.

Published Diffusion Policy results use 200k-500k steps. Our 35k run was ~6x too short. Had we trained for 200k, we'd expect 60-85% based on published results — competitive with ACT on this task.

## Finding 4: Inference Speed Is a Real Constraint

ACT: ~10 seconds per evaluation episode (50Hz, 400 steps).
Diffusion: ~170 seconds per evaluation episode.

For real-time robot control, this matters enormously. The SO-101 operates at ~30fps. ACT can compute action chunks well within the budget. Diffusion Policy would need to run with fewer denoising steps (DDIM with 10 steps instead of 100), which degrades action quality.

## How We Made the Comparison Fair

For the ACT vs Diffusion comparison to be meaningful, we controlled as many variables as possible:

**Same dataset:** `lerobot/aloha_sim_transfer_cube_human` — 50 episodes, 20,000 frames.

**Same task:** `AlohaTransferCube-v0` — identical success criteria.

**Same evaluation protocol:** 50 episodes, batch_size=1, same evaluation script.

**Same GPU for the key comparison:** Kaggle T4, batch_size=16 for both ACT (40k) and Diffusion (35k).

**What we couldn't control:** Step count (35k vs 40k — close enough) and the inherent architectural differences in convergence speed. A "truly fair" comparison would train both to convergence (ACT at 80k, Diffusion at 200k) — but our compute budget didn't allow Diffusion to converge.

## Decision Framework for SO-101

```
Start with ACT
    │
    ├── Success rate acceptable? → Deploy. Done.
    │
    ├── Success rate low, task is unimodal?
    │   → Tune hyperparameters (learning rate, chunk size)
    │   → Collect more demonstrations
    │   → Check data quality (run analyze_dataset.py)
    │
    └── Success rate low, task has multiple valid strategies?
        → Try Diffusion Policy with 100k+ steps
        → If Diffusion works better → Deploy Diffusion
        → If similar → Stick with ACT (faster inference)
```

## The Comparison Report Generator

`src/compare_policies.py` automates this analysis. It:

1. Reads `eval_info.json` from each evaluation directory
2. Extracts success rates, rewards, per-episode success/failure
3. Generates four visualizations:
   - **Success rate bar chart** — direct comparison across all runs
   - **Reward bar chart** — average episode reward
   - **Per-episode success heatmap** — green/red grid showing which episodes each policy solved
   - **Training efficiency scatter** — steps vs success rate, showing the compute-performance tradeoff
4. Writes a structured markdown report with findings and recommendations

The script is configurable via the `EVAL_RUNS` list — add new runs by specifying the eval directory, metadata, and the script handles the rest.

## Summary

| Finding | Implication |
|---------|-------------|
| Batch size > training steps | Prioritize VRAM (larger batch) over training time (more steps) |
| ACT plateaus at 52-58% | Default hyperparameters may not be optimal; tuning needed |
| Diffusion needs 5-10x more steps | Only use when ACT fails on multimodal tasks |
| Inference speed: ACT >> Diffusion | ACT for real-time control, Diffusion for offline/slow tasks |
| Hyperparameters > compute | Tuning matters more than raw GPU power beyond a threshold |

## What's Next

[Chapter 8: From Sim to Real →](08_sim_to_real.md)