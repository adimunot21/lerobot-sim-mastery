# Chapter 4: Training Practicalities

## The Problem

You understand the architecture. You have the data. Now you need to actually train the model. This means: choosing a GPU, managing VRAM, dealing with session timeouts, saving checkpoints, and not losing work. These practical concerns dominate real-world ML projects but rarely get taught.

This chapter covers everything we learned the hard way across three different GPUs and multiple failed runs.

## GPU Selection: What Matters

The two numbers that matter: **VRAM** (determines batch size) and **compute speed** (determines wall-clock time).

| GPU | VRAM | ~Speed (ACT) | Typical Source | Cost |
|-----|------|-------------|----------------|------|
| GTX 1650 | 4 GB | 3.3 step/s | Your laptop | Free |
| T4 | 16 GB | 1.7 step/s | Kaggle/Colab | Free |
| A40 | 48 GB | 4.0 step/s | RunPod | $0.20/hr |
| A100 | 80 GB | ~10 step/s | RunPod/Cloud | $1.20/hr |

Note that the T4 is slower per step than the GTX 1650 despite having more VRAM. This is because the T4's raw compute is modest — its advantage is memory, not speed. This matters: more VRAM lets you use larger batch sizes (better convergence) even if each step is slower.

## VRAM: The Binding Constraint

During training, VRAM holds:
- **Model weights** (~200MB for ACT)
- **Current batch of data** (images + states + action chunks)
- **Intermediate activations** (needed for backpropagation — the network's internal computations at every layer)
- **Optimizer state** (AdamW stores two additional copies of every parameter: first and second moment estimates)

The images are the biggest factor. One 480×640×3 float32 image is 3.7MB. With batch_size=32, that's 118MB just for the raw images — before any processing. After ResNet18, the feature maps are much smaller but the activations through the Transformer are substantial.

**Rule of thumb for ACT with 480×640 images:**
- 4 GB → batch_size=4
- 16 GB → batch_size=16
- 48 GB → batch_size=32
- 80 GB → batch_size=64

## Batch Size vs Training Steps

Our most important finding: **batch size matters more than training steps** for this task.

| Batch Size | Steps | GPU | Success Rate |
|------------|-------|-----|-------------|
| 4 | 80,000 | GTX 1650 | 44% |
| 16 | 40,000 | T4 | 58% |
| 32 | 100,000 | A40 | 52% |

The Kaggle T4 run (bs16, 40k steps) beat the local GTX 1650 run (bs4, 80k steps) despite training for half the steps.

Why? Each training step computes a gradient estimate from `batch_size` samples. With batch_size=4, the gradient is noisy — based on only 4 data points. With batch_size=16, the estimate is 4x more stable. More stable gradients → smoother optimization → better convergence.

There's a diminishing return: going from bs16 to bs32 didn't improve results (58% → 52%). The returns from batch size plateau once the gradient estimate is "stable enough."

## Checkpointing: Saving Your Work

LeRobot saves checkpoints every 20k steps by default. Each checkpoint contains:
- `pretrained_model/` — model weights, config, preprocessor files (everything needed for inference)
- `training_state/` — optimizer state, RNG state, step counter (everything needed to resume training)

**Always check your checkpoints exist** before walking away from a training run. We lost a full Kaggle training run because it exceeded the 12-hour limit before pushing to Hub.

**To resume from a checkpoint:**
```bash
lerobot-train \
    --config_path=outputs/train/act/checkpoints/last/pretrained_model/train_config.json \
    --resume=true
```

## Cloud Training: Lessons Learned

### Kaggle (free T4)
- 12-hour session limit — plan your step count accordingly
- Filesystem is **ephemeral** — everything is deleted when the session ends
- The model only pushes to Hub at the **end** of training. If Kaggle kills your session first, you lose everything unless intermediate checkpoints were manually uploaded
- Internet must be enabled in notebook settings
- `MUJOCO_GL=egl` is required for headless rendering

### RunPod (paid GPUs)
- You pay per hour — stop the pod immediately when done
- The web terminal **kills processes when you close the browser tab**
- Always use `tmux` — it keeps processes alive after tab closure
- Install tmux first: `apt install -y tmux`
- Start training inside tmux: `tmux new -s training`
- Detach: `Ctrl+B` then `D`
- Reattach: `tmux attach -t training`
- The container may not have ffmpeg, EGL libraries, or the right Python version. Budget 10-15 minutes for environment setup.

### Common cloud issues
- **No ffmpeg:** `apt install -y ffmpeg`
- **No EGL rendering:** `apt install -y libegl1-mesa-dev libgles2-mesa-dev libglvnd-dev`
- **Wrong Python version:** lerobot requires ≥3.12. Some containers ship 3.10 or 3.11. Use `python3.12 -m venv` if available, or install via deadsnakes PPA.
- **Output directory exists:** Add `--resume=true` or delete the directory first.

## Local Training: Lessons Learned

### GTX 1650 (4GB VRAM)
- batch_size=4 is the maximum for ACT with 480×640 images
- In-training evaluation OOMs if `eval.batch_size > 1` — the eval creates simulation environments that consume additional VRAM alongside the model
- Training 80k steps took ~7 hours
- Prevent laptop suspend during training: `sudo systemctl mask sleep.target suspend.target hibernate.target`
- Don't close the laptop lid without configuring `HandleLidSwitch=ignore` in logind.conf
- Use tmux for long runs

## Weights & Biases: What to Monitor

The key metrics during ACT training:

**`loss`** — total training loss. Should drop sharply in the first 5-10k steps then gradually flatten.

**`lr` (learning rate)** — follows the configured schedule. Default is a warmup + cosine decay.

**`grdn` (gradient norm)** — magnitude of the gradients. If this spikes suddenly, something went wrong (learning rate too high, corrupted batch). Should be relatively stable after warmup.

**`eval/pc_success`** — success rate at evaluation checkpoints (every 20k steps). This is the number that actually matters.

**What to watch for:**
- Loss not decreasing → fundamentally wrong (data loading issue, architecture mismatch)
- Loss oscillating wildly → learning rate too high
- Loss decreasing but eval success rate not improving → overfitting to training data or evaluation is too noisy
- Gradient norm exploding → reduce learning rate

## The Training Command Decoded

```bash
MUJOCO_GL=egl lerobot-train \
    --policy.type=act \                    # Which policy architecture
    --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \  # Dataset from Hub
    --env.type=aloha \                     # Simulation environment type
    --env.task=AlohaTransferCube-v0 \      # Specific task within the env
    --output_dir=/workspace/train/act \    # Where to save checkpoints
    --steps=100000 \                       # Total training steps
    --batch_size=32 \                      # Samples per gradient update
    --wandb.enable=true \                  # Log to Weights & Biases
    --wandb.project=lerobot-sim-mastery \  # W&B project name
    --policy.repo_id=adimunot/act_a100 \   # Hub repo for the trained model
    --eval.batch_size=4                    # Parallel sim environments for eval
```

`MUJOCO_GL=egl` tells MuJoCo to use EGL for headless rendering — necessary on any system without a display (cloud, SSH, containers).

## Summary

| Lesson | Detail |
|--------|--------|
| VRAM determines batch size | 4GB→bs4, 16GB→bs16, 48GB→bs32 |
| Batch size > training steps | bs16 at 40k beats bs4 at 80k |
| Always use tmux on cloud | Web terminal kills processes on tab close |
| Checkpoint every 20k | Default in lerobot. Don't change this. |
| Monitor wandb | Loss curve tells you if training is working |
| Budget cloud setup time | ffmpeg, EGL, Python version issues are common |

## What's Next

[Chapter 5: Evaluation in Simulation →](05_evaluation_in_sim.md)