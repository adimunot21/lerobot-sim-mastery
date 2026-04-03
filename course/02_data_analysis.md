# Chapter 2: Data Analysis for Robot Learning

## The Problem

Having data isn't enough — you need to know if the data is good. Bad demonstrations produce bad policies. A teleoperator who fumbles a grasp, an episode where the object slipped, a camera that dropped frames — all of these corrupt your training data. Worse, you won't know it until you've spent hours training and the policy fails mysteriously.

This chapter covers the five analyses we run on every dataset before training. These aren't academic exercises — they're the exact checks you'll run on your SO-101 teleop recordings.

## Analysis 1: Action Histograms

**What:** Distribution of each joint's action values across the entire dataset.

**Why it matters:** The shape of the distribution reveals critical information about the task:

A **unimodal distribution** (single peak) means the joint operates in one regime — straightforward for any policy to learn.

A **bimodal distribution** (two peaks) means the joint has two distinct modes. The gripper is the classic example: it's either open (~1.0) or closed (~0.0), with very little time in between. This is important because simple behaviour cloning (predicting the mean) would predict the valley between peaks — a position the robot should never be in. Diffusion Policy handles multimodality better than ACT's L1 loss because it models the full distribution.

A **clipped distribution** (hard cutoff at one end) means demonstrations are hitting joint limits. This could indicate the task requires a different robot configuration or that the teleoperator is fighting mechanical constraints.

**What we found:** The gripper dimensions show clear bimodality (open/closed). Most joint dimensions have roughly Gaussian distributions centered around the task's operating range. No clipping issues.

## Analysis 2: Episode Trajectories

**What:** Each joint's position plotted over time for individual episodes.

**Why it matters:** This is the most intuitive way to "see" a demonstration. You're looking for:

**Smooth curves** — the teleoperator moved fluidly. Good.

**Sudden jumps** — dropped frames or communication glitches between the controller and the arm. These create impossible-to-learn transitions (the robot can't teleport).

**Flat segments** — the robot was stuck or the operator paused. Not necessarily bad, but long pauses dilute the useful data.

**Consistent patterns across episodes** — the same task should look similar each time. If episode 5 looks completely different from episode 1, either the task setup varied or the operator used a different strategy. Strategy variation is fine if your policy can handle multimodality; random variation is noise.

**What we found:** We split plots into left arm and right arm because they serve different roles. The left arm (receiving) shows simple, consistent reach-and-hold patterns. The right arm (grasping) shows more complex trajectories with distinct phases: approach, grasp, lift, transfer. All episodes are smooth and consistent — expected from scripted sim data.

## Analysis 3: Action Smoothness

**What:** The L2 norm of consecutive action differences for every episode. Measures "how much the action changes between frames."

**Why it matters:** Jerky demonstrations → jerky policies. If you imagine a human teleoperating a robot arm, their hand movements are naturally smooth. But teleop systems introduce noise: communication latency, low controller update rates, shaky hands, or bumping the table. This noise propagates directly into the action labels the policy learns from.

We measure three things per episode:
- **Mean action delta:** overall smoothness (lower = smoother)
- **Max action delta:** the single worst jerk
- **Std of action deltas:** consistency of motion speed

**The formula:** For actions $a_t$ at timestep $t$, the delta is $\Delta_t = ||a_{t+1} - a_t||_2$. We compute this for all consecutive pairs.

**What we found:** Mean delta of 0.024 with only 0.001 std across all 50 episodes. Extremely smooth and consistent, as expected from scripted sim data.

**Baseline for real data:** When your SO-101 teleop data shows mean deltas above ~0.1 or episodes with 5x+ the population std, investigate those episodes. They may need to be discarded or the teleop setup may need adjustment.

## Analysis 4: Joint Correlations

**What:** Pearson correlation matrix between all 14 joint dimensions.

**Why it matters:** Reveals the kinematic structure of the task — which joints move together as coordinated motions.

**What we found:**

`left_shoulder ↔ left_elbow: -0.975` — near-perfect anti-correlation. This is classic reaching kinematics: when the shoulder flexes down, the elbow extends up to move the end-effector forward while maintaining height. This is how robot arms (and human arms) reach.

The left arm joints form a tightly correlated block because they're performing a simple, stereotyped receive motion. The right arm doesn't appear in the top correlations — its motion is more complex and varied, consistent with the harder pick-and-transfer task.

**Practical implication:** Strong correlations mean the effective dimensionality of the action space is lower than 14. If the left arm's 7 joints are described by 2-3 independent dimensions (a parameterised reach trajectory), the learning problem is simpler than the raw dimensionality suggests. This is good news — it means policies don't need massive capacity to learn this task.

**For policy architecture decisions:** If the action space has strong structure, simpler models might suffice. If it's mostly uncorrelated (each joint acting independently), you need higher model capacity.

## Analysis 5: Outlier Episode Detection

**What:** Flag episodes where any metric deviates more than 2 standard deviations from the mean.

**Why it matters:** In real data collection, some demonstrations will be bad. The operator fumbled, the object slipped, the gripper didn't close. Training on bad data hurts policy performance — sometimes dramatically. One terrible episode in a dataset of 50 can noticeably degrade the trained policy.

We flag on four metrics:
1. **Episode length** — unusually short (task wasn't completed) or long (operator struggled)
2. **Mean action magnitude** — episode where the robot barely moved (possible failed demo)
3. **Action smoothness** — unusually jerky episode (teleop problems)
4. **Total state range** — total joint travel (unusually large = wild motions, unusually small = robot stuck)

**What we found:** 7 flags across 3 episodes, all mild (z-scores 2.0-2.5). For sim data, this is natural variation, not bad data. Episode 38 was slightly jerkier with more joint travel — likely a wider reaching motion. Episode 47 had lower action magnitude — the robot may have started closer to the cube.

**For real data:** Flag with z > 2, investigate with z > 3. Don't blindly discard — always watch the episode (replay the video or look at trajectory plots) before deciding. Sometimes "outlier" episodes are the most informative (the operator recovered from a mistake, teaching the policy robustness).

## The Analysis Script: What It Does

`src/analyze_dataset.py` runs all five analyses sequentially. At a high level:

1. Loads the dataset from HuggingFace Hub
2. Samples across the dataset collecting action and state tensors
3. Computes per-joint histograms and saves grid plots
4. Loads full episodes and plots joint trajectories over time
5. Computes action deltas for every episode, aggregates smoothness metrics
6. Computes correlation matrix and saves an annotated heatmap
7. Computes per-episode stats and flags outliers with z-score analysis
8. Saves all metrics as JSON and all plots as PNG

The script accepts `--repo-id` to analyze any LeRobotDataset and `--episodes` to select specific episodes for trajectory plots.

## Common Gotchas

**JSON serialization with tensors:** LeRobot returns some metadata fields (like `episode_index`) as PyTorch tensors rather than plain Python ints. When saving to JSON, you need `default=lambda x: x.item() if hasattr(x, "item") else str(x)` to handle this.

**Episode boundaries without `episode_data_index`:** Earlier versions of LeRobot had a convenient `episode_data_index` attribute. Current versions don't — you compute boundaries yourself from `hf_dataset["episode_index"]`.

**Matplotlib on headless systems:** Use `matplotlib.use("Agg")` at the top of any script that generates plots without a display (SSH, containers, Colab).

## Summary

| Analysis | What It Catches | Key Metric |
|----------|----------------|------------|
| Histograms | Multimodality, clipping, degenerate joints | Distribution shape |
| Trajectories | Dropped frames, stuck robots, inconsistent demos | Visual inspection |
| Smoothness | Jerky teleop, noisy demos | Mean action delta L2 |
| Correlations | Kinematic structure, effective dimensionality | Pearson correlation |
| Outliers | Failed demos, equipment issues | Z-score > 2 |

## What's Next

[Chapter 3: ACT Policy Deep-Dive →](03_act_policy.md)