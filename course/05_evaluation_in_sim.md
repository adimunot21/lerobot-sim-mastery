# Chapter 5: Evaluation in Simulation

## The Problem

Training loss going down doesn't mean the policy works. Loss measures "how close are the predicted actions to the demonstration actions on the training data." But a policy runs in a closed loop — its actions change the environment, which changes the observation, which changes the next action. Small prediction errors compound in ways that loss curves can't capture.

The only way to know if a policy works is to run it: let it control the robot in the simulated environment and measure how often it succeeds.

## The Evaluation Loop

The evaluation script (`lerobot-eval`) executes this loop for each episode:

```
1. Reset the simulation (place cube at a new position)
2. Get initial observation (camera image + joint state)
3. While episode not done (max 400 steps):
   a. Feed observation to the policy
   b. Policy predicts an action chunk (100 future actions)
   c. Execute the first n_action_steps actions in the simulator
   d. MuJoCo computes physics (joint motion, contact forces)
   e. Get new observation from the simulator
   f. Check success condition
4. Record: did the cube transfer succeed?
```

This loop runs at the simulation rate (50Hz). Each step, MuJoCo advances the physics by 1/50th of a second. The camera renders the new scene. The policy processes the new image and joint state. The cycle repeats.

## Success Metrics

**pc_success (success rate):** Percentage of episodes where the cube was successfully transferred from the right gripper to the left gripper. This is the primary metric — it directly measures task completion.

**avg_sum_reward:** Cumulative reward over the episode. The environment assigns partial rewards for progress: reaching toward the cube, touching it, lifting it, moving it toward the other arm. Higher reward means the robot got further through the task, even in failed episodes.

**avg_max_reward:** The highest single-step reward achieved. Helps distinguish "got close but failed" from "didn't even try."

**Interpreting the numbers:**

A success rate of 58% means: in 58 out of 100 episodes, the cube was fully transferred. The other 42 episodes failed at various stages — some barely reached for the cube, others picked it up but dropped it during transfer.

Looking at per-episode rewards reveals the failure distribution. In our ACT (Kaggle) eval:
- Episodes with reward ~250-330: full success
- Episodes with reward ~40-200: partial progress (reached/grasped but failed transfer)
- Episodes with reward 0-2: complete failure (didn't even reach the cube)

## Evaluation Batch Size

The `eval.batch_size` parameter controls how many simulation environments run in parallel. With `eval.batch_size=4`, four independent episodes run simultaneously, sharing the GPU for policy inference. This is faster than sequential evaluation (4x on paper, slightly less in practice due to overhead).

However, each parallel environment consumes VRAM:
- Each environment holds its own MuJoCo state and renderer
- The policy processes a batch of 4 observations instead of 1

On our GTX 1650 (4GB), `eval.batch_size=1` was the maximum during training. On the A40 (48GB), we used `eval.batch_size=4` comfortably.

## In-Training vs Post-Training Evaluation

**In-training evaluation** happens every 20k steps during training. The training script pauses, creates simulation environments, runs a few episodes (typically 50), logs the success rate to wandb, then resumes training. This gives you periodic checkpoints with known performance.

The risk: it consumes extra VRAM. On our GTX 1650, in-training eval with `eval.batch_size=4` caused OOM. We fixed this with `eval.batch_size=1`.

**Post-training evaluation** is what we ran with `lerobot-eval`. It loads a checkpoint, runs a specified number of episodes, and reports aggregated metrics. This is more thorough (we ran 50 episodes) and doesn't compete with training for VRAM.

**Always evaluate multiple checkpoints.** Our local ACT showed:
- 60k checkpoint: 46% success
- 80k checkpoint: 44% success

The last checkpoint isn't always the best. Loss can still be decreasing while evaluation performance has already peaked.

## The Evaluation Command Decoded

```bash
lerobot-eval \
    --policy.path=adimunot/act_aloha_transfer_cube_kaggle \  # Model from Hub or local
    --env.type=aloha \                                        # Environment type
    --env.task=AlohaTransferCube-v0 \                         # Task
    --eval.n_episodes=50 \                                    # How many episodes to run
    --eval.batch_size=1 \                                     # Parallel environments
    --output_dir=outputs/eval_act_kaggle                      # Where to save results
```

Output includes:
- `eval_info.json` — structured metrics (success per episode, rewards, aggregates)
- `videos/` — recorded rollout videos (first 10 episodes by default)

## Stochasticity in Evaluation

Running the same policy twice on the same evaluation task can give different success rates. Sources of randomness:
- **Cube initial position** varies between episodes
- **CVAE sampling** — ACT samples z from a Gaussian prior, producing slightly different action plans each time
- **Simulation physics** can have minor numerical variations

This means a 52% vs 58% difference between two runs of the same model is within noise. Don't over-interpret small differences. Run more episodes (100+) if you need precise comparisons.

## What the Rollout Videos Show

Each evaluation episode is recorded as an MP4 video from the top camera. Watching these videos is invaluable for understanding failure modes:

- **Reaching failures:** The arm moves in the wrong direction or overshoots the cube
- **Grasping failures:** The gripper closes too early or too late, missing the cube
- **Transfer failures:** The cube is grasped but dropped during the handoff
- **Timeout failures:** The robot is slowly making progress but runs out of the 400-step budget

For your SO-101 work, watching failure videos will be the primary debugging tool. It tells you what the policy is getting wrong in a way that loss curves and success percentages cannot.

## Evaluation Time: ACT vs Diffusion

Evaluation time depends on the policy's inference speed:

| Policy | Time per Episode | Why |
|--------|-----------------|-----|
| ACT | ~10 seconds | Single forward pass per action chunk |
| Diffusion | ~170 seconds | 100 denoising steps per action chunk |

Diffusion Policy is ~17x slower at inference because each action prediction requires iterating through the full denoising process. For real-time robot control at 50Hz, ACT can easily keep up. Diffusion Policy would need to predict less frequently or use fewer denoising steps (at the cost of action quality).

## Summary

| Concept | Key Point |
|---------|-----------|
| Success rate | Primary metric — percentage of episodes with task completion |
| Compounding errors | Loss doesn't capture closed-loop failure modes |
| Eval batch size | Limited by VRAM, controls parallelism |
| Multiple checkpoints | Best checkpoint ≠ last checkpoint |
| Stochasticity | 5-10% variance between identical eval runs is normal |
| Rollout videos | Most informative debugging tool |

## What's Next

[Chapter 6: Diffusion Policy Deep-Dive →](06_diffusion_policy.md)