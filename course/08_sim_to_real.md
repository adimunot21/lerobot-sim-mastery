# Chapter 8: From Sim to Real — What Changes With the SO-101

## What Stays the Same

The entire software pipeline we built transfers directly:

**Dataset format:** Your SO-101 demonstrations will be stored as LeRobotDatasets — Parquet files for joint positions and actions, MP4 videos for camera feeds. Same format, same tools.

**Analysis toolkit:** Run `inspect_dataset.py` and `analyze_dataset.py` on your teleop recordings. Same smoothness checks, same correlation analysis, same outlier detection. The thresholds will be different (real data is noisier) but the methodology is identical.

**Training:** `lerobot-train` with ACT or Diffusion Policy. Same CLI, same wandb monitoring, same checkpointing. The only change is the dataset repo ID.

**Evaluation:** Instead of `lerobot-eval` running in simulation, you'll use `lerobot-record` with `--policy.path` to run the trained policy on the real robot and record evaluation episodes.

## What Changes

### 1. Data Collection

In simulation, we had 50 perfect demonstrations. In reality, you need to record them yourself.

**The recording command for SO-101:**
```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyUSB0 \
    --robot.id=my_follower_arm \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyUSB1 \
    --teleop.id=my_leader_arm \
    --display_data=true \
    --dataset.repo_id=${HF_USER}/so101_pick_place \
    --dataset.num_episodes=50 \
    --dataset.single_task="Pick up the cube and place it in the bin"
```

**Key differences from sim data:**
- Episodes will have variable lengths (you might take 5 seconds or 15 seconds)
- Action smoothness will be lower (human teleop is inherently less smooth)
- Some episodes will be failures (you fumble the grasp)
- Camera quality varies (lighting, occlusion, reflections)
- The action space is 6-DOF (SO-101 has 5 joints + 1 gripper), not 14-DOF like ALOHA

### 2. Calibration

Before any data collection, the SO-101 leader and follower arms must be calibrated so they share the same joint position space. If the leader arm reads 0.5 radians on joint 3 but the follower's joint 3 is physically at a different angle, the policy learns corrupted data.

```bash
lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyUSB0 \
    --robot.id=my_follower_arm
```

**Use the same robot ID consistently** across calibration, recording, and evaluation. The ID is tied to the calibration file.

### 3. Camera Setup

In simulation, the camera is perfect: fixed position, perfect focus, no noise, consistent lighting. In reality:

- **Mount the camera rigidly.** Any vibration during recording corrupts the visual observations.
- **Consistent lighting.** Train with the same lighting you'll deploy with. A policy trained under fluorescent lights may fail under natural light.
- **Resolution tradeoff:** Higher resolution (1920×1080) gives more detail but uses more VRAM during training and slows everything down. 640×480 is a good starting point for the SO-101.
- **Frame rate:** 30fps is standard for real-world recording. The ALOHA sim used 50fps — you'll need to adjust `delta_timestamps` if your policy expects a specific frame rate.

### 4. Data Quality Is Harder

In simulation, every demonstration succeeds perfectly. In teleop:

- **Discard failed episodes.** If you drop the cube, that episode teaches the policy to drop cubes. Mark it for deletion.
- **Aim for consistency.** Use the same grasp strategy for all episodes. If you approach from the left sometimes and the right other times, you're introducing multimodality that requires more data and a more capable policy.
- **50 demonstrations is the minimum.** Published SO-100/101 results use 50-100 episodes. More data is always better, but quality matters more than quantity — 30 perfect demos outperform 100 sloppy ones.

**Run analysis immediately after recording:**
```bash
python src/analyze_dataset.py --repo-id ${HF_USER}/so101_pick_place
```

Check:
- Action smoothness (compare to our sim baseline of 0.024)
- Outlier episodes (z > 3 → watch the video before keeping)
- Joint correlations (should reveal the kinematic structure of your task)
- Episode length distribution (very short = failed demos to remove)

### 5. The Sim-to-Real Gap

A policy trained in simulation won't work on a real robot. The visual appearance is completely different (rendered vs real), the physics don't perfectly match (contact dynamics, friction), and the control dynamics differ (motor response time, backlash in gears).

This is why we train on real demonstrations, not sim data. Our sim project was about learning the workflow, not producing a deployable policy.

However, the **analytical skills** transfer perfectly:
- Understanding what good data looks like (smooth actions, consistent trajectories)
- Knowing how to diagnose training failures (check wandb curves, evaluate checkpoints)
- Having intuition for hyperparameter effects (batch size, training steps)

### 6. Training on Real Data

The training command barely changes:

```bash
lerobot-train \
    --policy.type=act \
    --dataset.repo_id=${HF_USER}/so101_pick_place \
    --output_dir=outputs/train/act_so101 \
    --steps=50000 \
    --batch_size=16 \
    --wandb.enable=true \
    --wandb.project=so101-pick-place \
    --policy.repo_id=${HF_USER}/act_so101_pick_place \
    --policy.device=cuda
```

Note: no `--env.type` or `--env.task` — those are for simulation evaluation. For real robot training, you skip in-training eval (there's no simulator to evaluate in).

### 7. Deployment

Running the trained policy on the real robot:

```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyUSB0 \
    --robot.id=my_follower_arm \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --display_data=false \
    --dataset.repo_id=${HF_USER}/eval_act_so101 \
    --dataset.single_task="Pick up the cube and place it in the bin" \
    --policy.path=${HF_USER}/act_so101_pick_place
```

The `--policy.path` argument turns recording into evaluation mode — the policy controls the robot instead of the teleoperator. The robot's actions are recorded for later analysis.

## Expected Gotchas

**USB port changes.** Every time you unplug and replug the robot, the port (/dev/ttyUSB0) may change. Use `lerobot-find-port` to identify the correct port.

**Camera index changes.** Webcam indices can shift when other USB devices are connected. Verify with `ls /dev/video*` before recording.

**Files in .gitignore won't be on GitHub.** When cloning your repo to a different machine (e.g., for training on a cloud GPU), remember that data files, model checkpoints, and .env files won't be there. Download datasets from Hub, re-create .env.

**Network dependencies.** The first time you run `lerobot-train` with a new dataset repo_id, it downloads from Hub. Make sure you have internet access on your training machine.

**Motor overheating.** During long evaluation runs, servos can overheat. Give the robot breaks between batches of evaluation episodes.

## The Recommended Workflow

```
1. Assemble and calibrate SO-101 (leader + follower)
2. Set up camera, test teleoperation
3. Record 5 test episodes → run inspect_dataset.py → verify data looks right
4. Record 50 episodes of the target task
5. Run analyze_dataset.py → check quality, remove bad episodes
6. Push dataset to Hub
7. Train ACT (start with defaults, 50k steps, bs16)
8. Evaluate on robot → check success rate
9. If success rate low:
   a. Check data quality (most common issue)
   b. Tune hyperparameters
   c. Collect more demonstrations
   d. Try Diffusion Policy if multimodality suspected
10. Iterate until satisfactory
```

## What We'd Do Differently

Looking back at this project, three things we'd change:

**1. Hyperparameter search early.** We spent compute on larger GPUs when the bottleneck was likely hyperparameters. A small grid search over learning rate and chunk size on the T4 might have been more valuable than the A40 run.

**2. Fewer training runs, more analysis.** We trained 4 ACT runs and 1 Diffusion run. Two ACT runs (one baseline, one with bs16) plus deeper analysis of failure modes from the rollout videos would have taught us more.

**3. Test rendering setup before training on cloud.** We hit EGL, ffmpeg, and Python version issues on both Kaggle and RunPod. A 2-minute rendering test before committing to a multi-hour training run saves time and money.

## Summary

| Sim vs Real | Sim | Real |
|------------|-----|------|
| Data source | Pre-recorded dataset on Hub | Your teleop recordings |
| Data quality | Perfect, consistent | Variable, needs curation |
| Action space | 14-DOF (ALOHA dual arm) | 6-DOF (SO-101 single arm) |
| Camera | Rendered, perfect | Real webcam, lighting varies |
| Evaluation | Automated sim rollouts | Manual robot testing |
| Debugging | Loss curves + sim videos | Loss curves + real videos + physical inspection |
| What transfers | The entire pipeline, tools, and intuition |

## Course Complete

You now have end-to-end fluency with:
- The LeRobotDataset format and how to inspect/analyze it
- ACT and Diffusion Policy architectures and their tradeoffs
- The full training pipeline across local, cloud, and GPU-rented compute
- Evaluation methodology and how to interpret results
- Practical debugging skills for data, training, and deployment

When the SO-101 arrives, you're ready.# Chapter 8: From Sim to Real — What Changes With the SO-101

## What Stays the Same

The entire software pipeline we built transfers directly:

**Dataset format:** Your SO-101 demonstrations will be stored as LeRobotDatasets — Parquet files for joint positions and actions, MP4 videos for camera feeds. Same format, same tools.

**Analysis toolkit:** Run `inspect_dataset.py` and `analyze_dataset.py` on your teleop recordings. Same smoothness checks, same correlation analysis, same outlier detection. The thresholds will be different (real data is noisier) but the methodology is identical.

**Training:** `lerobot-train` with ACT or Diffusion Policy. Same CLI, same wandb monitoring, same checkpointing. The only change is the dataset repo ID.

**Evaluation:** Instead of `lerobot-eval` running in simulation, you'll use `lerobot-record` with `--policy.path` to run the trained policy on the real robot and record evaluation episodes.

## What Changes

### 1. Data Collection

In simulation, we had 50 perfect demonstrations. In reality, you need to record them yourself.

**The recording command for SO-101:**
```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyUSB0 \
    --robot.id=my_follower_arm \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyUSB1 \
    --teleop.id=my_leader_arm \
    --display_data=true \
    --dataset.repo_id=${HF_USER}/so101_pick_place \
    --dataset.num_episodes=50 \
    --dataset.single_task="Pick up the cube and place it in the bin"
```

**Key differences from sim data:**
- Episodes will have variable lengths (you might take 5 seconds or 15 seconds)
- Action smoothness will be lower (human teleop is inherently less smooth)
- Some episodes will be failures (you fumble the grasp)
- Camera quality varies (lighting, occlusion, reflections)
- The action space is 6-DOF (SO-101 has 5 joints + 1 gripper), not 14-DOF like ALOHA

### 2. Calibration

Before any data collection, the SO-101 leader and follower arms must be calibrated so they share the same joint position space. If the leader arm reads 0.5 radians on joint 3 but the follower's joint 3 is physically at a different angle, the policy learns corrupted data.

```bash
lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyUSB0 \
    --robot.id=my_follower_arm
```

**Use the same robot ID consistently** across calibration, recording, and evaluation. The ID is tied to the calibration file.

### 3. Camera Setup

In simulation, the camera is perfect: fixed position, perfect focus, no noise, consistent lighting. In reality:

- **Mount the camera rigidly.** Any vibration during recording corrupts the visual observations.
- **Consistent lighting.** Train with the same lighting you'll deploy with. A policy trained under fluorescent lights may fail under natural light.
- **Resolution tradeoff:** Higher resolution (1920×1080) gives more detail but uses more VRAM during training and slows everything down. 640×480 is a good starting point for the SO-101.
- **Frame rate:** 30fps is standard for real-world recording. The ALOHA sim used 50fps — you'll need to adjust `delta_timestamps` if your policy expects a specific frame rate.

### 4. Data Quality Is Harder

In simulation, every demonstration succeeds perfectly. In teleop:

- **Discard failed episodes.** If you drop the cube, that episode teaches the policy to drop cubes. Mark it for deletion.
- **Aim for consistency.** Use the same grasp strategy for all episodes. If you approach from the left sometimes and the right other times, you're introducing multimodality that requires more data and a more capable policy.
- **50 demonstrations is the minimum.** Published SO-100/101 results use 50-100 episodes. More data is always better, but quality matters more than quantity — 30 perfect demos outperform 100 sloppy ones.

**Run analysis immediately after recording:**
```bash
python src/analyze_dataset.py --repo-id ${HF_USER}/so101_pick_place
```

Check:
- Action smoothness (compare to our sim baseline of 0.024)
- Outlier episodes (z > 3 → watch the video before keeping)
- Joint correlations (should reveal the kinematic structure of your task)
- Episode length distribution (very short = failed demos to remove)

### 5. The Sim-to-Real Gap

A policy trained in simulation won't work on a real robot. The visual appearance is completely different (rendered vs real), the physics don't perfectly match (contact dynamics, friction), and the control dynamics differ (motor response time, backlash in gears).

This is why we train on real demonstrations, not sim data. Our sim project was about learning the workflow, not producing a deployable policy.

However, the **analytical skills** transfer perfectly:
- Understanding what good data looks like (smooth actions, consistent trajectories)
- Knowing how to diagnose training failures (check wandb curves, evaluate checkpoints)
- Having intuition for hyperparameter effects (batch size, training steps)

### 6. Training on Real Data

The training command barely changes:

```bash
lerobot-train \
    --policy.type=act \
    --dataset.repo_id=${HF_USER}/so101_pick_place \
    --output_dir=outputs/train/act_so101 \
    --steps=50000 \
    --batch_size=16 \
    --wandb.enable=true \
    --wandb.project=so101-pick-place \
    --policy.repo_id=${HF_USER}/act_so101_pick_place \
    --policy.device=cuda
```

Note: no `--env.type` or `--env.task` — those are for simulation evaluation. For real robot training, you skip in-training eval (there's no simulator to evaluate in).

### 7. Deployment

Running the trained policy on the real robot:

```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyUSB0 \
    --robot.id=my_follower_arm \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --display_data=false \
    --dataset.repo_id=${HF_USER}/eval_act_so101 \
    --dataset.single_task="Pick up the cube and place it in the bin" \
    --policy.path=${HF_USER}/act_so101_pick_place
```

The `--policy.path` argument turns recording into evaluation mode — the policy controls the robot instead of the teleoperator. The robot's actions are recorded for later analysis.

## Expected Gotchas

**USB port changes.** Every time you unplug and replug the robot, the port (/dev/ttyUSB0) may change. Use `lerobot-find-port` to identify the correct port.

**Camera index changes.** Webcam indices can shift when other USB devices are connected. Verify with `ls /dev/video*` before recording.

**Files in .gitignore won't be on GitHub.** When cloning your repo to a different machine (e.g., for training on a cloud GPU), remember that data files, model checkpoints, and .env files won't be there. Download datasets from Hub, re-create .env.

**Network dependencies.** The first time you run `lerobot-train` with a new dataset repo_id, it downloads from Hub. Make sure you have internet access on your training machine.

**Motor overheating.** During long evaluation runs, servos can overheat. Give the robot breaks between batches of evaluation episodes.

## The Recommended Workflow

```
1. Assemble and calibrate SO-101 (leader + follower)
2. Set up camera, test teleoperation
3. Record 5 test episodes → run inspect_dataset.py → verify data looks right
4. Record 50 episodes of the target task
5. Run analyze_dataset.py → check quality, remove bad episodes
6. Push dataset to Hub
7. Train ACT (start with defaults, 50k steps, bs16)
8. Evaluate on robot → check success rate
9. If success rate low:
   a. Check data quality (most common issue)
   b. Tune hyperparameters
   c. Collect more demonstrations
   d. Try Diffusion Policy if multimodality suspected
10. Iterate until satisfactory
```

## What We'd Do Differently

Looking back at this project, three things we'd change:

**1. Hyperparameter search early.** We spent compute on larger GPUs when the bottleneck was likely hyperparameters. A small grid search over learning rate and chunk size on the T4 might have been more valuable than the A40 run.

**2. Fewer training runs, more analysis.** We trained 4 ACT runs and 1 Diffusion run. Two ACT runs (one baseline, one with bs16) plus deeper analysis of failure modes from the rollout videos would have taught us more.

**3. Test rendering setup before training on cloud.** We hit EGL, ffmpeg, and Python version issues on both Kaggle and RunPod. A 2-minute rendering test before committing to a multi-hour training run saves time and money.

## Summary

| Sim vs Real | Sim | Real |
|------------|-----|------|
| Data source | Pre-recorded dataset on Hub | Your teleop recordings |
| Data quality | Perfect, consistent | Variable, needs curation |
| Action space | 14-DOF (ALOHA dual arm) | 6-DOF (SO-101 single arm) |
| Camera | Rendered, perfect | Real webcam, lighting varies |
| Evaluation | Automated sim rollouts | Manual robot testing |
| Debugging | Loss curves + sim videos | Loss curves + real videos + physical inspection |
| What transfers | The entire pipeline, tools, and intuition |

## Course Complete

You now have end-to-end fluency with:
- The LeRobotDataset format and how to inspect/analyze it
- ACT and Diffusion Policy architectures and their tradeoffs
- The full training pipeline across local, cloud, and GPU-rented compute
- Evaluation methodology and how to interpret results
- Practical debugging skills for data, training, and deployment

When the SO-101 arrives, you're ready.