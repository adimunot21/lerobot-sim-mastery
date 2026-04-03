# Chapter 0: Introduction

## The Problem

You have a robot arm arriving soon. When it gets here, the workflow looks like this:

1. Teleoperate the arm to record demonstrations of the task you want it to learn
2. Package those demonstrations as a structured dataset
3. Train a neural network policy on the demonstrations
4. Deploy that policy so the robot acts autonomously

Each step has its own tools, formats, failure modes, and debugging requirements. Trying to learn all of them simultaneously while also dealing with physical hardware (calibration, USB ports, motor drivers, camera alignment) is a recipe for frustration.

This project exists to learn the entire pipeline in simulation first — where there are no cables to unplug, no motors to calibrate, and no objects falling off the table. When the arm arrives, the only new variable is the physical robot. Everything else — the dataset format, the training loop, the evaluation protocol, the debugging tools — you already know cold.

## What We Built

This project covers the complete LeRobot workflow using ALOHA simulation manipulation tasks:

**Dataset Mastery (Phases 1-2):** Reusable tools to inspect and analyze any LeRobotDataset. Field profiling, value range analysis, action smoothness metrics, joint correlation analysis, outlier episode detection. These tools transfer directly to real-world data quality checks.

**Policy Training (Phases 3-5):** End-to-end training of two policy architectures — ACT (Action Chunking Transformer) and Diffusion Policy — on the ALOHA transfer cube task. Training ran across three different GPUs (GTX 1650, Kaggle T4, RunPod A40) providing practical lessons about batch size, VRAM constraints, and compute tradeoffs.

**Evaluation & Comparison (Phases 4-6):** Systematic evaluation in simulation with success rate metrics. Structured comparison report with visualizations answering: which policy should you use, and why?

## System Architecture

```
HF Hub (datasets/models)
         │
         ▼
┌─────────────────┐     ┌──────────────────┐
│ LeRobotDataset  │────▶│ inspect_dataset  │──▶ outputs/inspection/
│ (Parquet + MP4) │     │ analyze_dataset  │──▶ outputs/analysis/
└────────┬────────┘     └──────────────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────────┐
│ lerobot-train   │────▶│ ACT / Diffusion  │──▶ checkpoints → HF Hub
│ (GPU training)  │     │ policy training  │
└────────┬────────┘     └──────────────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────────┐
│ lerobot-eval    │────▶│ compare_policies │──▶ outputs/comparison/
│ (sim rollouts)  │     │ (report + plots) │
└─────────────────┘     └──────────────────┘
```

## Technology Stack

| Component | Choice | Why |
|-----------|--------|-----|
| LeRobot | Core library | Dataset format, policy implementations, training/eval scripts |
| PyTorch | ML framework | Tensor operations, GPU acceleration, autograd |
| gym-aloha | Simulation | MuJoCo-based ALOHA dual-arm environment |
| MuJoCo | Physics engine | Rigid body dynamics, contact simulation |
| Weights & Biases | Experiment tracking | Real-time training curves, run comparison |
| HuggingFace Hub | Model/dataset hosting | Push/pull models and datasets |
| pandas/matplotlib/seaborn | Analysis | Data profiling and visualization |

## The Task: AlohaTransferCube-v0

A dual-arm ALOHA robot must pick up a cube with the right arm and transfer it to the left arm. The dataset contains 50 human demonstrations, each 400 frames long at 50fps (8 seconds per episode). The robot has 14 degrees of freedom — 7 per arm (6 joints + 1 gripper).

This task was chosen because it's structurally similar to pick-and-place — the target use case for the SO-101 arm.

## Results Summary

| Policy | Steps | Batch Size | GPU | Success Rate |
|--------|-------|------------|-----|-------------|
| ACT (local) | 80k | 4 | GTX 1650 | 44% |
| ACT (Kaggle) | 40k | 16 | T4 | 58% |
| ACT (A40) | 100k | 32 | A40 | 52% |
| Diffusion (Kaggle) | 35k | 16 | T4 | 10% |

## How to Read This Course

Each chapter follows a consistent structure: the problem being solved, the design decisions made, the key concepts explained from first principles, the code walkthrough, how to verify it works, common gotchas, and how it connects to the bigger picture.

The course assumes you know Python and basic software engineering. Domain-specific concepts (imitation learning, action spaces, policy architectures) are explained from scratch.

## What's Next

[Chapter 1: The LeRobotDataset Format →](01_lerobot_dataset_format.md)