# LeRobot Sim Manipulation Mastery

End-to-end fluency with the [LeRobot](https://github.com/huggingface/lerobot) ecosystem — datasets, training, evaluation, and custom tooling — using ALOHA simulation manipulation tasks.

Preparation for real-world pick-and-place with the SO-101 arm.

## Results

| Policy | Steps | Batch Size | GPU | Success Rate |
|--------|-------|------------|-----|-------------|
| ACT (local) | 80,000 | 4 | GTX 1650 | 44% |
| ACT (Kaggle) | 40,000 | 16 | T4 | **58%** |
| Diffusion Policy (Kaggle) | 35,000 | 16 | T4 | 10% |
| Reference ACT (A100) | 80,000 | 8 | A100 | ~90% |

**Key finding:** Batch size matters more than training steps. ACT with bs16 at 40k steps outperformed ACT with bs4 at 80k steps by 14 percentage points. Diffusion Policy needs 100k+ steps to converge — at 35k steps it was severely undertrained.

See [outputs/comparison/comparison_report.md](outputs/comparison/comparison_report.md) for detailed analysis.

## Trained Models on HuggingFace Hub

- [adimunot/act_aloha_transfer_cube](https://huggingface.co/adimunot/act_aloha_transfer_cube) — ACT, local GTX 1650, 80k steps
- [adimunot/act_aloha_transfer_cube_kaggle](https://huggingface.co/adimunot/act_aloha_transfer_cube_kaggle) — ACT, Kaggle T4, 40k steps
- [adimunot/diffusion_aloha_transfer_cube](https://huggingface.co/adimunot/diffusion_aloha_transfer_cube) — Diffusion Policy, Kaggle T4, 35k steps

## What This Project Covers

- **Dataset inspection & analysis:** Reusable tooling to profile any LeRobotDataset — field inspection, value ranges, action smoothness, joint correlations, outlier detection
- **ACT policy training:** Full pipeline from Hub dataset → training → checkpointing → evaluation in simulation
- **Diffusion Policy training:** Same pipeline, different architecture, comparison under identical conditions
- **Policy comparison:** Structured report with success rates, reward analysis, inference speed, and practical recommendations

## Architecture

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

## Project Structure

```
lerobot-sim-mastery/
├── src/
│   ├── inspect_dataset.py       # Dataset profiler — fields, ranges, sample frames
│   ├── analyze_dataset.py       # Analysis toolkit — histograms, smoothness, correlations
│   ├── compare_policies.py      # Policy comparison report generator
│   └── utils.py
├── tests/
│   ├── test_inspect_dataset.py
│   ├── test_analyze_dataset.py
│   └── test_compare_policies.py
├── outputs/
│   ├── inspection/              # Dataset profiles and sample frames
│   ├── analysis/                # Histograms, trajectories, correlation heatmaps
│   ├── eval_act/                # ACT local evaluation (44% success)
│   ├── eval_act_kaggle/         # ACT Kaggle evaluation (58% success)
│   ├── eval_diffusion/          # Diffusion evaluation (10% success)
│   └── comparison/              # Comparison report and plots
├── course/                      # Multi-chapter course (see below)
├── notebooks/
├── config/
├── .gitignore
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Quick Start

```bash
# Clone
git clone git@github.com:adimunot21/lerobot-sim-mastery.git
cd lerobot-sim-mastery

# Environment
conda create -y -n lerobot python=3.12
conda activate lerobot
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install "lerobot[aloha] @ git+https://github.com/huggingface/lerobot.git"
pip install wandb seaborn pytest

# Run dataset inspection
python src/inspect_dataset.py

# Run dataset analysis
python src/analyze_dataset.py

# Run policy comparison (requires eval outputs)
python src/compare_policies.py

# Run tests
pytest tests/ -v
```

## Evaluate a Pretrained Model

```bash
# Download and evaluate the best ACT model
lerobot-eval \
    --policy.path=adimunot/act_aloha_transfer_cube_kaggle \
    --env.type=aloha \
    --env.task=AlohaTransferCube-v0 \
    --eval.n_episodes=50 \
    --eval.batch_size=1 \
    --output_dir=outputs/eval_act_kaggle
```

## Dataset

Uses [lerobot/aloha_sim_transfer_cube_human](https://huggingface.co/datasets/lerobot/aloha_sim_transfer_cube_human): 50 episodes of a dual-arm ALOHA robot transferring a cube from right arm to left arm. 400 frames per episode at 50fps. 14-DOF action/state space (7 joints per arm including gripper).

## Course

| Chapter | Topic |
|---------|-------|
| [00](course/00_introduction.md) | Introduction — project goals, system overview, tech stack |
| [01](course/01_lerobot_dataset_format.md) | The LeRobotDataset Format — Parquet, MP4, features |
| [02](course/02_data_analysis.md) | Data Analysis for Robot Learning — smoothness, correlations, outliers |
| [03](course/03_act_policy.md) | ACT Policy — architecture, action chunking, CVAE |
| [04](course/04_training_on_colab.md) | Training Practicalities — GPU, batch size, VRAM, checkpointing |
| [05](course/05_evaluation_in_sim.md) | Evaluation in Simulation — rollouts, success metrics |
| [06](course/06_diffusion_policy.md) | Diffusion Policy — denoising, U-Net, convergence |
| [07](course/07_policy_comparison.md) | Policy Comparison — methodology, results, recommendations |
| [08](course/08_sim_to_real.md) | From Sim to Real — what changes with the SO-101 |

## Hardware Used

- **Local training:** Lenovo Legion Y540 — i7-9750H, 32GB RAM, GTX 1650 4GB VRAM, Ubuntu 24.04
- **Cloud training:** Kaggle T4 (16GB VRAM) — free tier
- **Dataset work & analysis:** All local, CPU-only

## License

MIT