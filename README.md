# LeRobot Sim Manipulation Mastery

End-to-end fluency with the [LeRobot](https://github.com/huggingface/lerobot) ecosystem — datasets, training, evaluation, and custom tooling — using ALOHA simulation manipulation tasks.

Preparation for real-world pick-and-place with the SO-101 arm.

## Status

🚧 **In progress** — Phase 0 (Environment Setup)

## What This Project Covers

- Deep inspection and analysis of LeRobotDataset format (Parquet + MP4 video)
- Training ACT (Action Chunking Transformer) policy on ALOHA sim cube transfer task
- Training Diffusion Policy on the same task
- Structured comparison: success rates, action smoothness, training curves
- Reusable dataset analysis toolkit for any LeRobotDataset
- Detailed multi-chapter course documenting everything

## Quick Start
```bash
conda create -y -n lerobot python=3.12
conda activate lerobot
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install "lerobot[aloha] @ git+https://github.com/huggingface/lerobot.git"
pip install wandb seaborn pytest
```

## Project Structure
```
lerobot-sim-mastery/
├── src/                  # Dataset inspection, analysis, comparison tools
├── notebooks/            # Training and evaluation notebooks
├── tests/                # Unit and integration tests
├── outputs/              # Generated analysis, eval results, reports
├── course/               # Multi-chapter course
├── config/               # Environment config templates
├── .gitignore
├── requirements.txt
└── pyproject.toml
```

## Course

*(Coming after project completion)*
