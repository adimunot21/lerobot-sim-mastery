"""
Policy Comparison Report Generator.

Reads evaluation results from multiple policy runs and generates a structured
comparison report with visualizations.  Designed to answer: "Which policy
should I use for my SO-101, and why?"

Usage:
    python src/compare_policies.py
"""

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration — define all eval runs to compare
# ---------------------------------------------------------------------------

EVAL_RUNS = [
    {
        "name": "ACT (local, bs4, 80k)",
        "policy_type": "ACT",
        "eval_dir": "outputs/eval_act",
        "steps": 80_000,
        "batch_size": 4,
        "gpu": "GTX 1650",
        "notes": "Trained locally, limited by 4GB VRAM",
    },
    {
        "name": "ACT (Kaggle, bs16, 40k)",
        "policy_type": "ACT",
        "eval_dir": "outputs/eval_act_kaggle",
        "steps": 40_000,
        "batch_size": 16,
        "gpu": "T4",
        "notes": "Better batch size, fewer steps, still outperforms local",
    },
    {
        "name": "ACT (A40, bs32, 100k)",
        "policy_type": "ACT",
        "eval_dir": "outputs/eval_act_a40",
        "steps": 100_000,
        "batch_size": 32,
        "gpu": "A40",
        "notes": "RunPod A40, highest compute budget",
    },
    {
        "name": "Diffusion (Kaggle, bs16, 35k)",
        "policy_type": "Diffusion",
        "eval_dir": "outputs/eval_diffusion",
        "steps": 35_000,
        "batch_size": 16,
        "gpu": "T4",
        "notes": "Undertrained — Diffusion needs 100k+ steps to converge",
    },
]

OUTPUT_DIR = Path("outputs/comparison")


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_eval_results(eval_dir: str) -> dict | None:
    """Load evaluation metrics from an eval output directory.

    lerobot-eval saves results in eval_info.json with this structure:
    {
        "per_task": [
            {
                "task_group": "aloha",
                "task_id": 0,
                "metrics": {
                    "sum_rewards": [...],
                    "max_rewards": [...],
                    "successes": [...],
                    "video_paths": [...]
                }
            }
        ],
        "per_group": { "aloha": { "avg_sum_reward": ..., "pc_success": ... } },
        "overall": { "avg_sum_reward": ..., "pc_success": ..., "eval_s": ..., "eval_ep_s": ... }
    }
    """
    eval_path = Path(eval_dir)
    eval_json = eval_path / "eval_info.json"

    if not eval_json.exists():
        logger.warning(f"  eval_info.json not found in {eval_dir}")
        return None

    try:
        data = json.loads(eval_json.read_text())
        return data
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.warning(f"  Failed to parse {eval_json}: {e}")
        return None


def extract_metrics(eval_data: dict) -> dict:
    """Extract standardised metrics from eval_info.json.

    The actual structure has 'per_task', 'per_group', and 'overall' keys.
    We pull success rate and rewards from 'overall', and per-episode
    successes from 'per_task'.
    """
    metrics = {
        "pc_success": None,
        "avg_sum_reward": None,
        "avg_max_reward": None,
        "n_episodes": None,
        "per_episode_successes": None,
        "eval_time_s": None,
        "eval_ep_s": None,
    }

    # Try 'overall' key first (top-level aggregated metrics)
    overall = eval_data.get("overall", {})
    if overall:
        metrics["pc_success"] = overall.get("pc_success")
        metrics["avg_sum_reward"] = overall.get("avg_sum_reward")
        metrics["avg_max_reward"] = overall.get("avg_max_reward")
        metrics["n_episodes"] = overall.get("n_episodes")
        metrics["eval_time_s"] = overall.get("eval_s")
        metrics["eval_ep_s"] = overall.get("eval_ep_s")

    # Try 'per_group' if 'overall' didn't work
    if metrics["pc_success"] is None:
        per_group = eval_data.get("per_group", {})
        for group_data in per_group.values():
            if isinstance(group_data, dict) and "pc_success" in group_data:
                metrics["pc_success"] = group_data.get("pc_success")
                metrics["avg_sum_reward"] = group_data.get("avg_sum_reward")
                metrics["avg_max_reward"] = group_data.get("avg_max_reward")
                metrics["n_episodes"] = group_data.get("n_episodes")
                break

    # Get per-episode successes from 'per_task'
    per_task = eval_data.get("per_task", [])
    if per_task and isinstance(per_task, list):
        task_metrics = per_task[0].get("metrics", {})
        metrics["per_episode_successes"] = task_metrics.get("successes")
        if metrics["n_episodes"] is None and metrics["per_episode_successes"]:
            metrics["n_episodes"] = len(metrics["per_episode_successes"])

    return metrics


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def plot_success_rate_comparison(runs: list[dict], output_dir: Path) -> None:
    """Bar chart comparing success rates across all runs."""
    names = [r["name"] for r in runs]
    rates = [r["metrics"]["pc_success"] or 0 for r in runs]
    colors = ["#2196F3" if r["policy_type"] == "ACT" else "#FF9800" for r in runs]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(names)), rates, color=colors, alpha=0.85, edgecolor="white")

    # Add value labels on bars
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{rate:.0f}%", ha="center", va="bottom", fontweight="bold", fontsize=12)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Policy Comparison — Success Rate on AlohaTransferCube-v0")
    ax.set_ylim(0, 100)
    ax.axhline(y=90, color="green", linestyle="--", alpha=0.5, label="Reference ACT (A100, bs8, 80k)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = output_dir / "success_rate_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"  Saved success rate comparison → {path}")


def plot_reward_comparison(runs: list[dict], output_dir: Path) -> None:
    """Bar chart comparing average rewards."""
    names = [r["name"] for r in runs]
    rewards = [r["metrics"]["avg_sum_reward"] or 0 for r in runs]
    colors = ["#2196F3" if r["policy_type"] == "ACT" else "#FF9800" for r in runs]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(names)), rewards, color=colors, alpha=0.85, edgecolor="white")

    for bar, reward in zip(bars, rewards):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{reward:.0f}", ha="center", va="bottom", fontweight="bold", fontsize=12)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Average Sum Reward")
    ax.set_title("Policy Comparison — Average Episode Reward")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = output_dir / "reward_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"  Saved reward comparison → {path}")


def plot_per_episode_success(runs: list[dict], output_dir: Path) -> None:
    """Heatmap-style plot showing success/failure per episode per run."""
    # Only include runs that have per-episode data
    valid_runs = [r for r in runs if r["metrics"]["per_episode_successes"] is not None]
    if not valid_runs:
        logger.info("  No per-episode success data available — skipping heatmap")
        return

    n_runs = len(valid_runs)
    max_episodes = max(len(r["metrics"]["per_episode_successes"]) for r in valid_runs)

    fig, ax = plt.subplots(figsize=(14, 2 + n_runs * 0.8))

    for i, run in enumerate(valid_runs):
        successes = run["metrics"]["per_episode_successes"]
        for j, success in enumerate(successes):
            color = "#4CAF50" if success else "#F44336"
            ax.add_patch(plt.Rectangle((j, i), 0.9, 0.8, color=color, alpha=0.7))

    ax.set_xlim(0, max_episodes)
    ax.set_ylim(0, n_runs)
    ax.set_yticks([i + 0.4 for i in range(n_runs)])
    ax.set_yticklabels([r["name"] for r in valid_runs], fontsize=10)
    ax.set_xlabel("Episode Index")
    ax.set_title("Per-Episode Success (green = success, red = failure)")

    fig.tight_layout()
    path = output_dir / "per_episode_success.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"  Saved per-episode success heatmap → {path}")


def plot_training_efficiency(runs: list[dict], output_dir: Path) -> None:
    """Scatter plot: training steps vs success rate, colored by policy type."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for run in runs:
        color = "#2196F3" if run["policy_type"] == "ACT" else "#FF9800"
        marker = "o" if run["policy_type"] == "ACT" else "s"
        success = run["metrics"]["pc_success"] or 0
        ax.scatter(run["steps"], success, color=color, marker=marker,
                   s=200, edgecolors="black", linewidth=1.5, zorder=5)
        ax.annotate(run["name"], (run["steps"], success),
                    textcoords="offset points", xytext=(10, 10), fontsize=8)

    # Reference point
    ax.scatter(80000, 90, color="green", marker="*", s=300, edgecolors="black",
               linewidth=1.5, zorder=5, label="Reference ACT (A100)")

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Training Efficiency — Steps vs Success Rate")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = output_dir / "training_efficiency.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"  Saved training efficiency plot → {path}")


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------

def generate_report(runs: list[dict], output_dir: Path) -> None:
    """Generate a structured markdown comparison report."""

    report_lines = [
        "# Policy Comparison Report",
        "",
        "## Task: AlohaTransferCube-v0",
        "",
        "Pick up a cube with the right arm and transfer it to the left arm.",
        "50 human demonstration episodes, 400 frames each, 50fps.",
        "",
        "---",
        "",
        "## Results Summary",
        "",
        "| Run | Policy | Steps | Batch Size | GPU | Success Rate | Avg Reward |",
        "|-----|--------|-------|------------|-----|-------------|------------|",
    ]

    for run in runs:
        m = run["metrics"]
        success = f"{m['pc_success']:.0f}%" if m["pc_success"] is not None else "N/A"
        reward = f"{m['avg_sum_reward']:.1f}" if m["avg_sum_reward"] is not None else "N/A"
        report_lines.append(
            f"| {run['name']} | {run['policy_type']} | {run['steps']:,} | "
            f"{run['batch_size']} | {run['gpu']} | {success} | {reward} |"
        )

    report_lines.extend([
        "",
        "Reference: ACT trained on A100 with batch_size=8 for 80k steps achieves ~90% success.",
        "",
        "---",
        "",
        "## Key Findings",
        "",
        "### 1. Batch Size Matters More Than Training Steps",
        "",
        "ACT with batch_size=16 at 40k steps (58%) significantly outperformed ACT with "
        "batch_size=4 at 80k steps (44%). Doubling the batch size and halving the steps "
        "yielded a 14 percentage point improvement. Larger batches produce more stable "
        "gradient estimates, leading to better convergence even with fewer optimization steps.",
        "",
        "### 2. ACT Converges Faster Than Diffusion Policy",
        "",
        "ACT reached 58% success in 40k steps. Diffusion Policy reached only 10% in 35k steps. "
        "This is expected: ACT predicts actions in a single forward pass, while Diffusion Policy "
        "must learn to iteratively denoise random noise into coherent action sequences over 100 "
        "denoising steps. This fundamentally harder learning problem requires 100-200k+ steps to "
        "converge. With limited compute budget, ACT is the clear winner.",
        "",
        "### 3. Diffusion Policy's Strengths Don't Show at Low Training Budget",
        "",
        "Diffusion Policy's advantage is modeling multimodal action distributions — when there "
        "are multiple valid ways to perform a task, it can represent all of them rather than "
        "averaging. But this advantage only manifests when the model is fully trained. At 35k "
        "steps, the denoising process hasn't converged, producing noisy, incoherent actions.",
        "",
        "### 4. Inference Speed: ACT >> Diffusion",
        "",
    ])

    # Add inference speed comparison if we have eval timing
    act_kaggle = next((r for r in runs if "Kaggle" in r["name"] and r["policy_type"] == "ACT"), None)
    diffusion = next((r for r in runs if r["policy_type"] == "Diffusion"), None)

    if act_kaggle and diffusion:
        act_time = act_kaggle["metrics"].get("eval_ep_s")
        diff_time = diffusion["metrics"].get("eval_ep_s")
        if act_time and diff_time:
            report_lines.extend([
                f"ACT evaluation: ~{act_time:.0f} seconds per episode.",
                f"Diffusion evaluation: ~{diff_time:.0f} seconds per episode.",
                f"Diffusion is ~{diff_time/act_time:.0f}x slower at inference due to the iterative "
                "denoising process (100 forward passes per action prediction vs 1 for ACT).",
                "",
            ])

    report_lines.extend([
        "---",
        "",
        "## Recommendations for SO-101",
        "",
        "Based on these results, the recommended workflow when the SO-101 arrives:",
        "",
        "1. Start with ACT — it converges faster, is cheaper to train, and runs faster at inference. "
        "For a pick-and-place task with 50-100 demonstrations, ACT with batch_size=8-16 for "
        "50-80k steps should produce a working policy.",
        "",
        "2. Use batch_size as large as your GPU allows — the improvement from bs4→bs16 was "
        "larger than the improvement from 40k→80k steps. Prioritize batch size over step count.",
        "",
        "3. Only try Diffusion Policy if ACT fails on a multimodal task — if there are genuinely "
        "multiple valid strategies and ACT averages between them (producing invalid motions), "
        "Diffusion Policy is worth the extra training time. Budget 100k+ steps.",
        "",
        "4. Always evaluate multiple checkpoints — the best checkpoint may not be the last one. "
        "Our local ACT run showed 46% at 60k vs 44% at 80k.",
        "",
        "---",
        "",
        "## Visualizations",
        "",
        "![Success Rate Comparison](success_rate_comparison.png)",
        "",
        "![Reward Comparison](reward_comparison.png)",
        "",
        "![Training Efficiency](training_efficiency.png)",
        "",
        "![Per-Episode Success](per_episode_success.png)",
        "",
    ])

    report_path = output_dir / "comparison_report.md"
    report_path.write_text("\n".join(report_lines))
    logger.info(f"  Saved comparison report → {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("POLICY COMPARISON REPORT")
    logger.info("=" * 60)

    # Load results for each run
    valid_runs = []
    for run in EVAL_RUNS:
        logger.info(f"\n  Loading: {run['name']}")
        eval_data = load_eval_results(run["eval_dir"])
        if eval_data is not None:
            metrics = extract_metrics(eval_data)
            run["metrics"] = metrics
            valid_runs.append(run)
            logger.info(f"    Success rate: {metrics['pc_success']}%")
            logger.info(f"    Avg reward:   {metrics['avg_sum_reward']}")
        else:
            logger.warning(f"    No results found — skipping")

    if not valid_runs:
        logger.error("No valid eval results found. Run evaluations first.")
        return

    # Generate plots
    logger.info("\n  Generating visualizations...")
    plot_success_rate_comparison(valid_runs, OUTPUT_DIR)
    plot_reward_comparison(valid_runs, OUTPUT_DIR)
    plot_per_episode_success(valid_runs, OUTPUT_DIR)
    plot_training_efficiency(valid_runs, OUTPUT_DIR)

    # Generate report
    logger.info("\n  Generating report...")
    generate_report(valid_runs, OUTPUT_DIR)

    logger.info("=" * 60)
    logger.info("COMPARISON COMPLETE")
    logger.info(f"All outputs saved to: {OUTPUT_DIR}/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()