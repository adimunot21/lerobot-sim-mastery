"""
Dataset Analysis Toolkit for LeRobotDataset format.

Goes beyond basic inspection to produce training-relevant analysis:
  - Per-joint action/state histograms (are distributions well-behaved?)
  - State trajectories over time (do episodes look consistent?)
  - Action smoothness / jerkiness metrics (noisy demos = bad policies)
  - Joint correlation analysis (which joints move together?)
  - Outlier episode detection (which demos should be discarded?)

Designed to be reusable on ANY LeRobotDataset — when your SO-101 arrives,
run this on your teleop recordings to catch bad data before training.

Usage:
    python src/analyze_dataset.py
    python src/analyze_dataset.py --repo-id lerobot/aloha_sim_insertion_human
    python src/analyze_dataset.py --repo-id lerobot/aloha_sim_transfer_cube_human --episodes 0 5 10
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_episode_boundaries(dataset: LeRobotDataset) -> list[dict[str, int]]:
    """Compute start/end frame indices for each episode from hf_dataset."""
    ep_indices = dataset.hf_dataset["episode_index"]
    boundaries = []
    current_ep = ep_indices[0]
    start = 0
    for i in range(1, len(ep_indices)):
        if ep_indices[i] != current_ep:
            boundaries.append({
                "episode_index": int(current_ep),
                "from": start,
                "to": i,
                "length": i - start,
            })
            current_ep = ep_indices[i]
            start = i
    boundaries.append({
        "episode_index": int(current_ep),
        "from": start,
        "to": len(ep_indices),
        "length": len(ep_indices) - start,
    })
    return boundaries


def get_joint_names(dataset: LeRobotDataset, feature_key: str) -> list[str]:
    """Extract joint names from dataset features, with fallback to dim_N."""
    feat = dataset.features.get(feature_key, {})
    if isinstance(feat, dict):
        names_field = feat.get("names", {})
        if isinstance(names_field, dict) and "motors" in names_field:
            return names_field["motors"]
        elif isinstance(names_field, list):
            return names_field
    # Fallback: get dimension from shape and generate generic names
    shape = feat.get("shape", [0]) if isinstance(feat, dict) else [0]
    n_dims = shape[0] if shape else 0
    return [f"dim_{d}" for d in range(n_dims)]


def load_episode_data(
    dataset: LeRobotDataset,
    ep_boundary: dict[str, int],
    keys: list[str],
) -> dict[str, torch.Tensor]:
    """Load all frames for a single episode, stacking tensors per key.

    Returns dict mapping each key to a [T, D] tensor where T is the number
    of frames in the episode.  Only loads tensor-valued keys with dim >= 1.

    This is how you'd debug a single episode — pull all its data and plot it.
    For real-world data, this is where you'd spot teleop glitches, dropped
    frames, or accidental bumps.
    """
    frames = {k: [] for k in keys}
    for i in range(ep_boundary["from"], ep_boundary["to"]):
        sample = dataset[i]
        for k in keys:
            if k in sample and isinstance(sample[k], torch.Tensor) and sample[k].dim() >= 1:
                frames[k].append(sample[k])

    return {k: torch.stack(v) if v else torch.tensor([]) for k, v in frames.items()}


# ---------------------------------------------------------------------------
# Analysis Functions
# ---------------------------------------------------------------------------

def analyze_action_histograms(
    dataset: LeRobotDataset,
    output_dir: Path,
    sample_every_n: int = 5,
) -> None:
    """Plot per-joint histograms for actions and states.

    WHY THIS MATTERS: If a joint's action distribution is heavily skewed,
    multimodal, or has sharp peaks at the extremes, it signals potential
    issues.  Clipped distributions mean the demos are hitting joint limits.
    Multimodal distributions could indicate the task has distinct phases
    (grasp vs release) — which is fine, but important to know because
    some policies (e.g. vanilla BC) struggle with multimodality while
    others (Diffusion Policy) handle it well.
    """
    logger.info("=" * 60)
    logger.info("ACTION & STATE HISTOGRAMS")
    logger.info("=" * 60)

    states, actions = [], []
    for i in range(0, len(dataset), sample_every_n):
        sample = dataset[i]
        if "observation.state" in sample:
            states.append(sample["observation.state"])
        if "action" in sample:
            actions.append(sample["action"])

    state_names = get_joint_names(dataset, "observation.state")
    action_names = get_joint_names(dataset, "action")

    for data, names, label in [
        (states, state_names, "state"),
        (actions, action_names, "action"),
    ]:
        if not data:
            continue

        stacked = torch.stack(data).numpy()  # [N, D]
        n_dims = stacked.shape[1]

        # Grid of histograms — one per joint
        n_cols = 4
        n_rows = (n_dims + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
        axes = axes.flatten()

        for d in range(n_dims):
            ax = axes[d]
            ax.hist(stacked[:, d], bins=50, color="steelblue", alpha=0.7, edgecolor="white")
            ax.set_title(names[d] if d < len(names) else f"dim_{d}", fontsize=10)
            ax.set_ylabel("Count")
            ax.axvline(x=stacked[:, d].mean(), color="red", linestyle="--", linewidth=1)

        # Hide unused subplots
        for d in range(n_dims, len(axes)):
            axes[d].set_visible(False)

        fig.suptitle(f"{label.title()} Histograms (red = mean)", fontsize=14)
        fig.tight_layout()
        path = output_dir / f"{label}_histograms.png"
        fig.savefig(path, dpi=100)
        plt.close(fig)
        logger.info(f"  Saved {label} histograms → {path}")


def analyze_episode_trajectories(
    dataset: LeRobotDataset,
    output_dir: Path,
    episode_indices: list[int] | None = None,
) -> None:
    """Plot state and action trajectories over time for selected episodes.

    WHY THIS MATTERS: This is the most intuitive way to "see" what a
    demonstration looks like.  Each line is a joint's position over time.
    Smooth, consistent trajectories across episodes = good data.  Erratic
    jumps, flat segments (robot stuck), or wildly different patterns between
    episodes = problems.

    For real-world teleop data, this is where you catch: operator mistakes,
    dropped frames (sudden jumps), mechanical issues (oscillations), and
    episodes where the task wasn't actually completed.
    """
    logger.info("=" * 60)
    logger.info("EPISODE TRAJECTORIES")
    logger.info("=" * 60)

    boundaries = get_episode_boundaries(dataset)

    if episode_indices is None:
        # Pick first, middle, and last episode for variety
        n_ep = len(boundaries)
        episode_indices = [0, n_ep // 2, n_ep - 1]

    state_names = get_joint_names(dataset, "observation.state")
    action_names = get_joint_names(dataset, "action")

    for ep_idx in episode_indices:
        if ep_idx >= len(boundaries):
            logger.warning(f"  Episode {ep_idx} out of range — skipping")
            continue

        b = boundaries[ep_idx]
        ep_data = load_episode_data(dataset, b, ["observation.state", "action"])

        for key, names, label in [
            ("observation.state", state_names, "state"),
            ("action", action_names, "action"),
        ]:
            tensor = ep_data.get(key)
            if tensor is None or tensor.numel() == 0:
                continue

            data_np = tensor.numpy()  # [T, D]
            n_dims = data_np.shape[1]
            time_axis = np.arange(data_np.shape[0]) / dataset.fps  # seconds

            # Split into left arm and right arm for clarity
            mid = n_dims // 2
            for side, start_d, end_d in [("left_arm", 0, mid), ("right_arm", mid, n_dims)]:
                fig, ax = plt.subplots(figsize=(12, 5))
                for d in range(start_d, end_d):
                    name = names[d] if d < len(names) else f"dim_{d}"
                    # Strip the side prefix for cleaner legend
                    short_name = name.replace("left_", "").replace("right_", "")
                    ax.plot(time_axis, data_np[:, d], label=short_name, linewidth=1.2)

                ax.set_xlabel("Time (seconds)")
                ax.set_ylabel("Joint Position (radians)")
                ax.set_title(f"Episode {ep_idx} — {label.title()} — {side.replace('_', ' ').title()}")
                ax.legend(loc="upper right", fontsize=8)
                ax.grid(True, alpha=0.3)
                fig.tight_layout()

                path = output_dir / f"ep{ep_idx:03d}_{label}_{side}_trajectory.png"
                fig.savefig(path, dpi=100)
                plt.close(fig)

        logger.info(f"  Saved trajectories for episode {ep_idx}")


def analyze_action_smoothness(
    dataset: LeRobotDataset,
    output_dir: Path,
    sample_every_n: int = 1,
) -> dict[str, Any]:
    """Compute action smoothness metrics per episode.

    WHY THIS MATTERS: Jerky demonstrations produce jerky policies.  Action
    smoothness is measured as the L2 norm of consecutive action differences
    (i.e., the "velocity" of actions).  High smoothness = small deltas
    between consecutive frames = smooth motion.  Large spikes indicate
    sudden movements or teleop glitches.

    We compute per-episode metrics:
      - Mean action delta (lower = smoother)
      - Max action delta (spikes = potential problems)
      - Std of action deltas (consistency of motion speed)

    For the ALOHA sim data, we expect very smooth actions since these are
    scripted demonstrations.  For real teleop data, you'll typically see
    higher values and more variance — that's normal, but outliers should
    be investigated.
    """
    logger.info("=" * 60)
    logger.info("ACTION SMOOTHNESS ANALYSIS")
    logger.info("=" * 60)

    boundaries = get_episode_boundaries(dataset)
    action_names = get_joint_names(dataset, "action")

    episode_smoothness = []

    for b in boundaries:
        ep_data = load_episode_data(dataset, b, ["action"])
        actions = ep_data.get("action")
        if actions is None or actions.shape[0] < 2:
            continue

        # Action deltas: difference between consecutive frames
        # Shape: [T-1, D] — each row is how much each joint moved
        deltas = actions[1:] - actions[:-1]

        # L2 norm per timestep — overall "speed" of the action change
        l2_norms = torch.norm(deltas, dim=1)  # [T-1]

        # Per-joint absolute deltas
        per_joint_mean_delta = deltas.abs().mean(dim=0)  # [D]

        ep_metrics = {
            "episode_index": b["episode_index"],
            "mean_delta_l2": l2_norms.mean().item(),
            "max_delta_l2": l2_norms.max().item(),
            "std_delta_l2": l2_norms.std().item(),
            "per_joint_mean_delta": {
                name: per_joint_mean_delta[d].item()
                for d, name in enumerate(action_names)
            },
        }
        episode_smoothness.append(ep_metrics)

    # Summary across all episodes
    mean_deltas = [e["mean_delta_l2"] for e in episode_smoothness]
    max_deltas = [e["max_delta_l2"] for e in episode_smoothness]

    summary = {
        "overall_mean_delta": float(np.mean(mean_deltas)),
        "overall_std_delta": float(np.std(mean_deltas)),
        "smoothest_episode": int(np.argmin(mean_deltas)),
        "jerkiest_episode": int(np.argmax(mean_deltas)),
        "episodes": episode_smoothness,
    }

    logger.info(f"  Overall mean action delta (L2): {summary['overall_mean_delta']:.6f}")
    logger.info(f"  Overall std of episode deltas:  {summary['overall_std_delta']:.6f}")
    logger.info(f"  Smoothest episode: {summary['smoothest_episode']}")
    logger.info(f"  Jerkiest episode:  {summary['jerkiest_episode']}")

    # Plot 1: Mean delta per episode (bar chart)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    ax = axes[0]
    ep_ids = [e["episode_index"] for e in episode_smoothness]
    ax.bar(ep_ids, mean_deltas, color="steelblue", alpha=0.8)
    ax.axhline(y=summary["overall_mean_delta"], color="red", linestyle="--",
               label=f"Mean: {summary['overall_mean_delta']:.4f}")
    ax.set_xlabel("Episode Index")
    ax.set_ylabel("Mean Action Delta (L2)")
    ax.set_title("Action Smoothness per Episode (lower = smoother)")
    ax.legend()

    # Plot 2: Max delta per episode (flags jerkiest moments)
    ax = axes[1]
    ax.bar(ep_ids, max_deltas, color="coral", alpha=0.8)
    ax.set_xlabel("Episode Index")
    ax.set_ylabel("Max Action Delta (L2)")
    ax.set_title("Worst Single-Step Jerk per Episode")

    fig.tight_layout()
    path = output_dir / "action_smoothness.png"
    fig.savefig(path, dpi=100)
    plt.close(fig)
    logger.info(f"  Saved smoothness plots → {path}")

    # Save metrics
    metrics_path = output_dir / "action_smoothness.json"
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    logger.info(f"  Saved smoothness metrics → {metrics_path}")

    return summary


def analyze_joint_correlations(
    dataset: LeRobotDataset,
    output_dir: Path,
    sample_every_n: int = 5,
) -> None:
    """Compute and plot correlation matrix between joint dimensions.

    WHY THIS MATTERS: Highly correlated joints often move together as part
    of a coordinated motion (e.g., shoulder + elbow for reaching).  This
    tells you about the effective dimensionality of the task — if most
    joints are correlated, the task might be simpler than the raw 14-dim
    action space suggests.

    Negative correlations are also interesting — they indicate opposing
    motions (e.g., one gripper opens while the other closes during handoff).

    For policy architecture decisions: if the action space has strong
    structure (correlations), simpler models might suffice.  If it's mostly
    uncorrelated, you may need higher model capacity.
    """
    logger.info("=" * 60)
    logger.info("JOINT CORRELATION ANALYSIS")
    logger.info("=" * 60)

    action_names = get_joint_names(dataset, "action")

    actions = []
    for i in range(0, len(dataset), sample_every_n):
        sample = dataset[i]
        if "action" in sample:
            actions.append(sample["action"])

    if not actions:
        logger.info("  No action data found — skipping")
        return

    stacked = torch.stack(actions).numpy()  # [N, D]

    # Compute correlation matrix
    corr_matrix = np.corrcoef(stacked.T)  # [D, D]

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        xticklabels=action_names,
        yticklabels=action_names,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        ax=ax,
    )
    ax.set_title("Action Joint Correlation Matrix")
    fig.tight_layout()
    path = output_dir / "action_correlation_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"  Saved correlation heatmap → {path}")

    # Log strongest correlations (excluding diagonal)
    n_dims = corr_matrix.shape[0]
    pairs = []
    for i in range(n_dims):
        for j in range(i + 1, n_dims):
            pairs.append((action_names[i], action_names[j], corr_matrix[i, j]))

    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    logger.info("\n  Top 5 strongest correlations:")
    for name_a, name_b, corr in pairs[:5]:
        logger.info(f"    {name_a} ↔ {name_b}: {corr:+.3f}")

    logger.info("\n  Top 5 strongest negative correlations:")
    neg_pairs = [p for p in pairs if p[2] < 0]
    for name_a, name_b, corr in neg_pairs[:5]:
        logger.info(f"    {name_a} ↔ {name_b}: {corr:+.3f}")


def detect_outlier_episodes(
    dataset: LeRobotDataset,
    output_dir: Path,
) -> dict[str, Any]:
    """Flag episodes that are statistical outliers.

    WHY THIS MATTERS: In real-world data collection, some demonstrations
    will be bad — the operator fumbled, the object slipped, the gripper
    didn't close properly.  Training on these hurts policy performance.
    This function flags episodes whose action statistics deviate
    significantly from the population, so you can review and potentially
    discard them.

    We flag on three criteria:
      1. Episode length (if variable — not applicable to fixed-length sim data)
      2. Mean action magnitude (episode where the robot barely moved)
      3. Action smoothness (unusually jerky episode)

    An episode is flagged if any metric is >2 std from the mean.
    """
    logger.info("=" * 60)
    logger.info("OUTLIER EPISODE DETECTION")
    logger.info("=" * 60)

    boundaries = get_episode_boundaries(dataset)

    ep_stats = []
    for b in boundaries:
        ep_data = load_episode_data(dataset, b, ["action", "observation.state"])
        actions = ep_data.get("action")
        states = ep_data.get("observation.state")

        if actions is None or actions.numel() == 0:
            continue

        # Mean absolute action magnitude
        mean_action_mag = actions.abs().mean().item()

        # Action smoothness (mean L2 delta)
        if actions.shape[0] >= 2:
            deltas = actions[1:] - actions[:-1]
            mean_delta = torch.norm(deltas, dim=1).mean().item()
        else:
            mean_delta = 0.0

        # State range (total joint travel)
        if states is not None and states.numel() > 0:
            state_range = (states.max(dim=0).values - states.min(dim=0).values).sum().item()
        else:
            state_range = 0.0

        ep_stats.append({
            "episode_index": b["episode_index"],
            "length": b["length"],
            "mean_action_mag": mean_action_mag,
            "mean_delta": mean_delta,
            "state_range": state_range,
        })

    # Compute z-scores for each metric
    metrics = ["length", "mean_action_mag", "mean_delta", "state_range"]
    outliers = []

    for metric in metrics:
        values = np.array([e[metric] for e in ep_stats])
        mean_val = values.mean()
        std_val = values.std()

        if std_val < 1e-8:
            # No variance — all episodes identical on this metric (common in sim)
            logger.info(f"  {metric}: no variance (all episodes identical) — no outliers")
            continue

        for ep in ep_stats:
            z_score = (ep[metric] - mean_val) / std_val
            if abs(z_score) > 2.0:
                outliers.append({
                    "episode_index": ep["episode_index"],
                    "metric": metric,
                    "value": ep[metric],
                    "z_score": z_score,
                    "population_mean": float(mean_val),
                    "population_std": float(std_val),
                })

    result = {
        "total_episodes": len(ep_stats),
        "num_outliers": len(outliers),
        "outlier_episodes": outliers,
        "episode_stats": ep_stats,
    }

    if outliers:
        logger.info(f"\n  Found {len(outliers)} outlier flags:")
        for o in outliers:
            logger.info(
                f"    Episode {o['episode_index']}: {o['metric']} = {o['value']:.4f} "
                f"(z={o['z_score']:+.2f}, pop_mean={o['population_mean']:.4f})"
            )
    else:
        logger.info("  No outlier episodes detected — data looks clean.")

    # Plot episode stats overview
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    for ax, metric, title in zip(
        axes.flatten(),
        metrics,
        ["Episode Length", "Mean Action Magnitude", "Mean Action Delta (Smoothness)", "Total State Range"],
    ):
        values = [e[metric] for e in ep_stats]
        ep_ids = [e["episode_index"] for e in ep_stats]
        ax.bar(ep_ids, values, color="steelblue", alpha=0.7)
        mean_val = np.mean(values)
        std_val = np.std(values)
        ax.axhline(y=mean_val, color="red", linestyle="--", linewidth=1)
        if std_val > 1e-8:
            ax.axhline(y=mean_val + 2 * std_val, color="orange", linestyle=":", linewidth=1)
            ax.axhline(y=mean_val - 2 * std_val, color="orange", linestyle=":", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("Episode")

    fig.suptitle("Episode Statistics Overview (red=mean, orange=±2σ)", fontsize=13)
    fig.tight_layout()
    path = output_dir / "outlier_detection.png"
    fig.savefig(path, dpi=100)
    plt.close(fig)
    logger.info(f"  Saved outlier detection plot → {path}")

    # Save results
    results_path = output_dir / "outlier_detection.json"
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    logger.info(f"  Saved outlier results → {results_path}")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze a LeRobotDataset — histograms, trajectories, smoothness, correlations, outliers."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="lerobot/aloha_sim_transfer_cube_human",
        help="HuggingFace Hub repo ID for the dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/analysis",
        help="Directory to save analysis outputs.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="*",
        default=None,
        help="Specific episode indices to plot trajectories for. Default: first, middle, last.",
    )
    parser.add_argument(
        "--sample-every-n",
        type=int,
        default=5,
        help="For histogram/correlation analysis, sample every Nth frame.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading dataset: {args.repo_id}")
    dataset = LeRobotDataset(args.repo_id)
    logger.info(f"Dataset loaded: {len(dataset)} frames, {dataset.num_episodes} episodes")

    # Run all analyses
    analyze_action_histograms(dataset, output_dir, sample_every_n=args.sample_every_n)
    analyze_episode_trajectories(dataset, output_dir, episode_indices=args.episodes)
    analyze_action_smoothness(dataset, output_dir)
    analyze_joint_correlations(dataset, output_dir, sample_every_n=args.sample_every_n)
    detect_outlier_episodes(dataset, output_dir)

    logger.info("=" * 60)
    logger.info("ANALYSIS COMPLETE")
    logger.info(f"All outputs saved to: {output_dir}/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()