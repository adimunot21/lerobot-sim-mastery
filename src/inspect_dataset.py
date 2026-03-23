"""
Dataset Inspector for LeRobotDataset format.

Loads any LeRobotDataset from the HuggingFace Hub (or local cache), profiles every
field (shape, dtype, value range, statistics), decodes sample video frames, and
saves a complete inspection report.

Reusable for ANY LeRobotDataset — not just ALOHA.

Usage:
    python src/inspect_dataset.py                                    # default: aloha transfer cube
    python src/inspect_dataset.py --repo-id lerobot/aloha_sim_insertion_human
    python src/inspect_dataset.py --repo-id lerobot/pusht
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib
# Use non-interactive backend so plots save to file without needing a display.
# Relevant when running over SSH or in headless environments.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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

def tensor_stats(t: torch.Tensor) -> dict[str, Any]:
    """Compute summary statistics for a tensor, handling both float and int types."""
    t_float = t.float()
    stats = {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "min": t_float.min().item(),
        "max": t_float.max().item(),
        "mean": t_float.mean().item(),
        "std": t_float.std().item() if t.numel() > 1 else 0.0,
    }
    return stats


def save_image_grid(images: list[np.ndarray], path: Path, title: str = "") -> None:
    """Save a row of images as a single figure.

    Args:
        images: list of HWC uint8 numpy arrays (RGB).
        path: output file path.
        title: optional figure title.
    """
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, img in zip(axes, images):
        ax.imshow(img)
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved image grid → {path}")


def get_episode_boundaries(dataset: LeRobotDataset) -> list[dict[str, int]]:
    """Compute start/end frame indices for each episode.

    LeRobotDataset's underlying hf_dataset has an 'episode_index' column.
    We use that to find where each episode starts and ends in the flat
    frame array.  This is how the dataset maps global frame indices to
    episodes — understanding this mapping is essential for data analysis
    and debugging.

    Returns:
        List of dicts with keys 'episode_index', 'from', 'to', 'length'.
    """
    ep_indices = dataset.hf_dataset["episode_index"]
    boundaries = []
    current_ep = ep_indices[0]
    start = 0

    for i in range(1, len(ep_indices)):
        if ep_indices[i] != current_ep:
            boundaries.append({
                "episode_index": current_ep,
                "from": start,
                "to": i,
                "length": i - start,
            })
            current_ep = ep_indices[i]
            start = i

    # Last episode
    boundaries.append({
        "episode_index": current_ep,
        "from": start,
        "to": len(ep_indices),
        "length": len(ep_indices) - start,
    })

    return boundaries


# ---------------------------------------------------------------------------
# Core inspection
# ---------------------------------------------------------------------------

def inspect_metadata(dataset: LeRobotDataset, output_dir: Path) -> dict[str, Any]:
    """Extract and log dataset-level metadata.

    LeRobotDataset exposes metadata through:
      - dataset.meta: a LeRobotDatasetMetadata object with total_episodes,
        total_frames, etc.
      - dataset.features: dict mapping feature names to their dtype, shape,
        and (for motors) joint names.
      - dataset.fps: recording frame rate.
      - dataset.repo_id: HuggingFace Hub repository ID.

    The features dict is particularly important — it defines the data
    contract: what observations and actions look like, their dimensions,
    and for motor data, which dimension corresponds to which joint.
    """
    meta_obj = dataset.meta

    meta = {
        "repo_id": dataset.repo_id,
        "num_episodes": dataset.num_episodes,
        "num_frames": dataset.num_frames,
        "fps": dataset.fps,
        "features": {},
    }

    # Try to extract additional fields from meta object
    for attr in ["codebase_version", "robot_type", "total_tasks"]:
        try:
            meta[attr] = getattr(meta_obj, attr, "unknown")
        except Exception:
            meta[attr] = "unknown"

    logger.info("=" * 60)
    logger.info("DATASET METADATA")
    logger.info("=" * 60)
    logger.info(f"  repo_id        : {meta['repo_id']}")
    logger.info(f"  num_episodes   : {meta['num_episodes']}")
    logger.info(f"  num_frames     : {meta['num_frames']}")
    logger.info(f"  fps            : {meta['fps']}")
    logger.info(f"  codebase_ver   : {meta.get('codebase_version', 'unknown')}")
    logger.info(f"  robot_type     : {meta.get('robot_type', 'unknown')}")

    # Feature definitions
    logger.info("-" * 60)
    logger.info("FEATURES")
    logger.info("-" * 60)

    for feat_name, feat_info in dataset.features.items():
        feat_dict = {}
        if isinstance(feat_info, dict):
            feat_dict = {k: v for k, v in feat_info.items()}
        else:
            feat_dict = {"raw": str(feat_info)}

        meta["features"][feat_name] = feat_dict
        logger.info(f"  {feat_name}:")
        for k, v in feat_dict.items():
            logger.info(f"    {k}: {v}")

    # Save metadata to JSON
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    logger.info(f"\nSaved metadata → {meta_path}")

    return meta


def inspect_samples(
    dataset: LeRobotDataset,
    output_dir: Path,
    num_samples: int = 3,
) -> dict[str, Any]:
    """Load individual frames and profile every field.

    When you index into a LeRobotDataset (e.g. dataset[0]), you get a dict
    with keys like:
      - 'observation.state': [14] float32 tensor of joint positions
      - 'action': [14] float32 tensor of target joint positions
      - 'observation.images.top': [3, 480, 640] float32 tensor in [0, 1]
      - 'episode_index': scalar int64 tensor
      - 'frame_index': scalar int64 tensor
      - 'timestamp': scalar float32 tensor (seconds since episode start)
      - 'task': string describing the task
      - 'task_index': scalar int64 tensor
      - 'next.done': scalar bool tensor (True at last frame of episode)
      - 'index': scalar int64 tensor (global index in flat dataset)

    For image/video features, the tensor is shape [C, H, W] with float32
    values in [0, 1] (normalised during decoding).  For state/action
    features, the tensor is shape [D] with raw float32 joint positions
    in radians.
    """
    logger.info("=" * 60)
    logger.info(f"SAMPLE INSPECTION (first {num_samples} frames)")
    logger.info("=" * 60)

    all_stats: dict[str, list[dict]] = {}

    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        logger.info(f"\n--- Frame {i} ---")
        logger.info(f"  Keys: {sorted(sample.keys())}")

        for key, value in sorted(sample.items()):
            if isinstance(value, torch.Tensor):
                if value.numel() > 1 and value.dim() <= 1:
                    # Vector data (state, action) — show full stats
                    stats = tensor_stats(value)
                    all_stats.setdefault(key, []).append(stats)
                    logger.info(
                        f"  {key}: shape={stats['shape']} dtype={stats['dtype']} "
                        f"min={stats['min']:.4f} max={stats['max']:.4f} "
                        f"mean={stats['mean']:.4f} std={stats['std']:.4f}"
                    )
                elif value.dim() >= 2:
                    # Image data — just shape and range
                    stats = tensor_stats(value)
                    all_stats.setdefault(key, []).append(stats)
                    logger.info(
                        f"  {key}: shape={stats['shape']} dtype={stats['dtype']} "
                        f"min={stats['min']:.4f} max={stats['max']:.4f}"
                    )
                else:
                    # Scalar metadata tensor
                    logger.info(f"  {key}: {value.item()} (dtype={value.dtype})")
            elif isinstance(value, str):
                logger.info(f"  {key}: \"{value}\"")
            else:
                logger.info(f"  {key}: {value} (type={type(value).__name__})")

    # Save raw sample stats
    stats_path = output_dir / "sample_stats.json"
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2, default=str)
    logger.info(f"\nSaved sample stats → {stats_path}")

    return all_stats


def inspect_episode_structure(
    dataset: LeRobotDataset,
    output_dir: Path,
) -> dict[str, Any]:
    """Profile episode-level structure: lengths, index boundaries.

    We compute episode boundaries from the hf_dataset's 'episode_index'
    column.  Each episode is a contiguous block of frames recorded during
    a single demonstration.  Understanding episode structure is critical
    for debugging — e.g., if one episode is much shorter than others, that
    demo might have been cut short or failed.
    """
    logger.info("=" * 60)
    logger.info("EPISODE STRUCTURE")
    logger.info("=" * 60)

    boundaries = get_episode_boundaries(dataset)
    episode_lengths = np.array([b["length"] for b in boundaries])
    num_episodes = len(boundaries)

    ep_stats = {
        "num_episodes": num_episodes,
        "total_frames": int(episode_lengths.sum()),
        "min_length": int(episode_lengths.min()),
        "max_length": int(episode_lengths.max()),
        "mean_length": float(episode_lengths.mean()),
        "std_length": float(episode_lengths.std()),
        "lengths": episode_lengths.tolist(),
        "boundaries": boundaries,
    }

    logger.info(f"  num_episodes : {ep_stats['num_episodes']}")
    logger.info(f"  total_frames : {ep_stats['total_frames']}")
    logger.info(f"  min_length   : {ep_stats['min_length']}")
    logger.info(f"  max_length   : {ep_stats['max_length']}")
    logger.info(f"  mean_length  : {ep_stats['mean_length']:.1f}")
    logger.info(f"  std_length   : {ep_stats['std_length']:.1f}")

    # Print first few episode boundaries for reference
    logger.info("\n  First 5 episode boundaries:")
    for b in boundaries[:5]:
        logger.info(
            f"    Episode {b['episode_index']:3d}: "
            f"frames [{b['from']:5d} → {b['to']:5d}] "
            f"length={b['length']}"
        )

    # Plot episode length distribution
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(num_episodes), episode_lengths, color="steelblue", alpha=0.8)
    ax.set_xlabel("Episode Index")
    ax.set_ylabel("Number of Frames")
    ax.set_title("Episode Lengths")
    ax.axhline(
        y=episode_lengths.mean(), color="red", linestyle="--",
        label=f"Mean: {episode_lengths.mean():.0f}",
    )
    ax.legend()
    fig.tight_layout()
    plot_path = output_dir / "episode_lengths.png"
    fig.savefig(plot_path, dpi=100)
    plt.close(fig)
    logger.info(f"Saved episode length plot → {plot_path}")

    # Save episode stats
    stats_path = output_dir / "episode_stats.json"
    # Don't save full boundaries to JSON (can be large) — save summary only
    save_stats = {k: v for k, v in ep_stats.items() if k != "boundaries"}
    with open(stats_path, "w") as f:
        json.dump(save_stats, f, indent=2)
    logger.info(f"Saved episode stats → {stats_path}")

    return ep_stats


def inspect_state_action_ranges(
    dataset: LeRobotDataset,
    output_dir: Path,
    sample_every_n: int = 10,
) -> None:
    """Profile the full range and distribution of state and action features.

    Rather than looking at just a few frames, we sample across the entire
    dataset to get a true picture of value ranges.  This is essential for
    catching normalisation issues, clipping problems, or degenerate joints
    that never move.

    For the ALOHA dataset, observation.state and action are both [14]-dim
    vectors representing joint positions for two 7-DOF arms (left + right):
      [left_waist, left_shoulder, left_elbow, left_forearm_roll,
       left_wrist_angle, left_wrist_rotate, left_gripper,
       right_waist, right_shoulder, right_elbow, right_forearm_roll,
       right_wrist_angle, right_wrist_rotate, right_gripper]

    Args:
        sample_every_n: Sample every Nth frame to keep this fast.  For a
            20,000-frame dataset with sample_every_n=10, we inspect 2,000
            frames — more than enough for statistical profiles.
    """
    logger.info("=" * 60)
    logger.info(f"STATE/ACTION RANGE ANALYSIS (sampling every {sample_every_n} frames)")
    logger.info("=" * 60)

    states: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []

    indices = range(0, len(dataset), sample_every_n)
    for i in indices:
        sample = dataset[i]
        if "observation.state" in sample:
            states.append(sample["observation.state"])
        if "action" in sample:
            actions.append(sample["action"])

    logger.info(f"  Sampled {len(indices)} frames")

    def profile_and_plot(
        tensors: list[torch.Tensor],
        name: str,
        joint_names: list[str] | None,
    ) -> None:
        """Stack tensors, compute per-dimension stats, save box plot."""
        if not tensors:
            logger.info(f"  No {name} data found — skipping")
            return

        stacked = torch.stack(tensors)  # [N, D]
        n_dims = stacked.shape[1]

        labels = joint_names[:n_dims] if joint_names else [f"dim_{d}" for d in range(n_dims)]

        logger.info(f"\n  {name} — shape per frame: [{n_dims}]")
        logger.info(f"  {'Dimension':<25} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10}")
        logger.info(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

        per_dim_stats = []
        for d in range(n_dims):
            col = stacked[:, d]
            dim_stats = {
                "name": labels[d],
                "min": col.min().item(),
                "max": col.max().item(),
                "mean": col.mean().item(),
                "std": col.std().item(),
            }
            per_dim_stats.append(dim_stats)
            logger.info(
                f"  {labels[d]:<25} {dim_stats['min']:>10.4f} {dim_stats['max']:>10.4f} "
                f"{dim_stats['mean']:>10.4f} {dim_stats['std']:>10.4f}"
            )

        # Save per-dimension stats to JSON
        safe_name = name.lower().replace(" ", "_")
        stats_path = output_dir / f"{safe_name}_per_dim_stats.json"
        with open(stats_path, "w") as f:
            json.dump(per_dim_stats, f, indent=2)
        logger.info(f"  Saved per-dim stats → {stats_path}")

        # Box plot across dimensions
        fig, ax = plt.subplots(figsize=(max(10, n_dims * 0.8), 5))
        data_np = stacked.numpy()
        ax.boxplot(data_np, tick_labels=labels, vert=True)
        ax.set_title(f"{name} — Distribution per Dimension")
        ax.set_ylabel("Value")
        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()
        plot_path = output_dir / f"{safe_name}_boxplot.png"
        fig.savefig(plot_path, dpi=100)
        plt.close(fig)
        logger.info(f"  Saved box plot → {plot_path}")

    # Extract joint names from dataset features if available
    def get_joint_names(feature_key: str) -> list[str] | None:
        feat = dataset.features.get(feature_key, {})
        if isinstance(feat, dict):
            names_field = feat.get("names", {})
            if isinstance(names_field, dict) and "motors" in names_field:
                return names_field["motors"]
            elif isinstance(names_field, list):
                return names_field
        return None

    state_names = get_joint_names("observation.state")
    action_names = get_joint_names("action")

    profile_and_plot(states, "Observation State", state_names)
    profile_and_plot(actions, "Action", action_names)


def extract_sample_frames(
    dataset: LeRobotDataset,
    output_dir: Path,
    episode_indices: list[int] | None = None,
    frames_per_episode: int = 4,
) -> None:
    """Decode and save sample video frames from specified episodes.

    LeRobotDataset stores camera observations as MP4 videos.  When you access
    a frame via dataset[i], the library automatically decodes the relevant
    video frame using torchcodec/torchvision and returns it as a float32
    tensor of shape [C, H, W] in range [0, 1].

    We convert back to uint8 RGB images and save grids for visual inspection.

    Args:
        episode_indices: Which episodes to sample from.  Defaults to first 3.
        frames_per_episode: How many evenly-spaced frames per episode.
    """
    logger.info("=" * 60)
    logger.info("SAMPLE FRAME EXTRACTION")
    logger.info("=" * 60)

    if episode_indices is None:
        episode_indices = list(range(min(3, dataset.num_episodes)))

    # Find image/video feature keys from the features dict
    image_keys = [
        k for k, v in dataset.features.items()
        if isinstance(v, dict) and v.get("dtype") in ("video", "image")
    ]

    if not image_keys:
        logger.info("  No image/video features found — skipping frame extraction")
        return

    logger.info(f"  Image features: {image_keys}")
    logger.info(f"  Episodes to sample: {episode_indices}")

    # Get episode boundaries
    boundaries = get_episode_boundaries(dataset)

    for ep_idx in episode_indices:
        if ep_idx >= len(boundaries):
            logger.warning(f"  Episode {ep_idx} out of range — skipping")
            continue

        b = boundaries[ep_idx]
        from_idx = b["from"]
        ep_len = b["length"]

        # Pick evenly spaced frame indices within this episode
        frame_offsets = np.linspace(0, ep_len - 1, frames_per_episode, dtype=int)
        global_indices = [from_idx + int(offset) for offset in frame_offsets]

        for img_key in image_keys:
            images = []
            for gi in global_indices:
                sample = dataset[gi]
                img_tensor = sample[img_key]
                # Convert from [C, H, W] float [0,1] to [H, W, C] uint8
                img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                images.append(img_np)

            safe_key = img_key.replace(".", "_").replace("/", "_")
            path = output_dir / f"ep{ep_idx:03d}_{safe_key}_frames.png"
            save_image_grid(
                images,
                path,
                title=f"Episode {ep_idx} — {img_key} (frame offsets: {frame_offsets.tolist()})",
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect any LeRobotDataset — profiles fields, stats, and sample frames."
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
        default="outputs/inspection",
        help="Directory to save inspection outputs.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of individual frames to inspect in detail.",
    )
    parser.add_argument(
        "--sample-every-n",
        type=int,
        default=10,
        help="For range analysis, sample every Nth frame.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    # LeRobotDataset handles everything: downloads from HF Hub (if not
    # cached), parses the Parquet files for state/action data, sets up
    # video decoding for camera observations, and exposes a PyTorch
    # Dataset interface.  The first load downloads data to
    # ~/.cache/huggingface/lerobot/{repo_id}/.
    # ------------------------------------------------------------------
    logger.info(f"Loading dataset: {args.repo_id}")
    logger.info("(First run will download from HF Hub — may take a minute)")
    dataset = LeRobotDataset(args.repo_id)
    logger.info(f"Dataset loaded: {len(dataset)} frames")

    # ------------------------------------------------------------------
    # Run all inspections
    # ------------------------------------------------------------------
    inspect_metadata(dataset, output_dir)
    inspect_samples(dataset, output_dir, num_samples=args.num_samples)
    inspect_episode_structure(dataset, output_dir)
    inspect_state_action_ranges(dataset, output_dir, sample_every_n=args.sample_every_n)
    extract_sample_frames(dataset, output_dir)

    logger.info("=" * 60)
    logger.info("INSPECTION COMPLETE")
    logger.info(f"All outputs saved to: {output_dir}/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()