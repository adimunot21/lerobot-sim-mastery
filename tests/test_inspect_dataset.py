"""
Tests for the dataset inspection script.

Tests core helper functions and verifies the inspection pipeline
produces expected outputs.  Uses the cached ALOHA dataset if available,
otherwise skips dataset-dependent tests gracefully.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.inspect_dataset import (
    tensor_stats,
    save_image_grid,
    get_episode_boundaries,
)


# ---------------------------------------------------------------------------
# Unit tests for helper functions (no dataset needed)
# ---------------------------------------------------------------------------

class TestTensorStats:
    """Test tensor_stats helper."""

    def test_basic_float_tensor(self):
        t = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = tensor_stats(t)
        assert stats["shape"] == [5]
        assert stats["dtype"] == "torch.float32"
        assert stats["min"] == pytest.approx(1.0)
        assert stats["max"] == pytest.approx(5.0)
        assert stats["mean"] == pytest.approx(3.0)
        assert stats["std"] == pytest.approx(1.5811, rel=1e-3)

    def test_single_element(self):
        t = torch.tensor([42.0])
        stats = tensor_stats(t)
        assert stats["min"] == pytest.approx(42.0)
        assert stats["max"] == pytest.approx(42.0)
        assert stats["std"] == 0.0

    def test_integer_tensor(self):
        t = torch.tensor([1, 2, 3])
        stats = tensor_stats(t)
        assert stats["dtype"] == "torch.int64"
        assert stats["mean"] == pytest.approx(2.0)

    def test_negative_values(self):
        t = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
        stats = tensor_stats(t)
        assert stats["min"] == pytest.approx(-3.0)
        assert stats["max"] == pytest.approx(3.0)
        assert stats["mean"] == pytest.approx(0.0)

    def test_2d_tensor(self):
        t = torch.randn(3, 480, 640)
        stats = tensor_stats(t)
        assert stats["shape"] == [3, 480, 640]


class TestSaveImageGrid:
    """Test image grid saving."""

    def test_saves_single_image(self, tmp_path):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        path = tmp_path / "test_grid.png"
        save_image_grid([img], path, title="Test")
        assert path.exists()
        assert path.stat().st_size > 0

    def test_saves_multiple_images(self, tmp_path):
        imgs = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(4)]
        path = tmp_path / "test_grid_multi.png"
        save_image_grid(imgs, path, title="Multi")
        assert path.exists()

    def test_no_title(self, tmp_path):
        img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        path = tmp_path / "no_title.png"
        save_image_grid([img], path)
        assert path.exists()


# ---------------------------------------------------------------------------
# Integration test: full inspection pipeline
# ---------------------------------------------------------------------------

class TestInspectionPipeline:
    """Test that the full inspection produces expected output files.

    Only runs if the ALOHA dataset is already cached locally.
    """

    @pytest.fixture
    def output_dir(self, tmp_path):
        return tmp_path / "inspection_test"

    def test_inspection_outputs_exist(self):
        """Verify that previous inspection run created expected files."""
        inspection_dir = Path("outputs/inspection")
        if not inspection_dir.exists():
            pytest.skip("No inspection outputs found — run inspect_dataset.py first")

        expected_files = [
            "metadata.json",
            "sample_stats.json",
            "episode_stats.json",
            "episode_lengths.png",
            "observation_state_boxplot.png",
            "action_boxplot.png",
            "observation_state_per_dim_stats.json",
            "action_per_dim_stats.json",
        ]

        for fname in expected_files:
            fpath = inspection_dir / fname
            assert fpath.exists(), f"Missing expected output: {fname}"
            assert fpath.stat().st_size > 0, f"Empty file: {fname}"

    def test_metadata_content(self):
        """Verify metadata JSON has expected structure."""
        meta_path = Path("outputs/inspection/metadata.json")
        if not meta_path.exists():
            pytest.skip("No metadata.json found")

        meta = json.loads(meta_path.read_text())
        assert meta["repo_id"] == "lerobot/aloha_sim_transfer_cube_human"
        assert meta["num_episodes"] == 50
        assert meta["num_frames"] == 20000
        assert meta["fps"] == 50
        assert "observation.state" in meta["features"]
        assert "action" in meta["features"]
        assert "observation.images.top" in meta["features"]

    def test_episode_stats_content(self):
        """Verify episode stats have expected values."""
        stats_path = Path("outputs/inspection/episode_stats.json")
        if not stats_path.exists():
            pytest.skip("No episode_stats.json found")

        stats = json.loads(stats_path.read_text())
        assert stats["num_episodes"] == 50
        assert stats["total_frames"] == 20000
        assert stats["mean_length"] == pytest.approx(400.0)