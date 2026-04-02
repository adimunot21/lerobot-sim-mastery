"""
Tests for the dataset analysis toolkit.

Verifies that analysis functions produce expected output files
and that computed metrics are within reasonable ranges.
"""

import json
from pathlib import Path

import pytest


class TestAnalysisOutputs:
    """Verify that analysis run created expected output files."""

    @pytest.fixture
    def analysis_dir(self):
        d = Path("outputs/analysis")
        if not d.exists():
            pytest.skip("No analysis outputs found — run analyze_dataset.py first")
        return d

    def test_histogram_files_exist(self, analysis_dir):
        assert (analysis_dir / "action_histograms.png").exists()
        assert (analysis_dir / "state_histograms.png").exists()

    def test_trajectory_files_exist(self, analysis_dir):
        trajectory_files = list(analysis_dir.glob("ep*_trajectory.png"))
        assert len(trajectory_files) > 0, "No trajectory plots found"

    def test_smoothness_files_exist(self, analysis_dir):
        assert (analysis_dir / "action_smoothness.png").exists()
        assert (analysis_dir / "action_smoothness.json").exists()

    def test_correlation_heatmap_exists(self, analysis_dir):
        assert (analysis_dir / "action_correlation_heatmap.png").exists()

    def test_outlier_detection_exists(self, analysis_dir):
        assert (analysis_dir / "outlier_detection.png").exists()
        assert (analysis_dir / "outlier_detection.json").exists()


class TestSmoothnessMetrics:
    """Verify smoothness metrics are within expected ranges for ALOHA sim data."""

    @pytest.fixture
    def smoothness_data(self):
        path = Path("outputs/analysis/action_smoothness.json")
        if not path.exists():
            pytest.skip("No smoothness data found")
        return json.loads(path.read_text())

    def test_overall_mean_delta_reasonable(self, smoothness_data):
        mean_delta = smoothness_data["overall_mean_delta"]
        # ALOHA sim data is very smooth — mean delta should be small
        assert 0.01 < mean_delta < 0.1, f"Unexpected mean delta: {mean_delta}"

    def test_overall_std_small(self, smoothness_data):
        std_delta = smoothness_data["overall_std_delta"]
        # Std across episodes should be small for consistent sim data
        assert std_delta < 0.01, f"Unexpected std: {std_delta}"

    def test_all_episodes_present(self, smoothness_data):
        episodes = smoothness_data["episodes"]
        assert len(episodes) == 50, f"Expected 50 episodes, got {len(episodes)}"

    def test_no_zero_delta_episodes(self, smoothness_data):
        for ep in smoothness_data["episodes"]:
            assert ep["mean_delta_l2"] > 0, f"Episode {ep['episode_index']} has zero delta"


class TestOutlierDetection:
    """Verify outlier detection results."""

    @pytest.fixture
    def outlier_data(self):
        path = Path("outputs/analysis/outlier_detection.json")
        if not path.exists():
            pytest.skip("No outlier data found")
        return json.loads(path.read_text())

    def test_total_episodes(self, outlier_data):
        assert outlier_data["total_episodes"] == 50

    def test_outliers_have_required_fields(self, outlier_data):
        for outlier in outlier_data["outlier_episodes"]:
            assert "episode_index" in outlier
            assert "metric" in outlier
            assert "z_score" in outlier
            assert abs(outlier["z_score"]) > 2.0, "Outlier z-score should exceed threshold"