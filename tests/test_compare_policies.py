"""
Tests for the policy comparison report generator.

Verifies that comparison outputs exist and that the report
contains expected content.
"""

import json
from pathlib import Path

import pytest


class TestComparisonOutputs:
    """Verify that comparison run created expected output files."""

    @pytest.fixture
    def comparison_dir(self):
        d = Path("outputs/comparison")
        if not d.exists():
            pytest.skip("No comparison outputs found — run compare_policies.py first")
        return d

    def test_success_rate_plot_exists(self, comparison_dir):
        assert (comparison_dir / "success_rate_comparison.png").exists()

    def test_reward_plot_exists(self, comparison_dir):
        assert (comparison_dir / "reward_comparison.png").exists()

    def test_training_efficiency_plot_exists(self, comparison_dir):
        assert (comparison_dir / "training_efficiency.png").exists()

    def test_per_episode_heatmap_exists(self, comparison_dir):
        assert (comparison_dir / "per_episode_success.png").exists()

    def test_report_exists(self, comparison_dir):
        report = comparison_dir / "comparison_report.md"
        assert report.exists()
        assert report.stat().st_size > 0


class TestReportContent:
    """Verify the comparison report contains expected sections and data."""

    @pytest.fixture
    def report_text(self):
        path = Path("outputs/comparison/comparison_report.md")
        if not path.exists():
            pytest.skip("No comparison report found")
        return path.read_text()

    def test_has_results_table(self, report_text):
        assert "| Run |" in report_text
        assert "Success Rate" in report_text

    def test_mentions_all_runs(self, report_text):
        assert "ACT" in report_text
        assert "Diffusion" in report_text

    def test_has_key_findings(self, report_text):
        assert "Key Findings" in report_text

    def test_has_recommendations(self, report_text):
        assert "SO-101" in report_text

    def test_has_success_rates(self, report_text):
        # Our known results
        assert "44%" in report_text
        assert "58%" in report_text
        assert "10%" in report_text

    def test_has_visualizations_section(self, report_text):
        assert "success_rate_comparison.png" in report_text
        assert "reward_comparison.png" in report_text