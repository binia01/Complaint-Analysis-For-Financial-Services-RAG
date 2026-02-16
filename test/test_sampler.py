"""Tests for src.sampler.ComplaintSampler."""

import pandas as pd
import pytest

from src.sampler import ComplaintSampler


@pytest.fixture()
def large_df() -> pd.DataFrame:
    """DataFrame with 200 rows across 4 products."""
    products = ["Credit Cards", "Savings Accounts", "Personal Loans", "Money Transfers"]
    rows = []
    for i in range(200):
        rows.append(
            {
                "CrediTrust_Product": products[i % len(products)],
                "Consumer complaint narrative": f"Complaint text {i}",
            }
        )
    return pd.DataFrame(rows)


class TestStratifiedSample:
    def test_sample_size(self, large_df: pd.DataFrame) -> None:
        sampler = ComplaintSampler(large_df)
        sample = sampler.stratified_sample(n_samples=50)
        assert len(sample) <= 50

    def test_preserves_products(self, large_df: pd.DataFrame) -> None:
        sampler = ComplaintSampler(large_df)
        sample = sampler.stratified_sample(n_samples=100)
        assert set(sample["CrediTrust_Product"]) == set(
            large_df["CrediTrust_Product"]
        )

    def test_reproducibility(self, large_df: pd.DataFrame) -> None:
        s1 = ComplaintSampler(large_df).stratified_sample(50, random_state=42)
        s2 = ComplaintSampler(large_df).stratified_sample(50, random_state=42)
        pd.testing.assert_frame_equal(s1, s2)
