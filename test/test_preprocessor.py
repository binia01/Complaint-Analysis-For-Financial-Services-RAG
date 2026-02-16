"""Tests for src.preprocessor.ComplaintPreprocessor."""

import pandas as pd
import pytest

from src.preprocessor import ComplaintPreprocessor


@pytest.fixture()
def raw_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Product": [
                "Credit card",
                "Bank account or service",
                "Student loan",
                "Money transfer",
                "Prepaid card",
            ],
            "Consumer complaint narrative": [
                "I was charged XXXX twice!",
                "Unauthorised withdrawal from checking.",
                "Student loan help",
                "My money transfer was delayed.",
                None,  # should be dropped
            ],
        }
    )


class TestFilterMissingNarratives:
    def test_drops_none_rows(self, raw_df: pd.DataFrame) -> None:
        pp = ComplaintPreprocessor(raw_df).filter_missing_narratives()
        assert len(pp.get_data()) == 4


class TestMapProducts:
    def test_maps_known_products(self, raw_df: pd.DataFrame) -> None:
        pp = (
            ComplaintPreprocessor(raw_df)
            .filter_missing_narratives()
            .map_products()
        )
        products = set(pp.get_data()["CrediTrust_Product"])
        assert "Credit Cards" in products
        assert "Savings Accounts" in products
        assert "Money Transfers" in products

    def test_filters_other(self, raw_df: pd.DataFrame) -> None:
        pp = (
            ComplaintPreprocessor(raw_df)
            .filter_missing_narratives()
            .map_products()
        )
        assert "Other" not in set(pp.get_data()["CrediTrust_Product"])


class TestCleanNarratives:
    def test_removes_redaction_markers(self) -> None:
        df = pd.DataFrame(
            {
                "Product": ["Credit card"],
                "Consumer complaint narrative": ["I paid XXXX on my card!!!"],
            }
        )
        pp = ComplaintPreprocessor(df).clean_narratives()
        text = pp.get_data()["cleaned_narrative"].iloc[0]
        assert "xxxx" not in text
        assert "!" not in text


class TestChaining:
    def test_full_pipeline(self, raw_df: pd.DataFrame) -> None:
        result = (
            ComplaintPreprocessor(raw_df)
            .filter_missing_narratives()
            .map_products()
            .clean_narratives()
            .get_data()
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
