"""Tests for src.loader.ComplaintLoader."""

import os
import textwrap
from pathlib import Path
from typing import List

import pandas as pd
import pytest

from src.loader import ComplaintLoader


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture()
def csv_path(tmp_path: Path) -> str:
    """Write a small CSV to a temp dir and return its path."""
    content = textwrap.dedent("""\
        Complaint ID,Product,Consumer complaint narrative,Issue,Date received,State
        1,Credit card,I was overcharged on my bill.,Billing,2024-01-01,CA
        2,Savings account,,Fraud,2024-02-01,NY
        3,Personal loan,The interest rate changed suddenly.,Interest rate,2024-03-01,TX
    """)
    file = tmp_path / "complaints.csv"
    file.write_text(content)
    return str(file)


COLUMNS: List[str] = [
    "Complaint ID",
    "Product",
    "Consumer complaint narrative",
    "Issue",
    "Date received",
    "State",
]


# ── Tests ─────────────────────────────────────────────────────────────


class TestComplaintLoader:
    """Unit tests for ComplaintLoader."""

    def test_load_returns_dataframe(self, csv_path: str) -> None:
        loader = ComplaintLoader(file_path=csv_path, columns=COLUMNS)
        df = loader.load_data()
        assert isinstance(df, pd.DataFrame)

    def test_filters_missing_narratives(self, csv_path: str) -> None:
        """Row 2 has an empty narrative and should be dropped."""
        loader = ComplaintLoader(file_path=csv_path, columns=COLUMNS)
        df = loader.load_data()
        assert len(df) == 2  # rows 1 and 3 only

    def test_correct_columns(self, csv_path: str) -> None:
        loader = ComplaintLoader(file_path=csv_path, columns=COLUMNS)
        df = loader.load_data()
        for col in COLUMNS:
            assert col in df.columns

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        loader = ComplaintLoader(
            file_path=str(tmp_path / "nonexistent.csv"), columns=COLUMNS
        )
        with pytest.raises(FileNotFoundError):
            loader.load_data()

    def test_empty_csv_returns_empty_df(self, tmp_path: Path) -> None:
        """A CSV with only headers and no narrative data."""
        file = tmp_path / "empty.csv"
        file.write_text(
            "Complaint ID,Product,Consumer complaint narrative,"
            "Issue,Date received,State\n"
        )
        loader = ComplaintLoader(file_path=str(file), columns=COLUMNS)
        df = loader.load_data()
        assert len(df) == 0

    def test_custom_chunk_size(self, csv_path: str) -> None:
        loader = ComplaintLoader(
            file_path=csv_path, columns=COLUMNS, chunk_size=1
        )
        df = loader.load_data()
        assert len(df) == 2
