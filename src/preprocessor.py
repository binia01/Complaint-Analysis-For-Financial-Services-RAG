"""Complaint text preprocessing and product mapping."""

import logging
import re
from typing import Dict

import pandas as pd

logger = logging.getLogger(__name__)

NARRATIVE_COLUMN = "Consumer complaint narrative"

# CrediTrust product taxonomy
PRODUCT_MAP: Dict[str, list[str]] = {
    "Credit Cards": ["credit card", "prepaid card"],
    "Savings Accounts": ["savings", "checking", "bank account"],
    "Personal Loans": ["personal loan", "consumer loan", "installment loan"],
    "Money Transfers": ["money transfer", "virtual currency"],
}


class ComplaintPreprocessor:
    """Clean, filter, and categorise complaint data."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    # ------------------------------------------------------------------
    # Public pipeline steps (each returns *self* for chaining)
    # ------------------------------------------------------------------

    def filter_missing_narratives(self) -> "ComplaintPreprocessor":
        """Remove rows where the complaint narrative is empty."""
        initial_count = len(self.df)
        self.df = self.df.dropna(subset=[NARRATIVE_COLUMN]).copy()
        dropped = initial_count - len(self.df)
        logger.info("Dropped %s rows missing narratives.", f"{dropped:,}")
        return self

    def map_products(self) -> "ComplaintPreprocessor":
        """Apply CrediTrust-specific product categorisation."""
        self.df["CrediTrust_Product"] = self.df["Product"].apply(
            self._map_product_logic
        )
        self.df = self.df[self.df["CrediTrust_Product"] != "Other"].copy()
        logger.info(
            "Filtered to target product categories. Rows: %s",
            f"{len(self.df):,}",
        )
        return self

    def clean_narratives(self) -> "ComplaintPreprocessor":
        """Apply text cleaning to the narrative column."""
        logger.info("Cleaning text narratives...")
        self.df["cleaned_narrative"] = self.df[NARRATIVE_COLUMN].apply(
            self._clean_text_logic
        )
        self.df = self.df[self.df["cleaned_narrative"] != ""]
        return self

    def get_data(self) -> pd.DataFrame:
        """Return the processed DataFrame."""
        return self.df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _map_product_logic(product_name: str) -> str:
        """Map a raw product name to the CrediTrust taxonomy."""
        lower = str(product_name).lower()
        for category, keywords in PRODUCT_MAP.items():
            if any(kw in lower for kw in keywords):
                return category
        return "Other"

    @staticmethod
    def _clean_text_logic(text: str) -> str:
        """Lowercase, strip redaction markers and special characters."""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r"x{2,}", "", text)
        text = re.sub(r"[^a-z0-9\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
