"""Plotting helpers for complaint data exploration."""

import logging
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


class ComplaintVisualizer:
    """Generate EDA charts for complaint DataFrames."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        sns.set_style("whitegrid")

    def plot_product_distribution(
        self, figsize: tuple[int, int] = (10, 5)
    ) -> None:
        """Plot the count of complaints per product."""
        plt.figure(figsize=figsize)
        sns.countplot(
            data=self.df,
            y="CrediTrust_Product",
            order=self.df["CrediTrust_Product"].value_counts().index,
            palette="viridis",
        )
        plt.title("Distribution of Complaints by Product")
        plt.xlabel("Count")
        plt.show()

    def analyze_word_counts(
        self,
        figsize: tuple[int, int] = (12, 5),
        x_limit: Optional[int] = 1000,
    ) -> None:
        """Calculate and plot word-count distribution."""
        word_counts = self.df["cleaned_narrative"].apply(
            lambda x: len(str(x).split())
        )

        logger.info("Avg word count: %.0f", word_counts.mean())
        logger.info("Max word count: %d", word_counts.max())

        plt.figure(figsize=figsize)
        sns.histplot(word_counts, bins=50, kde=True, color="teal")
        plt.title("Distribution of Complaint Word Counts")
        plt.xlabel("Words")
        if x_limit:
            plt.xlim(0, x_limit)
        plt.show()
