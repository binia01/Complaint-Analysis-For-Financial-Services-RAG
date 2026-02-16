"""Stratified sampling utility for complaint data."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class ComplaintSampler:
    """Create stratified samples preserving product distribution."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def stratified_sample(
        self, n_samples: int = 10_000, random_state: int = 42
    ) -> pd.DataFrame:
        """Return a stratified sample of *n_samples* complaints.

        Args:
            n_samples: Target number of rows in the sample.
            random_state: Seed for reproducibility.

        Returns:
            A DataFrame of approximately *n_samples* rows.
        """
        logger.info(
            "Creating stratified sample of %s complaints...",
            f"{n_samples:,}",
        )

        frac: float = n_samples / len(self.df)

        sample_df: pd.DataFrame = self.df.groupby(
            "CrediTrust_Product", group_keys=False
        ).apply(
            lambda x: x.sample(frac=frac, random_state=random_state),
        )

        if len(sample_df) > n_samples:
            sample_df = sample_df.sample(
                n=n_samples, random_state=random_state
            )

        logger.info("Sample shape: %s", sample_df.shape)
        return sample_df.reset_index(drop=True)
