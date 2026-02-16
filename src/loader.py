"""Chunked CSV loader for complaint data."""

import logging
import os
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)

NARRATIVE_COLUMN = "Consumer complaint narrative"


class ComplaintLoader:
    """Load complaint CSV files in memory-efficient chunks."""

    def __init__(
        self,
        file_path: str,
        columns: List[str],
        chunk_size: int = 50_000,
    ) -> None:
        self.file_path = file_path
        self.columns = columns
        self.chunk_size = chunk_size

    def load_data(self) -> pd.DataFrame:
        """Load data in chunks, filtering rows without narratives early.

        Returns:
            pd.DataFrame: DataFrame containing only rows with narratives.

        Raises:
            FileNotFoundError: If the source CSV does not exist.
            IOError: If there is an error reading the CSV.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found at {self.file_path}")

        logger.info("Loading data from %s in chunks...", self.file_path)

        chunks_list: List[pd.DataFrame] = []

        try:
            reader = pd.read_csv(
                self.file_path,
                usecols=self.columns,
                chunksize=self.chunk_size,
                low_memory=False,
            )
            with reader:
                for i, chunk in enumerate(reader):
                    clean_chunk = chunk.dropna(subset=[NARRATIVE_COLUMN])
                    if not clean_chunk.empty:
                        chunks_list.append(clean_chunk)
                    if i % 10 == 0:
                        logger.debug("Processed chunk %d", i)

            if not chunks_list:
                return pd.DataFrame(columns=self.columns)

            logger.info("Concatenating valid chunks...")
            df: pd.DataFrame = pd.concat(
                chunks_list, axis=0, ignore_index=True
            )

            logger.info(
                "Successfully loaded %s rows with narratives.",
                f"{len(df):,}",
            )
            return df

        except Exception as exc:
            raise IOError(f"Error reading CSV: {exc}") from exc
