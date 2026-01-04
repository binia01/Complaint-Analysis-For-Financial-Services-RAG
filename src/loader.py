import pandas as pd
import os
from typing import List

class ComplaintLoader:
    def __init__(self, file_path: str, columns: List[str]):
        self.file_path = file_path
        self.columns = columns

    def load_data(self) -> pd.DataFrame:
        """
        Loads data in chunks to prevent Memory Errors (Kernel crash).
        It immediately filters out rows without narratives to save RAM.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found at {self.file_path}")
        
        print(f"[Loader] Loading data from {self.file_path} in chunks...")
        
        chunk_size = 50000  # Process 50k rows at a time
        chunks_list = []
        
        try:
            # Create an iterator to read the file in pieces
            with pd.read_csv(self.file_path, usecols=self.columns, chunksize=chunk_size, low_memory=False) as reader:
                for i, chunk in enumerate(reader):
                    # OPTIMIZATION: Drop rows with missing narratives IMMEDIATELY
                    # This dramatically reduces memory usage before we combine everything
                    clean_chunk = chunk.dropna(subset=['Consumer complaint narrative'])
                    
                    if not clean_chunk.empty:
                        chunks_list.append(clean_chunk)
                    
                    # Optional: Print progress every 10 chunks
                    if i % 10 == 0:
                        print(f"  Processed chunk {i}...")

            print("[Loader] Concatenating valid chunks...")
            df = pd.concat(chunks_list, axis=0, ignore_index=True)
            
            print(f"[Loader] Successfully loaded {len(df):,} rows with narratives.")
            return df

        except Exception as e:
            raise IOError(f"Error reading CSV: {e}")