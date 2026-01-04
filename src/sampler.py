import pandas as pd

class ComplaintSampler:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def stratified_sample(self, n_samples: int = 10000) -> pd.DataFrame:
        """
        Creates a sample preserving the percentage distribution of Products.
        """
        print(f"[Sampler] Creating stratified sample of {n_samples} complaints...")
        
        # Calculate the fraction of the total dataset needed to get n_samples
        frac = n_samples / len(self.df)
        
        # Group by Product and sample
        # random_state=42 ensures reproducibility
        sample_df = self.df.groupby('CrediTrust_Product', group_keys=False).apply(
            lambda x: x.sample(frac=frac, random_state=42)
        )
        
        # Depending on rounding, we might be slightly off, so we cap or slice
        if len(sample_df) > n_samples:
            sample_df = sample_df.sample(n=n_samples, random_state=42)
            
        print(f"[Sampler] Sample shape: {sample_df.shape}")
        return sample_df.reset_index(drop=True)