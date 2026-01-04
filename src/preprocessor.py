import pandas as pd
import re

class ComplaintPreprocessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def filter_missing_narratives(self) -> 'ComplaintPreprocessor':
        """Removes rows where the complaint narrative is empty."""
        initial_count = len(self.df)
        self.df = self.df.dropna(subset=['Consumer complaint narrative']).copy()
        dropped = initial_count - len(self.df)
        print(f"[Preprocessing] Dropped {dropped:,} rows missing narratives.")
        return self

    def _map_product_logic(self, product_name: str) -> str:
        """Internal helper method for product taxonomy mapping."""
        p = str(product_name).lower()
        if 'credit card' in p or 'prepaid card' in p: return 'Credit Cards'
        if 'savings' in p or 'checking' in p or 'bank account' in p: return 'Savings Accounts'
        if 'personal loan' in p or 'consumer loan' in p or 'installment loan' in p: return 'Personal Loans'
        if 'money transfer' in p or 'virtual currency' in p: return 'Money Transfers'
        return 'Other'

    def map_products(self) -> 'ComplaintPreprocessor':
        """Applies CrediTrust specific product categorization."""
        self.df['CrediTrust_Product'] = self.df['Product'].apply(self._map_product_logic)
        # Filter out 'Other' immediately
        self.df = self.df[self.df['CrediTrust_Product'] != 'Other'].copy()
        print(f"[Preprocessing] Filtered to target product categories. Current rows: {len(self.df):,}")
        return self

    def _clean_text_logic(self, text: str) -> str:
        """Internal helper for regex text cleaning."""
        if not isinstance(text, str): return ""
        text = text.lower()
        text = re.sub(r'x{2,}', '', text) # Remove redacted placeholders (XXXX)
        text = re.sub(r'[^a-z0-9\s]', '', text) # Remove special chars
        text = re.sub(r'\s+', ' ', text).strip() # Collapse whitespace
        return text

    def clean_narratives(self) -> 'ComplaintPreprocessor':
        """Applies text cleaning to the narrative column."""
        print("[Preprocessing] Cleaning text narratives (this may take a moment)...")
        self.df['cleaned_narrative'] = self.df['Consumer complaint narrative'].apply(self._clean_text_logic)
        self.df = self.df[self.df['cleaned_narrative'] != ""]
        return self

    def get_data(self) -> pd.DataFrame:
        """Returns the processed DataFrame."""
        return self.df