import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class ComplaintVisualizer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        sns.set_style("whitegrid")

    def plot_product_distribution(self):
        """Plots the count of complaints per product."""
        plt.figure(figsize=(10, 5))
        sns.countplot(
            data=self.df, 
            y='CrediTrust_Product', 
            order=self.df['CrediTrust_Product'].value_counts().index, 
            palette='viridis'
        )
        plt.title('Distribution of Complaints by Product')
        plt.xlabel('Count')
        plt.show()

    def analyze_word_counts(self):
        """Calculates and plots word count distribution."""
        word_counts = self.df['cleaned_narrative'].apply(lambda x: len(str(x).split()))
        
        print(f"\n[Statistics] Avg Word Count: {word_counts.mean():.0f}")
        print(f"[Statistics] Max Word Count: {word_counts.max()}")

        plt.figure(figsize=(12, 5))
        sns.histplot(word_counts, bins=50, kde=True, color='teal')
        plt.title('Distribution of Complaint Word Counts')
        plt.xlabel('Words')
        plt.xlim(0, 1000)
        plt.show()