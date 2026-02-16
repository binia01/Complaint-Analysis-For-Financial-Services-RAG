"""conftest – shared fixtures for the test suite."""

import pandas as pd
import pytest


@pytest.fixture()
def sample_complaints_df() -> pd.DataFrame:
    """Minimal DataFrame that mirrors post-preprocessing shape."""
    return pd.DataFrame(
        {
            "Complaint ID": [1, 2, 3, 4],
            "Product": [
                "Credit card",
                "Savings account",
                "Personal loan",
                "Money transfer",
            ],
            "Consumer complaint narrative": [
                "I was charged twice for the same item on my credit card.",
                "My savings account had an unauthorised withdrawal.",
                "The personal loan interest rate changed without notice.",
                "My money transfer was delayed by two weeks.",
            ],
            "Issue": ["Billing", "Fraud", "Interest rate", "Delay"],
            "Date received": [
                "2024-01-01",
                "2024-02-01",
                "2024-03-01",
                "2024-04-01",
            ],
            "State": ["CA", "NY", "TX", "FL"],
            "CrediTrust_Product": [
                "Credit Cards",
                "Savings Accounts",
                "Personal Loans",
                "Money Transfers",
            ],
            "cleaned_narrative": [
                "i was charged twice for the same item on my credit card",
                "my savings account had an unauthorised withdrawal",
                "the personal loan interest rate changed without notice",
                "my money transfer was delayed by two weeks",
            ],
        }
    )
