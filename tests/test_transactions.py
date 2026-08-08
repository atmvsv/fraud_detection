import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_detection.transactions import TransactionDataError, load_transactions


class LoadTransactionsTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.csv_path = Path(self.temporary_directory.name) / "transactions.csv"

    def valid_transactions(self) -> pd.DataFrame:
        row_count = 20
        data = {
            "Time": np.arange(row_count, dtype=float),
            **{
                f"V{index}": np.linspace(index, index + 1, row_count)
                for index in range(1, 29)
            },
            "Amount": np.linspace(1.0, 100.0, row_count),
            "Class": [0] * 12 + [1] * 8,
        }
        return pd.DataFrame(data)

    def load(self, transactions: pd.DataFrame) -> pd.DataFrame:
        transactions.to_csv(self.csv_path, index=False)
        return load_transactions(self.csv_path)

    def test_loads_valid_transaction_data(self):
        transactions = self.valid_transactions()

        loaded = self.load(transactions)

        pd.testing.assert_frame_equal(loaded, transactions)

    def test_rejects_missing_required_fields(self):
        transactions = self.valid_transactions().drop(columns="Amount")

        with self.assertRaisesRegex(TransactionDataError, "Missing required.*Amount"):
            self.load(transactions)

    def test_rejects_unexpected_fraud_labels(self):
        transactions = self.valid_transactions()
        transactions.loc[0, "Class"] = 2

        with self.assertRaisesRegex(TransactionDataError, "Unexpected fraud labels"):
            self.load(transactions)

    def test_rejects_missing_numeric_values(self):
        transactions = self.valid_transactions()
        transactions.loc[0, "V1"] = np.nan

        with self.assertRaisesRegex(TransactionDataError, "Missing numeric values.*V1"):
            self.load(transactions)

    def test_rejects_malformed_feature_values(self):
        transactions = self.valid_transactions()
        transactions["V2"] = transactions["V2"].astype(object)
        transactions.loc[0, "V2"] = "not-a-number"

        with self.assertRaisesRegex(
            TransactionDataError, "Malformed numeric feature values.*V2"
        ):
            self.load(transactions)

    def test_rejects_too_few_fraud_examples_for_current_benchmark(self):
        transactions = self.valid_transactions()
        transactions["Class"] = [0] * 16 + [1] * 4

        with self.assertRaisesRegex(
            TransactionDataError, "Not enough fraudulent transactions"
        ):
            self.load(transactions)


if __name__ == "__main__":
    unittest.main()
