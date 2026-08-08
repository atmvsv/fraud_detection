import unittest

import numpy as np
import pandas as pd

from fraud_detection.splits import create_stratified_splits


class CreateStratifiedSplitsTests(unittest.TestCase):
    def transactions(self) -> pd.DataFrame:
        row_count = 1_000
        fraud_count = 100
        return pd.DataFrame(
            {
                "Time": np.arange(row_count, dtype=float),
                **{
                    f"V{index}": np.linspace(index, index + 1, row_count)
                    for index in range(1, 29)
                },
                "Amount": np.linspace(1.0, 1_000.0, row_count),
                "Class": [0] * (row_count - fraud_count) + [1] * fraud_count,
            }
        )

    def test_repeats_same_assignments_for_same_seed(self):
        transactions = self.transactions()

        first = create_stratified_splits(transactions, random_state=42)
        second = create_stratified_splits(transactions, random_state=42)

        pd.testing.assert_frame_equal(first.train, second.train)
        pd.testing.assert_frame_equal(first.validation, second.validation)
        pd.testing.assert_frame_equal(first.test, second.test)

    def test_partitions_every_transaction_without_overlap(self):
        transactions = self.transactions()

        splits = create_stratified_splits(transactions)

        train_ids = set(splits.train.index)
        validation_ids = set(splits.validation.index)
        test_ids = set(splits.test.index)

        self.assertEqual(len(splits.train), 640)
        self.assertEqual(len(splits.validation), 160)
        self.assertEqual(len(splits.test), 200)
        self.assertTrue(train_ids.isdisjoint(validation_ids))
        self.assertTrue(train_ids.isdisjoint(test_ids))
        self.assertTrue(validation_ids.isdisjoint(test_ids))
        self.assertEqual(
            train_ids | validation_ids | test_ids,
            set(transactions.index),
        )

    def test_preserves_fraud_prevalence_within_rounding_tolerance(self):
        transactions = self.transactions()
        source_prevalence = transactions["Class"].mean()

        splits = create_stratified_splits(transactions)

        for split in (splits.train, splits.validation, splits.test):
            with self.subTest(split_size=len(split)):
                self.assertAlmostEqual(
                    split["Class"].mean(),
                    source_prevalence,
                    delta=1 / len(split),
                )


if __name__ == "__main__":
    unittest.main()
