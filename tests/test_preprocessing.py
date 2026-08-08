import unittest

import numpy as np
import pandas as pd

from fraud_detection.preprocessing import (
    MODEL_FEATURE_COLUMNS,
    SCALED_AMOUNT_COLUMN,
    fit_transaction_preprocessor,
)


class TransactionPreprocessorTests(unittest.TestCase):
    def transactions(self, amounts, times=None) -> pd.DataFrame:
        row_count = len(amounts)
        if times is None:
            times = np.arange(row_count, dtype=float)
        return pd.DataFrame(
            {
                "Time": times,
                **{
                    f"V{index}": np.linspace(index, index + 1, row_count)
                    for index in range(1, 29)
                },
                "Amount": amounts,
                "Class": [0] * row_count,
            }
        )

    def test_fits_amount_scaling_from_training_transactions_only(self):
        train = self.transactions([10.0, 20.0])
        validation = self.transactions([1_000.0, 2_000.0])
        test = self.transactions([-500.0, 5_000.0])

        preprocessor = fit_transaction_preprocessor(train)
        transformed_train = preprocessor.transform(train)
        transformed_validation = preprocessor.transform(validation)
        transformed_test = preprocessor.transform(test)

        self.assertEqual(15.0, preprocessor.amount_mean)
        self.assertEqual(5.0, preprocessor.amount_scale)
        np.testing.assert_allclose(
            transformed_train[SCALED_AMOUNT_COLUMN],
            [-1.0, 1.0],
        )
        np.testing.assert_allclose(
            transformed_validation[SCALED_AMOUNT_COLUMN],
            [197.0, 397.0],
        )
        np.testing.assert_allclose(
            transformed_test[SCALED_AMOUNT_COLUMN],
            [-103.0, 997.0],
        )
        self.assertEqual(15.0, preprocessor.amount_mean)
        self.assertEqual(5.0, preprocessor.amount_scale)

    def test_validation_and_test_statistics_do_not_change_training_transform(self):
        train = self.transactions([100.0, 200.0, 300.0])
        first_preprocessor = fit_transaction_preprocessor(train)
        expected_train = first_preprocessor.transform(train)

        first_preprocessor.transform(self.transactions([1.0, 1.0]))
        first_preprocessor.transform(self.transactions([1_000_000.0, 2_000_000.0]))
        actual_train = first_preprocessor.transform(train)

        pd.testing.assert_frame_equal(actual_train, expected_train)

    def test_drops_time_and_returns_only_model_features(self):
        transactions = self.transactions(
            [10.0, 20.0],
            times=[0.0, 172_792.0],
        )

        transformed = fit_transaction_preprocessor(transactions).transform(
            transactions
        )

        self.assertEqual(
            [*MODEL_FEATURE_COLUMNS, SCALED_AMOUNT_COLUMN],
            transformed.columns.tolist(),
        )
        self.assertNotIn("Time", transformed.columns)
        self.assertNotIn("Amount", transformed.columns)
        self.assertNotIn("Class", transformed.columns)
        self.assertEqual(transactions.index.tolist(), transformed.index.tolist())


if __name__ == "__main__":
    unittest.main()
