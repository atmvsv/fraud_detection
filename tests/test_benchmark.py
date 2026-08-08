import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from fraud_detection.benchmark import main
from fraud_detection.preprocessing import SCALED_AMOUNT_COLUMN


class FakeFraudScoreModel:
    def __init__(self, name, events):
        self.name = name
        self.events = events

    def fraud_scores(self, features):
        self.events.append(f"score:{self.name}:{len(features)}")
        return features["V1"].to_numpy(dtype=float)


class BenchmarkWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.csv_path = Path(self.temporary_directory.name) / "transactions.csv"
        self.transactions().to_csv(self.csv_path, index=False)

    def transactions(self) -> pd.DataFrame:
        row_count = 100
        labels = np.array([0] * 80 + [1] * 20)
        return pd.DataFrame(
            {
                "Time": np.arange(row_count, dtype=float),
                "V1": labels.astype(float),
                **{
                    f"V{index}": np.linspace(index, index + 1, row_count)
                    for index in range(2, 29)
                },
                "Amount": np.linspace(1.0, 100.0, row_count),
                "Class": labels,
            }
        )

    def test_command_runs_complete_synthetic_benchmark(self):
        events = []
        output = io.StringIO()

        def fitted_model(name):
            events.append(f"fit:{name}")
            return FakeFraudScoreModel(name, events)

        with (
            patch(
                "fraud_detection.benchmark.fit_logistic_regression_baseline",
                side_effect=lambda *args, **kwargs: fitted_model("logistic"),
            ) as fit_logistic_regression,
            patch(
                "fraud_detection.benchmark.fit_isolation_forest_baseline",
                side_effect=lambda *args, **kwargs: fitted_model("isolation"),
            ) as fit_isolation_forest,
            patch(
                "fraud_detection.benchmark.fit_catboost_model",
                side_effect=lambda *args, **kwargs: fitted_model("catboost"),
            ) as fit_catboost,
            redirect_stdout(output),
        ):
            exit_code = main([str(self.csv_path), "--recall-target", "1.0"])

        self.assertEqual(0, exit_code)
        result = json.loads(output.getvalue())
        self.assertEqual(64, result["train_size"])
        self.assertEqual(16, result["validation_size"])
        self.assertEqual(20, result["test_size"])
        self.assertEqual(
            [
                "fit:logistic",
                "fit:isolation",
                "fit:catboost",
                "score:logistic:16",
                "score:isolation:16",
                "score:catboost:16",
                "score:logistic:20",
                "score:isolation:20",
                "score:catboost:20",
            ],
            events,
        )

        for model_name in ("logistic_regression", "isolation_forest", "catboost"):
            with self.subTest(model=model_name):
                model_result = result[model_name]
                selected_threshold = model_result["threshold_selection"]["threshold"]
                test_report = model_result["test_report"]
                self.assertEqual(selected_threshold, test_report["threshold"])
                self.assertEqual(1.0, test_report["recall"])
                self.assertEqual(1.0, test_report["precision"])

        logistic_features, logistic_labels = fit_logistic_regression.call_args.args
        isolation_features = fit_isolation_forest.call_args.args[0]
        (
            catboost_training_features,
            catboost_training_labels,
            catboost_validation_features,
            catboost_validation_labels,
        ) = fit_catboost.call_args.args

        self.assertEqual(64, len(logistic_features))
        self.assertEqual(64, len(logistic_labels))
        self.assertEqual(64, len(isolation_features))
        self.assertEqual(64, len(catboost_training_features))
        self.assertEqual(64, len(catboost_training_labels))
        self.assertEqual(16, len(catboost_validation_features))
        self.assertEqual(16, len(catboost_validation_labels))

        for features in (
            logistic_features,
            isolation_features,
            catboost_training_features,
            catboost_validation_features,
        ):
            self.assertNotIn("Time", features.columns)
            self.assertNotIn("Amount", features.columns)
            self.assertIn(SCALED_AMOUNT_COLUMN, features.columns)


if __name__ == "__main__":
    unittest.main()
