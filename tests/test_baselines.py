import unittest

import numpy as np

from fraud_detection.baselines import (
    evaluate_baseline,
    fit_isolation_forest_baseline,
    fit_logistic_regression_baseline,
)


class BaselineModelTests(unittest.TestCase):
    def test_logistic_regression_scores_fraud_like_rows_higher(self):
        training_features = np.array(
            [
                [-3.0, -1.0],
                [-2.0, -0.5],
                [-1.0, -0.25],
                [1.0, 0.25],
                [2.0, 0.5],
                [3.0, 1.0],
            ]
        )
        training_labels = np.array([0, 0, 0, 1, 1, 1])

        baseline = fit_logistic_regression_baseline(
            training_features,
            training_labels,
        )
        scores = baseline.fraud_scores(training_features)

        self.assertGreater(
            scores[training_labels == 1].min(),
            scores[training_labels == 0].max(),
        )

    def test_isolation_forest_scores_anomalous_rows_higher(self):
        random = np.random.default_rng(42)
        training_features = random.normal(0.0, 0.2, size=(200, 2))
        normal_features = np.array([[0.0, 0.0], [0.1, -0.1]])
        anomalous_features = np.array([[8.0, 8.0], [-8.0, -8.0]])

        baseline = fit_isolation_forest_baseline(training_features)
        normal_scores = baseline.fraud_scores(normal_features)
        anomalous_scores = baseline.fraud_scores(anomalous_features)

        self.assertGreater(anomalous_scores.min(), normal_scores.max())

    def test_evaluation_freezes_validation_selected_threshold_for_test(self):
        training_features = np.array(
            [
                [-3.0],
                [-2.0],
                [-1.0],
                [1.0],
                [2.0],
                [3.0],
            ]
        )
        training_labels = np.array([0, 0, 0, 1, 1, 1])
        validation_features = np.array([[-2.0], [-1.0], [1.0], [2.0]])
        validation_labels = np.array([0, 0, 1, 1])
        test_features = np.array([[-1.5], [-0.5], [0.5], [1.5]])
        test_labels = np.array([0, 0, 1, 1])
        baseline = fit_logistic_regression_baseline(
            training_features,
            training_labels,
        )

        evaluation = evaluate_baseline(
            baseline,
            validation_features,
            validation_labels,
            test_features,
            test_labels,
            recall_target=1.0,
        )

        self.assertEqual(
            evaluation.threshold_selection.threshold,
            evaluation.test_report.threshold,
        )
        self.assertEqual(1.0, evaluation.threshold_selection.validation_report.recall)
        self.assertEqual(0.5, evaluation.test_report.recall)


if __name__ == "__main__":
    unittest.main()
