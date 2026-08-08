import unittest
from unittest.mock import patch

import numpy as np

from fraud_detection.catboost_model import (
    DEFAULT_CATBOOST_DEPTH,
    DEFAULT_CATBOOST_ITERATIONS,
    DEFAULT_CATBOOST_LEARNING_RATE,
    DEFAULT_EARLY_STOPPING_ROUNDS,
    evaluate_catboost_model,
    fit_catboost_model,
)


class CatBoostModelTests(unittest.TestCase):
    @patch("fraud_detection.catboost_model.CatBoostClassifier")
    def test_fit_uses_train_only_imbalance_and_validation_early_stopping(
        self,
        classifier_type,
    ):
        training_features = np.arange(8, dtype=float).reshape(-1, 1)
        training_labels = np.array([0, 0, 0, 0, 0, 0, 1, 1])
        validation_features = np.arange(4, dtype=float).reshape(-1, 1)
        validation_labels = np.array([0, 1, 1, 1])

        model = fit_catboost_model(
            training_features,
            training_labels,
            validation_features,
            validation_labels,
        )

        classifier_type.assert_called_once_with(
            iterations=DEFAULT_CATBOOST_ITERATIONS,
            depth=DEFAULT_CATBOOST_DEPTH,
            learning_rate=DEFAULT_CATBOOST_LEARNING_RATE,
            random_state=42,
            scale_pos_weight=3.0,
            allow_writing_files=False,
            verbose=False,
        )
        self.assertIs(model.estimator, classifier_type.return_value)
        fit_args, fit_kwargs = classifier_type.return_value.fit.call_args
        self.assertIs(fit_args[0], training_features)
        self.assertIs(fit_args[1], training_labels)
        self.assertIs(fit_kwargs["eval_set"][0], validation_features)
        self.assertIs(fit_kwargs["eval_set"][1], validation_labels)
        self.assertEqual(
            DEFAULT_EARLY_STOPPING_ROUNDS,
            fit_kwargs["early_stopping_rounds"],
        )

    def test_fraud_probabilities_rank_fraud_like_rows_higher(self):
        training_features = np.array(
            [
                [-4.0, -2.0],
                [-3.5, -1.5],
                [-3.0, -1.0],
                [-2.5, -0.5],
                [-2.0, -0.25],
                [-1.5, -0.1],
                [1.5, 0.1],
                [2.0, 0.25],
                [2.5, 0.5],
                [3.0, 1.0],
            ]
        )
        training_labels = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
        validation_features = np.array(
            [[-3.0, -1.0], [-2.0, -0.25], [2.0, 0.25], [3.0, 1.0]]
        )
        validation_labels = np.array([0, 0, 1, 1])

        model = fit_catboost_model(
            training_features,
            training_labels,
            validation_features,
            validation_labels,
        )
        scores = model.fraud_scores(validation_features)

        self.assertGreater(
            scores[validation_labels == 1].min(),
            scores[validation_labels == 0].max(),
        )

    def test_evaluation_freezes_validation_threshold_for_test(self):
        training_features = np.array(
            [[-4.0], [-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0], [4.0]]
        )
        training_labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        validation_features = np.array([[-3.0], [-2.0], [2.0], [3.0]])
        validation_labels = np.array([0, 0, 1, 1])
        test_features = np.array([[-2.5], [-1.5], [1.5], [2.5]])
        test_labels = np.array([0, 0, 1, 1])
        model = fit_catboost_model(
            training_features,
            training_labels,
            validation_features,
            validation_labels,
        )

        evaluation = evaluate_catboost_model(
            model,
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


if __name__ == "__main__":
    unittest.main()
