import unittest

from fraud_detection.evaluation import (
    report_fraud_scores,
    select_review_threshold,
)


class FraudScoreEvaluationTests(unittest.TestCase):
    def test_selects_highest_cutoff_that_satisfies_recall_target(self):
        labels = [1, 0, 1, 0, 1, 0, 1, 0]
        fraud_scores = [0.99, 0.95, 0.90, 0.80, 0.70, 0.65, 0.60, 0.10]

        selection = select_review_threshold(
            labels,
            fraud_scores,
            recall_target=0.75,
        )

        self.assertEqual(0.70, selection.threshold)
        self.assertEqual(0.75, selection.validation_report.recall)
        self.assertEqual(5, selection.validation_report.review_queue_size)
        self.assertEqual(2, selection.validation_report.false_positive_count)
        self.assertEqual(1, selection.validation_report.false_negative_count)

    def test_equal_recall_plateau_prefers_smaller_queue_and_lower_fpr(self):
        labels = [1, 0, 0, 1]
        fraud_scores = [0.90, 0.85, 0.80, 0.60]

        selection = select_review_threshold(
            labels,
            fraud_scores,
            recall_target=0.50,
        )

        self.assertEqual(0.90, selection.threshold)
        self.assertEqual(1, selection.validation_report.review_queue_size)
        self.assertEqual(0, selection.validation_report.false_positive_count)
        self.assertEqual(0.0, selection.validation_report.false_positive_rate)

    def test_applies_validation_selected_threshold_unchanged_to_test_scores(self):
        selection = select_review_threshold(
            [1, 0, 1, 0],
            [0.90, 0.80, 0.60, 0.20],
            recall_target=1.0,
        )

        test_report = report_fraud_scores(
            [1, 0, 1, 0],
            [0.70, 0.65, 0.50, 0.10],
            threshold=selection.threshold,
        )

        self.assertEqual(0.60, selection.threshold)
        self.assertEqual(selection.threshold, test_report.threshold)
        self.assertEqual(0.50, test_report.recall)
        self.assertEqual(0.50, test_report.precision)
        self.assertEqual(1, test_report.false_positive_count)
        self.assertEqual(1, test_report.false_negative_count)
        self.assertEqual(0.50, test_report.false_positive_rate)
        self.assertEqual(2, test_report.review_queue_size)

    def test_reports_primary_and_secondary_ranking_metrics(self):
        report = report_fraud_scores(
            [1, 0, 1, 0],
            [0.90, 0.80, 0.60, 0.20],
            threshold=0.60,
        )

        self.assertAlmostEqual(5 / 6, report.pr_auc)
        self.assertAlmostEqual(0.75, report.roc_auc)

    def test_rejects_invalid_evaluation_inputs(self):
        invalid_cases = [
            ([1], [0.90], 0.90),
            ([1, 0], [0.90], 0.90),
            ([1, 2], [0.90, 0.10], 0.90),
            ([1, 0], [0.90, float("nan")], 0.90),
        ]

        for labels, scores, recall_target in invalid_cases:
            with self.subTest(labels=labels, scores=scores):
                with self.assertRaises(ValueError):
                    select_review_threshold(
                        labels,
                        scores,
                        recall_target=recall_target,
                    )

        for recall_target in (0.0, 1.01, float("nan")):
            with self.subTest(recall_target=recall_target):
                with self.assertRaises(ValueError):
                    select_review_threshold(
                        [1, 0],
                        [0.90, 0.10],
                        recall_target=recall_target,
                    )


if __name__ == "__main__":
    unittest.main()
