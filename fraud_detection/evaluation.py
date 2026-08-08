"""Validation-selected review thresholds and fraud-score reporting."""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass(frozen=True)
class FraudScoreReport:
    """Ranking and operational metrics at one fixed review threshold."""

    threshold: float
    pr_auc: float
    roc_auc: float
    recall: float
    precision: float
    false_positive_count: int
    false_negative_count: int
    false_positive_rate: float
    review_queue_size: int


@dataclass(frozen=True)
class ReviewThresholdSelection:
    """A review threshold selected from validation fraud scores."""

    recall_target: float
    threshold: float
    validation_report: FraudScoreReport


def select_review_threshold(
    labels,
    fraud_scores,
    *,
    recall_target: float,
) -> ReviewThresholdSelection:
    """Select the highest validation score cutoff meeting ``recall_target``.

    Candidate cutoffs are observed validation fraud scores. Choosing the highest
    valid cutoff also prefers the smallest review queue and lowest false positive
    rate when recall remains unchanged across adjacent cutoffs.
    """
    if isinstance(recall_target, bool) or not np.isfinite(recall_target):
        raise ValueError("recall_target must be a finite number between 0 and 1.")
    recall_target = float(recall_target)
    if not 0 < recall_target <= 1:
        raise ValueError("recall_target must be greater than 0 and at most 1.")

    label_values, score_values = _validated_inputs(labels, fraud_scores)
    positive_count = int(label_values.sum())

    descending_order = np.argsort(-score_values, kind="stable")
    sorted_scores = score_values[descending_order]
    sorted_labels = label_values[descending_order]
    cumulative_positives = np.cumsum(sorted_labels)
    score_group_ends = np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    valid_group_ends = np.flatnonzero(
        score_group_ends
        & (cumulative_positives / positive_count >= recall_target)
    )

    if valid_group_ends.size == 0:
        raise ValueError("No validation threshold satisfies recall_target.")
    selected_threshold = float(sorted_scores[valid_group_ends[0]])

    report = report_fraud_scores(
        label_values,
        score_values,
        threshold=selected_threshold,
    )
    return ReviewThresholdSelection(
        recall_target=recall_target,
        threshold=selected_threshold,
        validation_report=report,
    )


def report_fraud_scores(
    labels,
    fraud_scores,
    *,
    threshold: float,
) -> FraudScoreReport:
    """Report ranking and operational metrics at an unchanged threshold."""
    if isinstance(threshold, bool) or not np.isfinite(threshold):
        raise ValueError("threshold must be a finite number.")
    threshold = float(threshold)

    label_values, score_values = _validated_inputs(labels, fraud_scores)
    reviewed = score_values >= threshold

    true_positives = int(label_values[reviewed].sum())
    false_positives = int(reviewed.sum() - true_positives)
    false_negatives = int(label_values[~reviewed].sum())
    true_negatives = int((~reviewed).sum() - false_negatives)
    positive_count = true_positives + false_negatives
    negative_count = true_negatives + false_positives
    queue_size = int(reviewed.sum())

    return FraudScoreReport(
        threshold=threshold,
        pr_auc=float(average_precision_score(label_values, score_values)),
        roc_auc=float(roc_auc_score(label_values, score_values)),
        recall=true_positives / positive_count,
        precision=true_positives / queue_size if queue_size else 0.0,
        false_positive_count=false_positives,
        false_negative_count=false_negatives,
        false_positive_rate=false_positives / negative_count,
        review_queue_size=queue_size,
    )


def _validated_inputs(labels, fraud_scores) -> tuple[np.ndarray, np.ndarray]:
    label_values = np.asarray(labels)
    score_values = np.asarray(fraud_scores)

    if label_values.ndim != 1 or score_values.ndim != 1:
        raise ValueError("labels and fraud_scores must be one-dimensional.")
    if len(label_values) == 0:
        raise ValueError("labels and fraud_scores must not be empty.")
    if len(label_values) != len(score_values):
        raise ValueError("labels and fraud_scores must have the same length.")

    try:
        numeric_labels = label_values.astype(float)
        numeric_scores = score_values.astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError("labels and fraud_scores must be numeric.") from error

    if not np.isfinite(numeric_labels).all():
        raise ValueError("labels must contain only finite binary values.")
    if not np.isin(numeric_labels, (0.0, 1.0)).all():
        raise ValueError("labels must contain only 0 and 1.")
    if not np.isfinite(numeric_scores).all():
        raise ValueError("fraud_scores must contain only finite values.")
    if np.unique(numeric_labels).size != 2:
        raise ValueError("labels must contain both non-fraud and fraud examples.")

    return numeric_labels.astype(int), numeric_scores
