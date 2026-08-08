"""Baseline models with a shared higher-is-more-fraudulent score direction."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression

from .evaluation import (
    FraudScoreReport,
    ReviewThresholdSelection,
    report_fraud_scores,
    select_review_threshold,
)

DEFAULT_RANDOM_STATE = 42
DEFAULT_LOGISTIC_MAX_ITER = 1_000
DEFAULT_ISOLATION_ESTIMATORS = 100
DEFAULT_ISOLATION_CONTAMINATION = 0.01


class FraudScoreBaseline(Protocol):
    """A fitted model that ranks more suspicious transactions higher."""

    def fraud_scores(self, features) -> np.ndarray:
        """Return one higher-is-more-fraudulent score per transaction."""


@dataclass(frozen=True)
class LogisticRegressionBaseline:
    """A fitted supervised class-weighted Logistic Regression baseline."""

    estimator: LogisticRegression

    def fraud_scores(self, features) -> np.ndarray:
        """Return the fitted probability of the fraud class."""
        fraud_class_indices = np.flatnonzero(self.estimator.classes_ == 1)
        if fraud_class_indices.size != 1:
            raise RuntimeError("Logistic Regression must contain fraud class 1.")
        probabilities = self.estimator.predict_proba(features)
        return np.asarray(probabilities[:, fraud_class_indices[0]], dtype=float)


@dataclass(frozen=True)
class IsolationForestBaseline:
    """A fitted unsupervised Isolation Forest baseline."""

    estimator: IsolationForest

    def fraud_scores(self, features) -> np.ndarray:
        """Negate normality scores so more anomalous transactions rank higher."""
        return -np.asarray(self.estimator.score_samples(features), dtype=float)


@dataclass(frozen=True)
class BaselineEvaluation:
    """Validation threshold selection and unchanged test reporting."""

    threshold_selection: ReviewThresholdSelection
    test_report: FraudScoreReport


def fit_logistic_regression_baseline(
    training_features,
    training_labels,
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> LogisticRegressionBaseline:
    """Fit the supervised baseline using training features and labels only."""
    estimator = LogisticRegression(
        class_weight="balanced",
        max_iter=DEFAULT_LOGISTIC_MAX_ITER,
        random_state=random_state,
    )
    estimator.fit(training_features, training_labels)
    return LogisticRegressionBaseline(estimator=estimator)


def fit_isolation_forest_baseline(
    training_features,
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> IsolationForestBaseline:
    """Fit the unsupervised baseline using training features without labels."""
    estimator = IsolationForest(
        n_estimators=DEFAULT_ISOLATION_ESTIMATORS,
        contamination=DEFAULT_ISOLATION_CONTAMINATION,
        random_state=random_state,
        n_jobs=-1,
    )
    estimator.fit(training_features)
    return IsolationForestBaseline(estimator=estimator)


def evaluate_baseline(
    baseline: FraudScoreBaseline,
    validation_features,
    validation_labels,
    test_features,
    test_labels,
    *,
    recall_target: float,
) -> BaselineEvaluation:
    """Select on validation scores and report test scores at that threshold."""
    validation_scores = baseline.fraud_scores(validation_features)
    threshold_selection = select_review_threshold(
        validation_labels,
        validation_scores,
        recall_target=recall_target,
    )
    test_scores = baseline.fraud_scores(test_features)
    test_report = report_fraud_scores(
        test_labels,
        test_scores,
        threshold=threshold_selection.threshold,
    )
    return BaselineEvaluation(
        threshold_selection=threshold_selection,
        test_report=test_report,
    )
