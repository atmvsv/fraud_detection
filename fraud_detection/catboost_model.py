"""CatBoost benchmark model with leakage-safe fitting boundaries."""

from dataclasses import dataclass

import numpy as np
from catboost import CatBoostClassifier

from .baselines import BaselineEvaluation, evaluate_baseline

DEFAULT_RANDOM_STATE = 42
DEFAULT_CATBOOST_ITERATIONS = 2_000
DEFAULT_CATBOOST_DEPTH = 6
DEFAULT_CATBOOST_LEARNING_RATE = 0.05
DEFAULT_EARLY_STOPPING_ROUNDS = 50


@dataclass(frozen=True)
class CatBoostFraudModel:
    """A fitted CatBoost model that ranks more suspicious transactions higher."""

    estimator: CatBoostClassifier

    def fraud_scores(self, features) -> np.ndarray:
        """Return the fitted probability of fraud class ``1``."""
        fraud_class_indices = np.flatnonzero(np.asarray(self.estimator.classes_) == 1)
        if fraud_class_indices.size != 1:
            raise RuntimeError("CatBoost must contain fraud class 1.")
        probabilities = self.estimator.predict_proba(features)
        return np.asarray(probabilities[:, fraud_class_indices[0]], dtype=float)


def fit_catboost_model(
    training_features,
    training_labels,
    validation_features,
    validation_labels,
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> CatBoostFraudModel:
    """Fit CatBoost with train-only imbalance state and validation early stopping."""
    scale_pos_weight = _training_scale_pos_weight(training_labels)
    estimator = CatBoostClassifier(
        iterations=DEFAULT_CATBOOST_ITERATIONS,
        depth=DEFAULT_CATBOOST_DEPTH,
        learning_rate=DEFAULT_CATBOOST_LEARNING_RATE,
        random_state=random_state,
        scale_pos_weight=scale_pos_weight,
        allow_writing_files=False,
        verbose=False,
    )
    estimator.fit(
        training_features,
        training_labels,
        eval_set=(validation_features, validation_labels),
        early_stopping_rounds=DEFAULT_EARLY_STOPPING_ROUNDS,
        verbose=False,
    )
    return CatBoostFraudModel(estimator=estimator)


def evaluate_catboost_model(
    model: CatBoostFraudModel,
    validation_features,
    validation_labels,
    test_features,
    test_labels,
    *,
    recall_target: float,
) -> BaselineEvaluation:
    """Apply the shared validation-threshold and frozen test-report protocol."""
    return evaluate_baseline(
        model,
        validation_features,
        validation_labels,
        test_features,
        test_labels,
        recall_target=recall_target,
    )


def _training_scale_pos_weight(training_labels) -> float:
    labels = np.asarray(training_labels)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("training_labels must be a non-empty one-dimensional array.")
    try:
        numeric_labels = labels.astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError("training_labels must contain only 0 and 1.") from error
    if not np.isfinite(numeric_labels).all() or not np.isin(
        numeric_labels, (0.0, 1.0)
    ).all():
        raise ValueError("training_labels must contain only 0 and 1.")

    negative_count = int(np.count_nonzero(numeric_labels == 0.0))
    positive_count = int(np.count_nonzero(numeric_labels == 1.0))
    if negative_count == 0 or positive_count == 0:
        raise ValueError("training_labels must contain both non-fraud and fraud examples.")
    return negative_count / positive_count
