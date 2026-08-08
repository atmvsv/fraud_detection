"""Reproducible end-to-end fraud benchmark orchestration."""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from .baselines import (
    BaselineEvaluation,
    FraudScoreBaseline,
    fit_isolation_forest_baseline,
    fit_logistic_regression_baseline,
)
from .catboost_model import fit_catboost_model
from .evaluation import (
    ReviewThresholdSelection,
    report_fraud_scores,
    select_review_threshold,
)
from .preprocessing import fit_transaction_preprocessor
from .splits import create_stratified_splits
from .transactions import DEFAULT_RANDOM_STATE, TARGET_COLUMN, load_transactions

DEFAULT_RECALL_TARGET = 0.90


@dataclass(frozen=True)
class BenchmarkResult:
    """Split metadata and final reports for every supported benchmark model."""

    random_state: int
    recall_target: float
    train_size: int
    validation_size: int
    test_size: int
    logistic_regression: BaselineEvaluation
    isolation_forest: BaselineEvaluation
    catboost: BaselineEvaluation


def run_benchmark(
    source_path: str | Path,
    *,
    recall_target: float = DEFAULT_RECALL_TARGET,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> BenchmarkResult:
    """Run the benchmark from source transactions through final test reporting."""
    transactions = load_transactions(source_path, random_state=random_state)
    splits = create_stratified_splits(transactions, random_state=random_state)

    preprocessor = fit_transaction_preprocessor(splits.train)
    training_features = preprocessor.transform(splits.train)
    validation_features = preprocessor.transform(splits.validation)
    training_labels = splits.train[TARGET_COLUMN].to_numpy(dtype=int)
    validation_labels = splits.validation[TARGET_COLUMN].to_numpy(dtype=int)

    logistic_regression = fit_logistic_regression_baseline(
        training_features,
        training_labels,
        random_state=random_state,
    )
    isolation_forest = fit_isolation_forest_baseline(
        training_features,
        random_state=random_state,
    )
    catboost = fit_catboost_model(
        training_features,
        training_labels,
        validation_features,
        validation_labels,
        random_state=random_state,
    )

    logistic_regression_threshold = _select_threshold(
        logistic_regression,
        validation_features,
        validation_labels,
        recall_target=recall_target,
    )
    isolation_forest_threshold = _select_threshold(
        isolation_forest,
        validation_features,
        validation_labels,
        recall_target=recall_target,
    )
    catboost_threshold = _select_threshold(
        catboost,
        validation_features,
        validation_labels,
        recall_target=recall_target,
    )

    test_features = preprocessor.transform(splits.test)
    test_labels = splits.test[TARGET_COLUMN].to_numpy(dtype=int)

    return BenchmarkResult(
        random_state=random_state,
        recall_target=recall_target,
        train_size=len(splits.train),
        validation_size=len(splits.validation),
        test_size=len(splits.test),
        logistic_regression=_report_test(
            logistic_regression,
            logistic_regression_threshold,
            test_features,
            test_labels,
        ),
        isolation_forest=_report_test(
            isolation_forest,
            isolation_forest_threshold,
            test_features,
            test_labels,
        ),
        catboost=_report_test(
            catboost,
            catboost_threshold,
            test_features,
            test_labels,
        ),
    )


def _select_threshold(
    model: FraudScoreBaseline,
    validation_features,
    validation_labels,
    *,
    recall_target: float,
) -> ReviewThresholdSelection:
    validation_scores = model.fraud_scores(validation_features)
    return select_review_threshold(
        validation_labels,
        validation_scores,
        recall_target=recall_target,
    )


def _report_test(
    model: FraudScoreBaseline,
    threshold_selection: ReviewThresholdSelection,
    test_features,
    test_labels,
) -> BaselineEvaluation:
    test_scores = model.fraud_scores(test_features)
    test_report = report_fraud_scores(
        test_labels,
        test_scores,
        threshold=threshold_selection.threshold,
    )
    return BaselineEvaluation(
        threshold_selection=threshold_selection,
        test_report=test_report,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark command and write its structured result to stdout."""
    parser = argparse.ArgumentParser(
        description="Run the reproducible fraud-detection benchmark."
    )
    parser.add_argument(
        "source_path",
        type=Path,
        help="Path to the source transaction CSV.",
    )
    parser.add_argument(
        "--recall-target",
        type=float,
        default=DEFAULT_RECALL_TARGET,
        help="Validation recall target used to select each review threshold.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Seed used by deterministic splits and supported models.",
    )
    arguments = parser.parse_args(argv)

    result = run_benchmark(
        arguments.source_path,
        recall_target=arguments.recall_target,
        random_state=arguments.random_state,
    )
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
