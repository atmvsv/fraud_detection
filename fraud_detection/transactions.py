"""Load and validate source transaction data before benchmark execution."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = ("Time", *(f"V{index}" for index in range(1, 29)), "Amount")
TARGET_COLUMN = "Class"
REQUIRED_COLUMNS = (*FEATURE_COLUMNS, TARGET_COLUMN)

DEFAULT_TEST_SIZE = 0.2
DEFAULT_CV_SPLITS = 5
DEFAULT_RANDOM_STATE = 42


class TransactionDataError(ValueError):
    """Raised when source transaction data cannot support the benchmark."""


def load_transactions(
    path: str | Path,
    *,
    test_size: float = DEFAULT_TEST_SIZE,
    cv_splits: int = DEFAULT_CV_SPLITS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    """Load a transaction CSV and fail early when its schema is invalid."""
    source = Path(path)
    try:
        transactions = pd.read_csv(source)
    except (OSError, pd.errors.ParserError) as error:
        raise TransactionDataError(
            f"Could not load transaction data from {source}: {error}"
        ) from error

    validate_transactions(
        transactions,
        test_size=test_size,
        cv_splits=cv_splits,
        random_state=random_state,
    )
    return transactions


def validate_transactions(
    transactions: pd.DataFrame,
    *,
    test_size: float = DEFAULT_TEST_SIZE,
    cv_splits: int = DEFAULT_CV_SPLITS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> None:
    """Validate the source schema and current stratified split requirements."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if cv_splits < 2:
        raise ValueError("cv_splits must be at least 2.")

    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in transactions.columns
    ]
    if missing_columns:
        raise TransactionDataError(
            "Missing required transaction fields: " + ", ".join(missing_columns)
        )

    required = transactions.loc[:, REQUIRED_COLUMNS]
    columns_with_missing_values = [
        column for column in REQUIRED_COLUMNS if required[column].isna().any()
    ]
    if columns_with_missing_values:
        raise TransactionDataError(
            "Missing numeric values in fields: "
            + ", ".join(columns_with_missing_values)
        )

    numeric_features = required.loc[:, FEATURE_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )
    malformed_features = [
        column for column in FEATURE_COLUMNS if numeric_features[column].isna().any()
    ]
    if malformed_features:
        raise TransactionDataError(
            "Malformed numeric feature values in fields: "
            + ", ".join(malformed_features)
        )
    if not np.isfinite(numeric_features.to_numpy(dtype=float)).all():
        raise TransactionDataError("Feature values must be finite numbers.")

    numeric_labels = pd.to_numeric(required[TARGET_COLUMN], errors="coerce")
    unexpected_labels = sorted(
        {
            str(value)
            for value, numeric_value in zip(
                required[TARGET_COLUMN].tolist(), numeric_labels.tolist()
            )
            if pd.isna(numeric_value) or numeric_value not in (0, 1)
        }
    )
    if unexpected_labels:
        raise TransactionDataError(
            "Unexpected fraud labels in Class; expected only 0 and 1, found: "
            + ", ".join(unexpected_labels)
        )

    labels = numeric_labels.astype(int)
    _validate_stratified_benchmark_support(
        labels,
        test_size=test_size,
        cv_splits=cv_splits,
        random_state=random_state,
    )


def _validate_stratified_benchmark_support(
    labels: pd.Series,
    *,
    test_size: float,
    cv_splits: int,
    random_state: int,
) -> None:
    indices = np.arange(len(labels))
    try:
        train_indices, test_indices = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=labels,
        )
    except ValueError as error:
        raise TransactionDataError(
            "Transaction labels cannot support the stratified train/test split: "
            f"{error}"
        ) from error

    train_labels = labels.iloc[train_indices]
    test_labels = labels.iloc[test_indices]
    train_fraud_count = int((train_labels == 1).sum())
    test_fraud_count = int((test_labels == 1).sum())

    if test_fraud_count < 1 or train_fraud_count < cv_splits:
        total_fraud_count = int((labels == 1).sum())
        raise TransactionDataError(
            "Not enough fraudulent transactions for the current benchmark: "
            f"found {total_fraud_count}, but the {test_size:.0%} stratified holdout "
            f"must contain fraud and the training partition must support "
            f"{cv_splits}-fold stratified cross-validation."
        )
