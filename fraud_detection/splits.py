"""Deterministic random stratified splits for the initial benchmark."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .transactions import (
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
    TARGET_COLUMN,
    TransactionDataError,
    validate_transactions,
)

DEFAULT_VALIDATION_SIZE = 0.2


@dataclass(frozen=True)
class TransactionSplits:
    """Train, validation, and test transaction assignments."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def create_stratified_splits(
    transactions: pd.DataFrame,
    *,
    test_size: float = DEFAULT_TEST_SIZE,
    validation_size: float = DEFAULT_VALIDATION_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> TransactionSplits:
    """Create deterministic random stratified train/validation/test splits.

    ``test_size`` is the fraction of all transactions assigned to the test set.
    ``validation_size`` is the fraction of the remaining development rows assigned
    to validation. The defaults therefore produce 64% train, 16% validation, and
    20% test assignments.
    """
    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1.")

    validate_transactions(
        transactions,
        test_size=test_size,
        random_state=random_state,
    )

    labels = transactions[TARGET_COLUMN].astype(int)
    positions = np.arange(len(transactions))

    try:
        development_positions, test_positions = train_test_split(
            positions,
            test_size=test_size,
            random_state=random_state,
            stratify=labels,
        )
        train_positions, validation_positions = train_test_split(
            development_positions,
            test_size=validation_size,
            random_state=random_state,
            stratify=labels.iloc[development_positions],
        )
    except ValueError as error:
        raise TransactionDataError(
            "Transaction labels cannot support the stratified "
            f"train/validation/test split: {error}"
        ) from error

    return TransactionSplits(
        train=transactions.iloc[train_positions].copy(),
        validation=transactions.iloc[validation_positions].copy(),
        test=transactions.iloc[test_positions].copy(),
    )
