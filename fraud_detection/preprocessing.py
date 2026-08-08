"""Leakage-safe preprocessing for benchmark transaction features."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .transactions import FEATURE_COLUMNS

AMOUNT_COLUMN = "Amount"
TIME_COLUMN = "Time"
SCALED_AMOUNT_COLUMN = "scaled_Amount"
MODEL_FEATURE_COLUMNS = tuple(
    column for column in FEATURE_COLUMNS if column not in (TIME_COLUMN, AMOUNT_COLUMN)
)


@dataclass(frozen=True)
class TransactionPreprocessor:
    """Frozen preprocessing state fitted from training transactions only."""

    amount_mean: float
    amount_scale: float

    def transform(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """Transform transaction features without fitting any new state."""
        _require_feature_columns(transactions)

        amount = _finite_amount_values(transactions)
        transformed = transactions.loc[:, MODEL_FEATURE_COLUMNS].copy()
        transformed[SCALED_AMOUNT_COLUMN] = (
            amount - self.amount_mean
        ) / self.amount_scale
        return transformed


def fit_transaction_preprocessor(
    training_transactions: pd.DataFrame,
) -> TransactionPreprocessor:
    """Fit Amount scaling state using training transactions only.

    ``Time`` is intentionally excluded from model features. Validation and test
    transactions must be transformed with the returned frozen state rather than
    fitted independently.
    """
    _require_feature_columns(training_transactions)
    if training_transactions.empty:
        raise ValueError("training_transactions must contain at least one row.")

    amount = _finite_amount_values(training_transactions)
    scaler = StandardScaler().fit(amount.to_numpy().reshape(-1, 1))
    return TransactionPreprocessor(
        amount_mean=float(scaler.mean_[0]),
        amount_scale=float(scaler.scale_[0]),
    )


def _require_feature_columns(transactions: pd.DataFrame) -> None:
    missing_columns = [
        column for column in FEATURE_COLUMNS if column not in transactions.columns
    ]
    if missing_columns:
        raise ValueError(
            "Missing required preprocessing fields: " + ", ".join(missing_columns)
        )


def _finite_amount_values(transactions: pd.DataFrame) -> pd.Series:
    amount = pd.to_numeric(transactions[AMOUNT_COLUMN], errors="coerce")
    if amount.isna().any() or not np.isfinite(amount.to_numpy(dtype=float)).all():
        raise ValueError("Amount values must be finite numbers.")
    return amount.astype(float)
