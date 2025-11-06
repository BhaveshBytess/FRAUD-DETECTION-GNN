"""Utility helpers for robust timestamp column detection and normalization.

This avoids KeyError when different versions of the Elliptic++ dataset or
user-prepared DataFrames use inconsistent timestamp column names such as:
  'Time step', 'time_step', 'timestamp', 'time', 'timestep'.

Primary entrypoint: normalize_timestamp(df) which returns a copy of the input
DataFrame with a guaranteed 'timestamp' numeric column (int64) sorted ascending.
"""
from __future__ import annotations
from typing import Iterable, Tuple
import pandas as pd

CANDIDATE_TS_COLUMNS = [
    'Time step',       # Original Elliptic++ export (capital T, space)
    'time_step',       # Snake case variant
    'timestamp',       # Generic
    'time',            # Generic short
    'timestep',        # Concatenated
]

def find_timestamp_column(df: pd.DataFrame, candidates: Iterable[str] = None) -> str:
    """Return the first matching timestamp-like column name.

    Raises:
        KeyError: if no candidate column found.
    """
    if candidates is None:
        candidates = CANDIDATE_TS_COLUMNS
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        "No timestamp column found. Looked for: " + ", ".join(candidates) +
        f". Available columns: {list(df.columns)[:25]}..."
    )


def normalize_timestamp(df: pd.DataFrame, drop_na: bool = True) -> Tuple[pd.DataFrame, str]:
    """Return (df_out, original_col) where df_out has a unified numeric 'timestamp'.

    Steps:
      1. Detect timestamp column.
      2. Coerce to numeric (errors='coerce').
      3. Optionally drop rows with NaN timestamps.
      4. Sort by 'timestamp'.
    """
    ts_col = find_timestamp_column(df)
    df_out = df.copy()
    df_out['timestamp'] = pd.to_numeric(df_out[ts_col], errors='coerce')
    if drop_na:
        before = len(df_out)
        df_out = df_out.dropna(subset=['timestamp'])
        after = len(df_out)
        if before != after:
            print(f"[time_utils] Dropped {before-after} rows with invalid timestamp values")
    # Ensure integer type if all finite
    if df_out['timestamp'].notna().all():
        try:
            df_out['timestamp'] = df_out['timestamp'].astype('int64')
        except Exception:
            pass
    df_out = df_out.sort_values('timestamp').reset_index(drop=True)
    return df_out, ts_col

__all__ = [
    'find_timestamp_column',
    'normalize_timestamp',
    'CANDIDATE_TS_COLUMNS'
]
