"""
Polars utilities for faster data processing.
Install with: pip install polars
"""

try:
    import polars as pl

    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    pl = None

import pandas as pd
from typing import List


def merge_stock_data_polars(
    df: pd.DataFrame, new_data: pd.DataFrame, on: str
) -> pd.DataFrame:
    """
    Merge dataframes using Polars for better performance.
    Falls back to pandas if Polars is not available.
    """
    if not POLARS_AVAILABLE:
        return df.merge(new_data, on=on, how="left")

    # Convert to Polars
    pl_df = pl.from_pandas(df)
    pl_new = pl.from_pandas(new_data)

    # Perform merge
    pl_result = pl_df.join(pl_new, on=on, how="left")

    # Convert back to pandas
    return pl_result.to_pandas()


def filter_by_categories_polars(
    df: pd.DataFrame, categories: List[str], column: str = "產業別"
) -> pd.DataFrame:
    """
    Filter dataframe by categories using Polars for better performance.
    """
    if not POLARS_AVAILABLE or not categories:
        return (
            df[df[column].isin(categories)].copy()
            if categories
            else df.iloc[0:0].copy()
        )

    pl_df = pl.from_pandas(df)
    pl_filtered = pl_df.filter(pl.col(column).is_in(categories))
    return pl_filtered.to_pandas()


def calculate_moving_average_polars(values: pd.Series, window: int) -> pd.Series:
    """
    Calculate moving average using Polars for better performance.
    """
    if not POLARS_AVAILABLE or window <= 1:
        return values

    pl_series = pl.Series(values=values)
    pl_ma = pl_series.rolling_mean(window_size=window)
    return pl_ma.to_pandas()
