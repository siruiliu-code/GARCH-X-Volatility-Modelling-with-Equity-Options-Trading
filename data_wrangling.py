import pandas as pd
import numpy as np

def rolling_z_score(series: pd.Series, window: int, min_periods: int = 1) -> pd.Series:
    """
    Calculate the rolling z-score of a series.
    """
    return (series - series.rolling(window, min_periods=min_periods).mean()) / series.rolling(window, min_periods=min_periods).std()

