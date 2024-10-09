import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.compat import lzip


def rolling_z_score(series: pd.Series, window: int, min_periods: int = 1) -> pd.Series:
    """
    Calculate the rolling z-score of a series.
    """
    return (series - series.rolling(window, min_periods=min_periods).mean()) / series.rolling(window, min_periods=min_periods).std()

def adf_test(series: pd.Series) -> float:
    """
    Perform Augmented Dickey-Fuller test.
    """
    result = adfuller(series)
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')

    if result[1] > 0.05:
        print("Non-stationary: Consider differencing or other transformations")
        return result[1]
    else:
        print("Stationary: No differencing required")
        return result[1]
        
def goldfeld_quandt_test(series_x: pd.Series, series_y: pd.Series, split=None) -> dict:
    """
    Perform Goldfeld-Quandt test.
    """
    exog_var = sm.add_constant(series_x)
    model = sm.OLS(series_y, exog_var.astype(float), missing='drop').fit()
    result = sm.stats.diagnostic.het_goldfeldquandt(model.resid, model.model.exog, split=split)
    test_results = dict(lzip(['F-statistic', 'p-value'], result))
    
    return test_results

def white_test(series_x: pd.Series, series_y: pd.Series) -> dict:
    """
    Perform White test.
    """
    exog_var = sm.add_constant(series_x)
    model = sm.OLS(series_y, exog_var.astype(float), missing='drop').fit()
    result = sm.stats.diagnostic.het_white(model.resid, model.model.exog)
    test_results = dict(lzip(['LM-statistic', 'LM-test p-value', 'F-statistic', 'F-test p-value'], result))
    
    return test_results