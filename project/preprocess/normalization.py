import pandas

def rolling_zscore(column: pandas.Series, window: int, limit: int) -> pandas.Series:
    """Apply zscore normalization on the given column by looking at the past values to calculate
       the mean and standard deviation in order to avoid looking to the future.

    Args:
        column (pandas.Series): A pandas series of numeric values
        window (int): Size of the rolling window
        limit (int): Maximum value that the absolute value of zscore can be.
                     Used for dealing with outliers

    Returns:
        pandas.Series: Standardized column 
    """
    rolling_mean = column.rolling(window=window).mean()
    rolling_std = column.rolling(window=window).mean()
    
    rolling_zscores = (column - rolling_mean) / rolling_std
    
    rolling_zscores[rolling_zscores > limit] = limit
    rolling_zscores[rolling_zscores < -limit] = -limit

    return rolling_zscores    


# Remaining normalization methods will be kept confidential