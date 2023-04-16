import pandas

def macd_labeling(input_df: pandas.DataFrame, fast_period: int, slow_period: int, signal_period: int) -> pandas.Series:
    """Label Macd Histogram's buy signals as correct & incorrect

    Args:
        input_df (pandas.DataFrame): A dataframe that contains ohlcv data. The column that holds the close price data 
                                     must be named as "close" and the rows must be in ascending order by timestamp

        fast_period (int): macd fast period
        slow_period (int): macd slow period
        signal_period (int): macd signal period

    Returns:
        pandas.Series: A pandas series of length same as the length of the input candles_df,
                       Correct Macd Histogram buy signals are labeled as 1
                       Incorrect Macd Histogram buy signals are labeled as 2
                       Everything else is labeled as 0
    """
    def macd_histogram(candles_df: pandas.DataFrame) -> pandas.Series:
        fast_ema = candles_df["close"].ewm(span=fast_period, adjust=False, min_periods=fast_period).mean()
        slow_ema = candles_df["close"].ewm(span=slow_period, adjust=False, min_periods=slow_period).mean()

        macd = fast_ema - slow_ema
        macd_signal = macd.ewm(span=signal_period, adjust=False, min_periods=signal_period).mean()
        
        return macd - macd_signal
    
    # Copy the input dataframe to avoid modifying it
    candles_df = input_df.copy()
        
    # Calculate Macd Histogram
    macd_hist = macd_histogram(candles_df)
    candles_df = candles_df.assign(macd_hist=macd_hist)
    
    # Loc buy & sell points
    buy_sell_signals = macd_hist[macd_hist * macd_hist.shift(1) < 0]    
    
    # If first signal is a sell signal, make it gone
    buy_sell_signals = buy_sell_signals if buy_sell_signals.iloc[0] > 0 else buy_sell_signals.iloc[1:]
    
    # Buy sell points with ohlc data
    bsp_with_ohlc = candles_df.loc[buy_sell_signals.index]
    
    # Correct & Incorrect signals's indicies
    correct_indicies = bsp_with_ohlc.loc[bsp_with_ohlc.shift(-1).close > bsp_with_ohlc.close].loc[bsp_with_ohlc.macd_hist > 0].index
    incorrect_indicies = bsp_with_ohlc.loc[bsp_with_ohlc.shift(-1).close <= bsp_with_ohlc.close].loc[bsp_with_ohlc.macd_hist > 0].index
    
    # Label the input
    candles_df["label"] = 0
    candles_df.loc[correct_indicies, "label"] = 1
    candles_df.loc[incorrect_indicies, "label"] = 2
    
    return candles_df["label"]