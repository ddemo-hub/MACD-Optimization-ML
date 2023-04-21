from src.utils.singleton import Singleton

from src.utils.globals import Globals
from src.services.data_service import DataService
from src.services.config_service import ConfigService

from src.utils.logger import Logger

from .feature_builder import FeatureBuilder
from .normalization import rolling_zscore
from .labeling import macd_labeling

import pandas

class Preprocessor(metaclass=Singleton):
    def __init__(self, config_service: ConfigService, data_service: DataService):
        self.config_service = config_service
        self.data_service = data_service
    
    def normalize_data(self, input_df: pandas.DataFrame):
        feature_columns = input_df.columns[~(input_df.columns.isin(["timestamp", "label"]))]
        
        # Zscore
        zscore_columns = feature_columns[feature_columns.str.get(-1) != "0"]
        for column in zscore_columns:
            input_df[column] = rolling_zscore(
                column=input_df[column], 
                window=self.config_service.standardization_window, 
                limit=self.config_service.standardization_limit
            )
        
        # Ratio
        ratio_columns = feature_columns[feature_columns.str.get(-2) != "0"]
        input_df[ratio_columns] = input_df[ratio_columns] / input_df[ratio_columns].shift(1)
        
        input_df.dropna(inplace=True) 
        
        return input_df
    
    def prepare_model_inputs(
        self,
        start_ts: int, 
        end_ts: int, 
        symbol: str, 
        interval: str, 
        macd_fast: int, 
        macd_slow: int, 
        macd_signal: int
    ):
        # Read OHLCV data from the recordings
        candles_df = self.data_service.read_candles(start_ts=start_ts, end_ts=end_ts, symbol=symbol, interval=interval)
        
        # Label candles
        candles_df["label"] = macd_labeling(input_df=candles_df, fast_period=macd_fast, slow_period=macd_slow, signal_period=macd_signal)
        
        # Add features
        # builder = FeatureBuilder(input_df=candles_df)
        # if self.config_service.is_confidential == True:
        #     # If is_confidential is True, read features data from a .csv file 
        #     builder.read_features(f"{Globals.klines_path}/{symbol}/{interval}/features.csv")
        # else:
        #     builder.build_all()
        
        # Normalize the data
        candles_df = self.normalize_data(input_df=candles_df)
        
        # Drop unlabeled candles
        candles_df = candles_df.loc[candles_df.label.isin([1,2])]
        candles_df.reset_index(drop=True, inplace=True)
        candles_df.loc[:, "label"] = candles_df["label"] -1    #{0: Correct, 1. Incorrecy} 
        
        return candles_df