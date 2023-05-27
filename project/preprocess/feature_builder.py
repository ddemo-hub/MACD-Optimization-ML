from src.services.config_service import ConfigService
from src.utils.globals import Globals

from .normalization import rolling_zscore

import pandas 
import numpy

class FeatureBuilder():
    _must_columns = ["timestamp", "open", "high", "low", "close", "volume"]

    def __init__(self, input_df: pandas.DataFrame, config_service: ConfigService):
        """
        Args:
            input_df (pandas.DataFrame): A dataframe with timestamps and their corresponding OHLCV data
                                         must have columns: ["timestamp", "open", "high", "low", "close", "volume"]
        """
        if not set(self._must_columns).issubset(set(input_df.columns)):
            raise Exception("The input DataFrame must contain the columns [timestamp, open, high, low, close, volume]")

        self.candles_df = input_df
        self.config_service = config_service

        self.features_path = Globals.klines_path.joinpath(config_service.symbol, config_service.interval, "features.csv")
        self.target_features = config_service.target_features 

        self.categorization = config_service.categorize_features

        self._ignore_columns = list(set(self.candles_df.columns) - set(self._must_columns))
    
    
    def normalize_feature(self, column: pandas.Series):
        if column.name[-1] != "0":
            column = rolling_zscore(
                column=column, 
                window=self.config_service.standardization_window, 
                limit=self.config_service.standardization_limit
            )
        
        if column.name[-2] != "0":
            column = column / column.shift(1)

        return column
    
    
    def _categorize_features(self):
        ...
    
    
    def build_all():
        ...
        
    # Features will be kept confidential
    def read_features(self):
        features_df = pandas.read_csv(self.features_path)
        
        self.candles_df = self.candles_df.merge(features_df, on="timestamp")

        if self.categorization == True:
            self._categorize_features()
        
        self.candles_df = self.candles_df[["timestamp"] + self.target_features + self._ignore_columns]
