from src.utils.singleton import Singleton

from src.utils.globals import Globals
from src.services.data_service import DataService
from src.services.config_service import ConfigService

from src.utils.logger import Logger

from .feature_builder import FeatureBuilder
from .normalization import rolling_zscore
from .labeling import macd_labeling

import pandas
import numpy

class Preprocessor(metaclass=Singleton):
    def __init__(self, config_service: ConfigService, data_service: DataService):
        self.config_service = config_service
        self.data_service = data_service
    
    def timeseries_split(self, input_array: numpy.ndarray) -> tuple[numpy.ndarray]:
        """
        Returns:
            Tuple(numpy.ndarray): Returns train, valdiation and test datasets at indicies [0, 1, 2] 
        """ 

        num_samples = input_array.shape[0]
        split_sizes = [
            int(num_samples * self.config_service.training_size), 
            int(num_samples * (self.config_service.training_size + self.config_service.validation_size))
        ]
        
        train, validation, test = numpy.split(input_array, split_sizes)
        return train, validation, test
    
    def mini_batch(self, input_array: numpy.ndarray):
        return numpy.array_split(input_array, self.config_service.logistic_regression_num_batch)

    def prepare_features(self):
        # Read OHLCV data from the recordings
        candles_df = self.data_service.read_candles(
            start_ts=self.config_service.start_ts, 
            end_ts=self.config_service.end_ts, 
            symbol=self.config_service.symbol, 
            interval=self.config_service.interval
        )
        
        # Label candles
        candles_df["label"] = macd_labeling(
            input_df=candles_df, 
            fast_period=self.config_service.macd_fast, 
            slow_period=self.config_service.macd_slow, 
            signal_period=self.config_service.macd_signal
        )
        
        # Add features
        feature_builder = FeatureBuilder(input_df=candles_df, config_service=self.config_service)
        if self.config_service.is_confidential == True:
            # If is_confidential is True, read features data from a .csv file 
            feature_builder.read_features()
        else:
            # Else, produce the features and save them to a .csv file 
            feature_builder.build_all()
        candles_df = feature_builder.candles_df 
        
        # Normalize the data
        if self.config_service.categorize_features == False:
            # No need for normalization if the features are categorized
            feature_columns = candles_df.columns[~(candles_df.columns.isin(["timestamp", "label"]))]
            for column in feature_columns:
                candles_df[column] = feature_builder.normalize_feature(candles_df[column])
            candles_df.dropna(inplace=True)
            candles_df.reset_index(drop=True, inplace=True)
        
        # Drop unlabeled candles
        candles_df = candles_df.loc[candles_df.label.isin([1,2])]
        candles_df.loc[:, "label"] = candles_df["label"] - 1         #{0: Correct, 1. Incorrect} 
        candles_df.reset_index(drop=True, inplace=True)
                
        return candles_df