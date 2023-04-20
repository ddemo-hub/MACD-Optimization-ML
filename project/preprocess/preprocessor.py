from src.utils.singleton import Singleton

from src.utils.globals import Globals
from src.services.data_service import DataService
from src.services.config_service import ConfigService

from src.utils.logger import Logger

from .feature_builder import FeatureBuilder
from .labeling import macd_labeling

import pandas

class Preprocessor(metaclass=Singleton):
    def __init__(self, config_service: ConfigService, data_service: DataService):
        self.config_service = config_service
        self.data_service = data_service
        
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
        
        builder = FeatureBuilder(input_df=candles_df)
        if self.config_service.is_confidential == True:
            builder.read_features(self.config_service.features_path)
        else:
            builder.build_all()
        