import pandas 

class FeatureBuilder():
    def __init__(self, input_df: pandas.DataFrame):
        """
        Args:
            input_df (pandas.DataFrame): A dataframe with timestamps and their corresponding OHLCV data
        """
        self.candles_df = input_df
    
    def build_all():
        ...
        
    # Features are to be kept confidential
    def read_features(self, features_path):
        features_df = pandas.read_csv(features_path)
        
        self.candles_df.merge(features_df, on="timestamp")