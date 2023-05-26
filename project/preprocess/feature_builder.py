import pandas 

class FeatureBuilder():
    _must_columns = ["timestamp", "open", "high", "low", "close", "volume"]

    def __init__(self, input_df: pandas.DataFrame, features_path: pathlib.Path, target_features: list):
        """
        Args:
            input_df (pandas.DataFrame): A dataframe with timestamps and their corresponding OHLCV data
                                         must have columns: ["timestamp", "open", "high", "low", "close", "volume"]
        """
        if not set(self._must_columns).issubset(set(input_df.columns)):
            raise Exception("The input DataFrame must contain the columns [timestamp, open, high, low, close, volume]")

        self.candles_df = input_df
        self.features_path = features_path
        self.target_features = target_features
    
        self._ignore_columns = list(set(self.candles_df.columns) - set(self._must_columns))
    
    def build_all():
        ...
        
    # Features are to be kept confidential
    def read_features(self):
        features_df = pandas.read_csv(self.features_path)
        
        self.candles_df = self.candles_df.merge(features_df, on="timestamp")
        self.candles_df = self.candles_df[["timestamp"] + self.target_features + self._ignore_columns]
