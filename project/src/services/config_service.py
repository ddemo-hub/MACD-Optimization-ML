from src.utils.singleton import Singleton

import yaml 

class ConfigService(metaclass=Singleton):
    def __init__(self, config: str, feature_config: str):
        with open(config, "r") as config_file:
            self.config = yaml.safe_load(config_file)
        with open(feature_config, "r") as feature_config_file:
            self.feature_config = yaml.safe_load(feature_config_file)
    
    @property
    def application(self):
        return self.config["application"]
    
    @property
    def binance_data_url(self):
        return self.config["binance_data_url"]
    
    @property
    def is_confidential(self):
        return self.feature_config["is_confidential"]
    
    @property
    def features_path(self):
        return self.feature_config["features_path"]