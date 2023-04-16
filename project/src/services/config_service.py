from src.singleton import Singleton

import yaml 

class ConfigService(metaclass=Singleton):
    def __init__(self, config_file):
        with open(config_file, "r") as config_yaml:
            self.config = yaml.safe_load(config_yaml)
    
    @property
    def application(self):
        return self.config["application"]
    
    @property
    def binance_data_url(self):
        return self.config["binance_data_url"]
    