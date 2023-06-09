from src.utils.singleton import Singleton

import pathlib
import yaml 

class ConfigService(metaclass=Singleton):
    def __init__(self, config: pathlib.Path, feature_config: pathlib.Path):
        # Read config files
        with open(config, "r") as config_file:
            self.config = yaml.safe_load(config_file)
        with open(feature_config, "r") as feature_config_file:
            self.feature_config = yaml.safe_load(feature_config_file)
    
    @property
    def application(self):
        return self.config["application"]

    @property
    def breast_cancer(self):
        return self.config["breast_cancer"]
    
    @property
    def binance_data_url(self):
        return self.config["binance_data_url"]
    
    @property
    def start_ts(self):
        return self.config["candle_parameters"]["start_ts"]
    
    @property
    def end_ts(self):
        return self.config["candle_parameters"]["end_ts"]
    
    @property
    def symbol(self):
        return self.config["candle_parameters"]["symbol"]
    
    @property
    def interval(self):
        return self.config["candle_parameters"]["interval"]
    
    @property
    def macd_fast(self):
        return self.config["labeling_parameters"]["macd_fast"]
    
    @property
    def macd_slow(self):
        return self.config["labeling_parameters"]["macd_slow"]
    
    @property
    def macd_signal(self):
        return self.config["labeling_parameters"]["macd_signal"]
    
    @property
    def standardization_window(self):
        return self.config["normalization_parameters"]["standardization_window"]
    
    @property
    def standardization_limit(self):
        return self.config["normalization_parameters"]["standardization_limit"]
    
    @property
    def training_size(self):
        return self.config["training_parameters"]["training_size"]

    @property
    def validation_size(self):
        return self.config["training_parameters"]["validation_size"]
    
    @property
    def test_size(self):
        return self.config["training_parameters"]["test_size"]
    
    @property
    def logistic_regression_learning_rate(self):
        return self.config["logistic_regression_parameters"]["learning_rate"]
    
    @property
    def logistic_regression_num_batch(self):
        return self.config["logistic_regression_parameters"]["num_batch"]
    
    @property
    def logistic_regression_num_epochs(self):
        return self.config["logistic_regression_parameters"]["num_epochs"]
    
    @property
    def logistic_regression_threshold(self):
        return self.config["logistic_regression_parameters"]["threshold"]
    
    @property
    def logistic_regression_stochastic(self):
        return self.config["logistic_regression_parameters"]["stochastic"]
    
    @property
    def logistic_regression_regularization(self):
        return self.config["logistic_regression_parameters"]["regularization"]

    @property
    def logistic_regression_constant(self):
        return self.config["logistic_regression_parameters"]["constant"]
       
    @property
    def logistic_regression_class_weights(self):
        return self.config["logistic_regression_parameters"]["class_weights"]
       
    @property
    def logistic_regression_transformation(self):
        return self.config["logistic_regression_parameters"]["transformation"]
       
    @property
    def k_nearest_neighnors_k(self):
        return self.config["k_nearest_neighbors_parameters"]["k"]

    @property
    def k_nearest_neighnors_similarity_measure(self):
        return self.config["k_nearest_neighbors_parameters"]["similarity_measure"]

    @property
    def k_nearest_neighnors_undersample(self):
        return self.config["k_nearest_neighbors_parameters"]["undersample"]

    @property
    def k_nearest_neighnors_oversample(self):
        return self.config["k_nearest_neighbors_parameters"]["oversample"]
       
    @property
    def naive_bayes_method(self):
        return self.config["naive_bayes_parameters"]["method"]

    @property
    def is_confidential(self):
        return self.feature_config["is_confidential"]
    
    @property
    def categorize_features(self):
        return self.feature_config["categorize_features"]
    
    @property
    def target_features(self):
        return self.feature_config["target_features"]