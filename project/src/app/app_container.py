from src.utils.globals import Globals
from src.utils.singleton import Singleton
from src.services.data_service import DataService
from src.services.config_service import ConfigService

from preprocess.preprocessor import Preprocessor

from .logistic_regression_app import LogisticRegressionApp
from .naive_bayes_app import NaiveBayesApp
from .k_nearest_neighbor_app import KNearestNeighborApp

from dataclasses import dataclass

@dataclass
class AppContainer(metaclass=Singleton):
    config_service = ConfigService(
        config=Globals.project_path.joinpath("src", "configs", "config.yaml"),
        feature_config=Globals.project_path.joinpath("src", "configs", "feature_config.yaml")
    )
    
    data_service = DataService(config_service=config_service)
    
    preprocessor = Preprocessor(config_service=config_service, data_service=data_service)
    
    logistic_regression_app = LogisticRegressionApp(
        config_service=config_service, 
        preprocessor=preprocessor
    )
    
    k_nearest_neighbor_app = KNearestNeighborApp(
        config_service=config_service,
        preprocessor=preprocessor
    )