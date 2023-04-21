from src.utils.globals import Globals
from src.utils.singleton import Singleton
from src.services.data_service import DataService
from src.services.config_service import ConfigService

from preprocess.preprocessor import Preprocessor

from .logistic_regression_app import LogisticRegressionApp
from .random_forest_app import RandomForestApp
from .nn_app import NeuralNetworkApp

from dataclasses import dataclass

@dataclass
class AppContainer(metaclass=Singleton):
    config_service = ConfigService(
        config=f"{Globals.project_path}/src/configs/config.yaml",
        feature_config=f"{Globals.project_path}/src/configs/feature_config.yaml"
    )
    
    data_service = DataService(config_service=ConfigService)
    
    preprocessor = Preprocessor(config_service=config_service, data_service=data_service)
    
    logistic_regression_app = LogisticRegressionApp(
        config_service=config_service, 
        data_service=data_service, 
        preprocessor=preprocessor
    )
    