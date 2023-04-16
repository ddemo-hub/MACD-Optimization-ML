from src.globals import Globals
from src.singleton import Singleton
from src.services.data_service import DataService
from src.services.config_service import ConfigService

from preprocess.preprocessor import Preprocessor

from dataclasses import dataclass

@dataclass
class AppContainer(metaclass=Singleton):
    config_service = ConfigService(config_file=f"{Globals.project_path}/src/configs/config.yaml")
    
    data_service = DataService(config_service=ConfigService)
    
    preprocessor = Preprocessor(config_service=config_service, data_service=data_service)