from src.globals import Globals
from src.services.data_service import DataService
from src.services.config_service import ConfigService

class Preprocessor():
    def __init__(self, config_service: ConfigService, data_service: DataService):
        self.config_service = config_service
        self.data_service = data_service