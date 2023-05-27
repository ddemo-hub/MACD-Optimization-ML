from src.services.config_service import ConfigService

from preprocess.preprocessor import Preprocessor

from src.utils.globals import Globals

from abc import ABC, abstractmethod
import shutil
import os

class BaseApp(ABC):
    def __init__(self, config_service: ConfigService, preprocessor: Preprocessor):
        self.config_service = config_service
        self.preprocessor = preprocessor

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def breast_cancer(self):
        pass

    def save_configs(self):
        configs_path = Globals.project_path.joinpath("src", "configs")

        for config in os.listdir(configs_path):
            shutil.copyfile(configs_path.joinpath(config), Globals.artifacts_path.joinpath(f"{config[:-5]}.yamlignore"))    

    def run(self):
        if self.config_service.breast_cancer == True:
            self.breast_cancer()
        else:
            self.train()
