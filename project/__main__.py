from src.app.app_container import AppContainer
from src.utils.globals import Globals 
from src.utils.logger import Logger

Logger.set_logger_path(f"{Globals.artifacts_path}/logs.txt")

def run(app):
    if app.config_service.application["is_k_nearest_neighbor"] == True:
        app.k_nearest_neighbor_app.train()
    elif app.config_service.application["is_random_forest"] == True:
        pass
    elif app.config_service.application["is_logistic_regression"] == True:
        app.logistic_regression_app.train()

if __name__ == "__main__":
    run(app=AppContainer)
