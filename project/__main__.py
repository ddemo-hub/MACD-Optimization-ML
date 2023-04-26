from src.app.app_container import AppContainer
from src.utils.globals import Globals 
from src.utils.logger import Logger

Logger.set_logger_path(Globals.artifacts_path.joinpath("logs.txt"))

def run(app):
    if app.config_service.application["is_logistic_regression"] == True:
        app.logistic_regression_app.train()
    if app.config_service.application["is_k_nearest_neighbor"] == True:
        app.k_nearest_neighbor_app.train()
    if app.config_service.application["is_naive_bayes"] == True:
        pass

if __name__ == "__main__":
    run(app=AppContainer)
