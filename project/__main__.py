from src.app.app_container import AppContainer
from src.utils.globals import Globals 
from src.utils.logger import Logger

Logger.set_logger_path(Globals.artifacts_path.joinpath("logs.txt"))

def run(app):
    if app.config_service.application["is_logistic_regression"] == True:
        app.logistic_regression_app.run()
    if app.config_service.application["is_k_nearest_neighbors"] == True:
        app.k_nearest_neighbor_app.run()
    if app.config_service.application["is_naive_bayes"] == True:
        app.naive_bayes_app.run()

if __name__ == "__main__":
    run(app=AppContainer)
