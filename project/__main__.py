from src.app.app_container import AppContainer

def run(app):
    if app.config_service.application["is_NN"] == True:
        pass
    elif app.config_service.application["is_random_forest"] == True:
        pass
    elif app.config_service.application["is_logistic_regression"] == True:
        pass

if __name__ == "__main__":
    run(app=AppContainer)