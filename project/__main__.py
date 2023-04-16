from src.app.app_container import AppContainer

def run(app):
    if app["is_NN"] == True:
        pass
    elif app["is_random_forest"] == True:
        pass
    elif app["is_logistic_regression"] == True:
        pass

if __name__ == "__main__":
    run(app=AppContainer.config_service.app)